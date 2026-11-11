use super::{
    ChannelError, ChannelWaterLedger,
    context::Context,
    engine::Accepted,
    sources::{LateralState, PreparedLateral, SourceApplication},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) struct Balance {
    pub ledger: ChannelWaterLedger,
    pub incoming: f64,
    pub outgoing: f64,
    pub residual: f64,
    pub max_local: f64,
    pub max_momentum: f64,
    pub transfer_charge: f64,
}

pub(super) struct Impulses<'a, R: Runtime<DType = DType>> {
    pub water: &'a Tensor<R>,
    pub momentum: &'a Tensor<R>,
    pub left_pressure: &'a Tensor<R>,
    pub right_pressure: &'a Tensor<R>,
    pub friction: &'a Tensor<R>,
    pub momentum_underflow_bound: &'a Tensor<R>,
    pub transfer_underflow_bound: &'a Tensor<R>,
    pub lateral: &'a PreparedLateral<R>,
    pub application: &'a SourceApplication<R>,
    pub next_lateral: &'a LateralState<R>,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub fn balance(
        &self,
        previous: &Accepted<R>,
        volume: &Tensor<R>,
        momentum: &Tensor<R>,
        impulses: Impulses<'_, R>,
    ) -> Result<Balance, ChannelError> {
        let c = &self.client;
        let count = self.count();
        let left_volume = impulses.water.narrow(0, 0, count)?;
        let right_volume = impulses.water.narrow(0, 1, count)?;
        let local = c.sub(
            &c.sub(volume, &previous.volume)?,
            &c.sub(&left_volume, &right_volume)?,
        )?;
        let local = if impulses.lateral.has_nonzero_values {
            c.sub(
                &local,
                &c.sub(
                    &impulses.lateral.incoming_volume,
                    &impulses.lateral.withdrawal_volume,
                )?,
            )?
        } else {
            local
        };
        self.finite(&local, "local water residual")?;
        let max_local = c.max(&c.abs(&local)?, &[0], false)?.item::<f64>()?;
        let local_transfer_bound = self.bound_add(
            &impulses.transfer_underflow_bound.narrow(0, 0, count)?,
            &impulses.transfer_underflow_bound.narrow(0, 1, count)?,
        )?;
        let reported_charge = self.total(&local_transfer_bound)?;
        let charge_error = self
            .charge_reduction
            .unit_sum_error_bound(reported_charge)
            .map_err(|error| ChannelError::numerical(error.to_string()))?;
        let charge_add = |current, step| {
            crate::soil::admission::add_adjustment_charge(current, step)
                .map_err(|error| ChannelError::numerical(error.to_string()))
        };
        let transfer_charge = charge_add(reported_charge, charge_error)?;
        let cumulative_transfer_charge = charge_add(
            previous.ledger.numerical_transfer_underflow_bound_m3,
            transfer_charge,
        )?;
        let left_momentum = c.add(
            &impulses.momentum.narrow(0, 0, count)?,
            &impulses.right_pressure.narrow(0, 0, count)?,
        )?;
        let right_momentum = c.add(
            &impulses.momentum.narrow(0, 1, count)?,
            &impulses.left_pressure.narrow(0, 1, count)?,
        )?;
        let local_momentum = c.sub(
            &c.sub(
                &c.sub(momentum, &previous.momentum)?,
                &c.sub(&left_momentum, &right_momentum)?,
            )?,
            impulses.friction,
        )?;
        let local_momentum = if impulses.lateral.has_nonzero_values {
            c.sub(
                &local_momentum,
                &c.sub(
                    &impulses.lateral.incoming_momentum,
                    &impulses.lateral.withdrawal_momentum,
                )?,
            )?
        } else {
            local_momentum
        };
        self.finite(&local_momentum, "local momentum residual")?;
        let max_momentum = c
            .max(&c.abs(&local_momentum)?, &[0], false)?
            .item::<f64>()?;
        let scale = c.maximum(&c.abs(&previous.momentum)?, &c.abs(momentum)?)?;
        let scale = c.maximum(&scale, &c.abs(&left_momentum)?)?;
        let scale = c.maximum(&scale, &c.abs(&right_momentum)?)?;
        let scale = c.maximum(&scale, &c.abs(impulses.friction)?)?;
        let scale = if impulses.lateral.has_nonzero_values {
            c.maximum(
                &c.maximum(&scale, &c.abs(&impulses.lateral.incoming_momentum)?)?,
                &c.abs(&impulses.lateral.withdrawal_momentum)?,
            )?
        } else {
            scale
        };
        let tolerance = c.add_scalar(
            &c.mul_scalar(&scale, self.definition.config.momentum_relative)?,
            self.definition.config.momentum_absolute,
        )?;
        self.finite(&tolerance, "momentum tolerance")?;
        let charged_momentum =
            self.bound_add(&c.abs(&local_momentum)?, impulses.momentum_underflow_bound)?;
        let charged_momentum = self.bound_add(
            &charged_momentum,
            &impulses.application.momentum_assembly_bound,
        )?;
        if !self.all(&c.le(&charged_momentum, &tolerance)?)? {
            return Err(ChannelError::numerical(format!(
                "local momentum residual={max_momentum} plus detected underflow bound exceeds tolerance: rescale inputs or use a representable interval"
            )));
        }
        let left = impulses.water.narrow(0, 0, 1)?.item::<f64>()?;
        let right = impulses.water.narrow(0, count, 1)?.item::<f64>()?;
        let incoming = left.max(0.0) + (-right).max(0.0);
        let outgoing = (-left).max(0.0) + right.max(0.0);
        let storage = self.total(volume)?;
        let lateral = &impulses.lateral.summary;
        let residual = if impulses.lateral.has_nonzero_values {
            (storage - previous.ledger.storage_m3)
                - ((incoming - outgoing) + (lateral.incoming_m3 - lateral.withdrawal_m3))
        } else {
            (storage - previous.ledger.storage_m3) - (incoming - outgoing)
        };
        let totals = self.lateral.totals(c, impulses.next_lateral)?;
        let ledger = ChannelWaterLedger {
            initial_storage_m3: previous.ledger.initial_storage_m3,
            storage_m3: storage,
            boundary_in_m3: previous.ledger.boundary_in_m3 + incoming,
            boundary_out_m3: previous.ledger.boundary_out_m3 + outgoing,
            lateral_in_m3: totals.incoming_m3,
            lateral_out_m3: totals.withdrawal_m3,
            lateral_counter_abs_bound_m3: totals.counter_abs_bound_m3,
            lateral_assembly_abs_bound_m3: totals.assembly_abs_bound_m3,
            residual_m3: if totals.incoming_m3 != 0.0 || totals.withdrawal_m3 != 0.0 {
                (storage - previous.ledger.initial_storage_m3)
                    - (((previous.ledger.boundary_in_m3 + incoming)
                        - (previous.ledger.boundary_out_m3 + outgoing))
                        + (totals.incoming_m3 - totals.withdrawal_m3))
            } else {
                (storage - previous.ledger.initial_storage_m3)
                    - ((previous.ledger.boundary_in_m3 + incoming)
                        - (previous.ledger.boundary_out_m3 + outgoing))
            },
            numerical_transfer_underflow_bound_m3: cumulative_transfer_charge,
        };
        let interval_scale = storage
            .max(previous.ledger.storage_m3)
            .max(incoming)
            .max(outgoing)
            .max(lateral.incoming_m3)
            .max(lateral.withdrawal_m3);
        let lifetime_scale = storage
            .max(ledger.initial_storage_m3)
            .max(ledger.boundary_in_m3)
            .max(ledger.boundary_out_m3)
            .max(ledger.lateral_in_m3)
            .max(ledger.lateral_out_m3);
        let interval_tolerance = self.definition.config.volume_absolute
            + self.definition.config.volume_relative * interval_scale;
        let lifetime_tolerance = self.definition.config.volume_absolute
            + self.definition.config.volume_relative * lifetime_scale;
        let charged_local = self.bound_add(&c.abs(&local)?, &local_transfer_bound)?;
        let charged_local =
            self.bound_add(&charged_local, &impulses.application.water_assembly_bound)?;
        let charged_local_admitted =
            self.all(&c.le(&charged_local, &self.constant(&local, interval_tolerance)?)?)?;
        let charged_interval = charge_add(residual.abs(), transfer_charge)?;
        let charged_interval = charge_add(
            charged_interval,
            impulses.application.water_assembly_charge_m3,
        )?;
        let charged_lifetime = charge_add(ledger.residual_m3.abs(), cumulative_transfer_charge)?;
        let charged_lifetime = charge_add(charged_lifetime, ledger.lateral_counter_abs_bound_m3)?;
        let charged_lifetime = charge_add(charged_lifetime, ledger.lateral_assembly_abs_bound_m3)?;
        if ![
            incoming,
            outgoing,
            residual,
            ledger.boundary_in_m3,
            ledger.boundary_out_m3,
            ledger.residual_m3,
            ledger.lateral_in_m3,
            ledger.lateral_out_m3,
            ledger.lateral_counter_abs_bound_m3,
            ledger.lateral_assembly_abs_bound_m3,
            interval_tolerance,
            lifetime_tolerance,
            transfer_charge,
            cumulative_transfer_charge,
            charged_interval,
            charged_lifetime,
        ]
        .iter()
        .all(|x| x.is_finite())
            || charged_interval > interval_tolerance
            || !charged_local_admitted
            || charged_lifetime > lifetime_tolerance
        {
            return Err(ChannelError::numerical(format!(
                "channel water residual={residual}, local={max_local}, lifetime={}, interval_tolerance={interval_tolerance}, lifetime_tolerance={lifetime_tolerance}: inspect water accounting",
                ledger.residual_m3
            )));
        }
        Ok(Balance {
            ledger,
            incoming,
            outgoing,
            residual,
            max_local,
            max_momentum,
            transfer_charge,
        })
    }
}
