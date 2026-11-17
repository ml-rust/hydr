//! Private coordinator reads: accepted-start donor reservations, cell stages, and coordinated reaches.

use super::{
    ChannelError, ChannelInitialState, ChannelReach, ChannelReachDefinition, context::Context,
    coupling::ChannelCouplingForcing, interval_start::ReservedStart,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

/// Accepted-start donor operands, from the tensors and functions the reach's own step uses.
///
/// Every tensor is F64 with one value per cell, in cell order.
#[derive(Debug)]
#[doc(hidden)]
pub struct ChannelCouplingReservation<R: Runtime<DType = DType>> {
    /// Water each cell can donate, in m³. Surveyed geometry uses the two-part lower bound.
    pub donor_volume_m3: Tensor<R>,
    /// Native shared-face outflow from each cell, in m³/s.
    pub outgoing_rate_m3_per_s: Tensor<R>,
    /// Accepted-start longitudinal velocity, in m/s. Lateral withdrawals leave at this velocity.
    pub velocity_m_per_s: Tensor<R>,
    /// Accepted-start CFL limit, in seconds.
    pub start_cfl_s: f64,
    /// Longest span whose face outflow alone exhausts no donor, in seconds.
    pub face_depletion_limit_s: f64,
    /// Accepted-start operands that [`ChannelReach::bind_coupling_reserved`] forcing reuses.
    pub(super) reserved: Arc<ReservedStart<R>>,
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    /// Construct a reach whose lateral bindings can use the reserved surface-exchange prefix.
    ///
    /// Only the surface-channel coordinator calls this. Every other definition check applies.
    #[doc(hidden)]
    pub fn new_coordinated(
        client: R::Client,
        definition: ChannelReachDefinition,
        initial: ChannelInitialState,
    ) -> Result<Self, ChannelError>
    where
        R::Client: HypotOps<R> + TensorOps<R> + NextRepresentableOps<R>,
    {
        let context = Arc::new(Context::with_binding_owner(
            client, definition, initial, true,
        )?);
        Self::from_context(Arc::clone(&context), &context.original_initial)
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// Accepted-start donor operands for a forcing with external policies at both endpoints.
    ///
    /// The face depletion limit is the step's own check when the interval has no nonzero lateral input.
    #[doc(hidden)]
    pub fn coupling_reservation(
        &self,
        forcing: &ChannelCouplingForcing<R>,
    ) -> Result<ChannelCouplingReservation<R>, ChannelError> {
        if !Arc::ptr_eq(&self.context.definition, &forcing.lateral.definition)
            && !self.definition().same_bits(&forcing.lateral.definition)
        {
            return Err(ChannelError::input(
                "coupling reservation forcing belongs to different geometry or source bindings: bind the forcing with this reach",
            ));
        }
        if forcing.start_s.to_bits() != self.time().to_bits() {
            return Err(ChannelError::input(format!(
                "coupling reservation forcing start={}, accepted_time={}: bind the forcing at the accepted time",
                forcing.start_s,
                self.time()
            )));
        }
        let (Some(left), Some(right)) = (&forcing.left, &forcing.right) else {
            return Err(ChannelError::input(format!(
                "coupling reservation endpoints: left external={}, right external={}. Connected pool endpoints are unsupported: supply external policies at both endpoints",
                forcing.left.is_some(),
                forcing.right.is_some()
            )));
        };
        let start = self.context.interval_start(
            &self.accepted.volume,
            &self.accepted.momentum,
            left,
            right,
        )?;
        let face_depletion_limit_s = self
            .context
            .face_depletion_limit(&start.donor, &start.outgoing)?;
        Ok(ChannelCouplingReservation {
            donor_volume_m3: start.donor.clone(),
            outgoing_rate_m3_per_s: start.outgoing.clone(),
            velocity_m_per_s: start.fields.velocity.clone(),
            start_cfl_s: start.start_cfl,
            face_depletion_limit_s,
            reserved: Arc::new(ReservedStart::new(
                start,
                &self.accepted,
                left.clone(),
                right.clone(),
            )),
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: TensorOps<R>
        + CompareOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>,
{
    /// Accepted water-surface elevation and F64 wet mask for `cells`, in request order.
    ///
    /// Stage is bed plus recovered depth, as `state()` reports. Dry cells carry the bed elevation.
    /// Wet is `1.0` when the stored volume is positive, otherwise `0.0`.
    #[doc(hidden)]
    pub fn coupling_cell_stages(
        &self,
        cells: &[usize],
    ) -> Result<(Tensor<R>, Tensor<R>), ChannelError> {
        let c = &self.context.client;
        let count = self.context.count();
        let mut indices = Vec::with_capacity(cells.len());
        for &cell in cells {
            let index = i64::try_from(cell).ok().filter(|_| cell < count);
            indices.push(index.ok_or_else(|| {
                ChannelError::input(format!(
                    "coupling stage cell={cell}, cells={count}: request existing cell indices"
                ))
            })?);
        }
        if indices.is_empty() {
            let empty = Tensor::<R>::from_slice(&[0.0_f64; 0], &[0], c.device())?;
            return Ok((empty.clone(), empty));
        }
        let fields = self
            .context
            .fields(&self.accepted.volume, &self.accepted.momentum)?;
        let volume = &self.accepted.volume.high;
        let stage = c.add(&self.context.bed, &fields.depth)?;
        let wet = c.gt(volume, &self.context.constant(volume, 0.0)?)?;
        let index = Tensor::<R>::from_slice(&indices, &[indices.len()], c.device())?;
        Ok((
            c.index_select(&stage, 0, &index)?,
            c.index_select(&wet, 0, &index)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry,
        ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
        RectangularChannelCell,
        coupling::{ChannelCouplingTrial, SURFACE_EXCHANGE_BINDING_PREFIX},
    };
    use super::*;
    use crate::forcing::WaterAmount;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    fn client() -> CpuClient {
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn definition(bindings: Vec<ChannelLateralBinding>) -> ChannelReachDefinition {
        let cell = |id: &str, bed: f64| RectangularChannelCell {
            id: id.into(),
            length_m: 10.0,
            width_m: 2.0,
            bed_elevation_m: bed,
            maximum_depth_m: 5.0,
            manning_n: 0.0,
        };
        ChannelReachDefinition {
            vertical_datum: "synthetic datum".into(),
            provenance: "coupling reservation fixture".into(),
            geometry: ChannelGeometry::Rectangular {
                cells: vec![
                    cell("a", 1.0),
                    cell("b", 0.75),
                    cell("c", 0.5),
                    cell("d", 0.25),
                ],
            },
            lateral_bindings: bindings,
            config: ChannelConfig {
                max_step: 1.0e6,
                ..ChannelConfig::default()
            },
        }
    }

    fn initial() -> ChannelInitialState {
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![40.0, 25.0, 12.0, 0.0],
            longitudinal_momentum_m4_per_s: vec![4.0, 10.0, -1.5, 0.0],
        }
    }

    fn exchange_binding() -> ChannelLateralBinding {
        ChannelLateralBinding {
            boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}b"),
            cell_id: "b".into(),
            direction: ChannelLateralDirection::Withdrawal,
            ownership_assertion: "coordinator surface exchange".into(),
        }
    }

    /// Parse `name=value` from a step's shortening reason, keeping every bit.
    fn reason_value(reason: &str, name: &str) -> Result<f64, Box<dyn std::error::Error>> {
        let tail = reason
            .split_once(name)
            .ok_or_else(|| format!("reason {reason:?} lacks {name:?}"))?
            .1;
        let value = tail.split(',').next().unwrap_or(tail).trim();
        Ok(value.parse::<f64>()?)
    }

    #[test]
    fn channel_reservation_matches_internal_depletion() -> TestResult {
        let reach = ChannelReach::<CpuRuntime>::new(client(), definition(vec![]), initial())?;
        let wall = ChannelBoundary::Wall;
        let forcing = reach.bind_coupling_host(0.0, 1.0e6, Some(&wall), Some(&wall), &[])?;
        let reservation = reach.coupling_reservation(&forcing)?;
        let donor = reservation.donor_volume_m3.try_to_vec::<f64>()?;
        let stored = reach
            .coupling_accounting()
            .cell_volume_m3
            .try_to_vec::<f64>()?;
        assert_eq!(
            donor.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            stored.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let outgoing = reservation.outgoing_rate_m3_per_s.try_to_vec::<f64>()?;
        let host_limit = donor
            .iter()
            .zip(&outgoing)
            .filter(|(_, rate)| **rate > 0.0)
            .map(|(volume, rate)| volume / rate)
            .fold(f64::INFINITY, f64::min);
        assert!(host_limit.is_finite());
        assert_eq!(
            reservation.face_depletion_limit_s.to_bits(),
            host_limit.to_bits()
        );
        let velocity = reservation.velocity_m_per_s.try_to_vec::<f64>()?;
        for (cell, result) in reach.state()?.iter().enumerate() {
            if let Some(expected) = result.velocity_m_per_s {
                assert_eq!(velocity[cell].to_bits(), expected.to_bits());
            }
        }

        // The step's own shortening reports its accepted-start CFL and depletion limits.
        let limit = reservation
            .start_cfl_s
            .min(reservation.face_depletion_limit_s);
        match reach.trial_coupled(1.0e6, &forcing, None, None)? {
            ChannelCouplingTrial::Shorter {
                suggested_end_s,
                reason,
            } => {
                let cfl = reason_value(&reason, "accepted-start CFL=")?;
                let depletion = reason_value(&reason, "original-donor depletion=")?;
                assert_eq!(cfl.to_bits(), reservation.start_cfl_s.to_bits());
                assert_eq!(
                    depletion.to_bits(),
                    reservation.face_depletion_limit_s.to_bits()
                );
                assert_eq!(suggested_end_s.to_bits(), (0.95 * limit).to_bits());
            }
            ChannelCouplingTrial::Accepted(_) => {
                return Err("an interval past both limits must shorten".into());
            }
        }
        let end = 0.5 * limit;
        let within = reach.bind_coupling_host(0.0, end, Some(&wall), Some(&wall), &[])?;
        assert!(matches!(
            reach.trial_coupled(end, &within, None, None)?,
            ChannelCouplingTrial::Accepted(_)
        ));

        let pooled = reach.bind_coupling_host(0.0, 1.0e6, None, Some(&wall), &[])?;
        let error = reach
            .coupling_reservation(&pooled)
            .err()
            .ok_or("a pool endpoint must reject")?;
        assert!(
            error
                .to_string()
                .contains("Connected pool endpoints are unsupported")
        );
        Ok(())
    }

    #[test]
    fn reserved_trial_matches_unreserved_bits() -> TestResult {
        let reach = ChannelReach::<CpuRuntime>::new_coordinated(
            client(),
            definition(vec![exchange_binding()]),
            initial(),
        )?;
        let wall = ChannelBoundary::Wall;
        let withdrawal = ChannelLateralInput::Withdrawal {
            boundary_id: exchange_binding().boundary_id,
            amount: WaterAmount::IntervalVolume { volume_m3: 0.5 },
        };
        let end = 0.05;
        let plain = reach.bind_coupling_host(0.0, end, Some(&wall), Some(&wall), &[withdrawal])?;
        let reservation = reach.coupling_reservation(&plain)?;
        let reserved = reach.bind_coupling_reserved(&reservation, end, &plain.lateral.original)?;
        // Debug text shows every bit of finite values.
        let bits = |trial| -> Result<String, Box<dyn std::error::Error>> {
            match trial {
                ChannelCouplingTrial::<CpuRuntime>::Accepted(next) => {
                    Ok(format!("{:?} {:?}", next.state()?, next.snapshot_data()?))
                }
                ChannelCouplingTrial::Shorter { reason, .. } => Err(reason.into()),
            }
        };
        assert_eq!(
            bits(reach.trial_coupled(end, &reserved, None, None)?)?,
            bits(reach.trial_coupled(end, &plain, None, None)?)?
        );
        let other = ChannelReach::<CpuRuntime>::new_coordinated(
            client(),
            definition(vec![exchange_binding()]),
            initial(),
        )?;
        let error = other
            .bind_coupling_reserved(&reservation, end, &[])
            .err()
            .ok_or("a reservation from another reach must reject")?;
        assert!(error.to_string().contains("another accepted state"));
        Ok(())
    }

    #[test]
    fn channel_cell_stages_match_output_stage() -> TestResult {
        let reach = ChannelReach::<CpuRuntime>::new(client(), definition(vec![]), initial())?;
        let cells = [3, 0, 2, 2];
        let (stage, wet) = reach.coupling_cell_stages(&cells)?;
        let stage = stage.try_to_vec::<f64>()?;
        let wet = wet.try_to_vec::<f64>()?;
        let state = reach.state()?;
        for (slot, &cell) in cells.iter().enumerate() {
            let result = &state[cell];
            assert_eq!(wet[slot], if result.dry { 0.0 } else { 1.0 });
            match result.stage_m {
                Some(expected) => assert_eq!(stage[slot].to_bits(), expected.to_bits()),
                None => assert_eq!(
                    stage[slot].to_bits(),
                    reach
                        .definition()
                        .geometry
                        .cell(cell)
                        .bed_elevation_m
                        .to_bits()
                ),
            }
        }
        assert!(state[3].dry);
        let (empty_stage, empty_wet) = reach.coupling_cell_stages(&[])?;
        assert_eq!((empty_stage.numel(), empty_wet.numel()), (0, 0));
        let error = reach
            .coupling_cell_stages(&[4])
            .err()
            .ok_or("a missing cell must reject")?;
        assert!(error.to_string().contains("cell=4, cells=4"));
        Ok(())
    }

    #[test]
    fn reserved_exchange_prefix_rejected() -> TestResult {
        let error = ChannelReach::<CpuRuntime>::new(
            client(),
            definition(vec![exchange_binding()]),
            initial(),
        )
        .err()
        .ok_or("a caller binding with the reserved prefix must reject")?;
        assert!(error.to_string().contains(SURFACE_EXCHANGE_BINDING_PREFIX));
        assert!(error.to_string().contains("rename the lateral binding"));

        let caller = ChannelReach::<CpuRuntime>::new(client(), definition(vec![]), initial())?;
        let reserved_input = ChannelLateralInput::Withdrawal {
            boundary_id: exchange_binding().boundary_id,
            amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
        };
        let error = caller
            .step(
                1.0,
                &ChannelForcingInterval {
                    start_s: 0.0,
                    end_s: 1.0,
                    left: ChannelBoundary::Wall,
                    right: ChannelBoundary::Wall,
                    lateral_inputs: vec![reserved_input.clone()],
                },
            )
            .err()
            .ok_or("caller forcing with the reserved prefix must reject")?;
        assert!(error.to_string().contains(SURFACE_EXCHANGE_BINDING_PREFIX));

        let coordinated = ChannelReach::<CpuRuntime>::new_coordinated(
            client(),
            definition(vec![exchange_binding()]),
            initial(),
        )?;
        let forcing = coordinated.bind_coupling_host(
            0.0,
            1.0e6,
            Some(&ChannelBoundary::Wall),
            Some(&ChannelBoundary::Wall),
            &[reserved_input],
        )?;
        coordinated.coupling_reservation(&forcing)?;
        assert_eq!(coordinated.lateral_ledgers()?.len(), 1);
        Ok(())
    }
}
