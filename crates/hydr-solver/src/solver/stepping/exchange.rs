//! Transport update and the coordinator exchange source applied before friction.

use super::super::Solver;
use super::super::coupling::ExchangeSource;
use super::super::receipts::ExchangeDepths;
use super::super::tensor::TensorState;
use super::super::tensor::transport::TensorTransport;
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Relative rounding allowance on a donor's post-transport volume, in units of `f64::EPSILON`.
const DONOR_ROUNDING_ULPS: f64 = 4.0;

/// Accounting of one applied exchange source.
pub(in crate::solver) struct ExchangeOutcome<R: Runtime<DType = DType>> {
    /// Full-grid inflow minus withdrawal volume in m³, zero outside the linked cells.
    pub net_volume: Tensor<R>,
    /// Full-grid exchange depths for the accepted receipt.
    pub depths: ExchangeDepths<R>,
    /// Summed coordinator inflow volume in m³.
    pub inflow_volume: f64,
    /// Summed coordinator withdrawal volume in m³.
    pub withdrawal_volume: f64,
    /// Largest per-cell gap between applied depth change times area and net volume, in m³.
    pub representation_error: f64,
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Advances the post-source state by the transport derivative over `dt`.
    ///
    /// The coordinator stage and the completion both call this, so their depths agree bit for bit.
    pub(in crate::solver) fn transported(
        &self,
        state: &TensorState<R>,
        flux: &TensorTransport<R>,
        dt: f64,
    ) -> Result<TensorState<R>, EngineError> {
        let client = &self.context.client;
        Ok(TensorState {
            h: client.add(&state.h, &client.mul_scalar(&flux.derivative.h, dt)?)?,
            qx: client.add(&state.qx, &client.mul_scalar(&flux.derivative.qx, dt)?)?,
            qy: client.add(&state.qy, &client.mul_scalar(&flux.derivative.qy, dt)?)?,
        })
    }

    /// Adds inflow and removes withdrawal at the linked cells of a nonnegative transported state.
    ///
    /// Returns the post-exchange state before friction, with its accounting.
    ///
    /// Depth gains `(inflow - withdrawal) / area`, floored at zero.
    /// Withdrawal keeps the fraction `(h - withdrawal / area) / h` of the donor discharge.
    /// Inflow adds `inflow / area` times its velocity. Every other cell keeps its bits.
    ///
    /// # Errors
    /// Rejects nonfinite or negative operands, an underflowing exchange depth, and withdrawal
    /// beyond the post-transport volume plus the rounding allowance.
    pub(in crate::solver) fn apply_exchange(
        &self,
        transported: TensorState<R>,
        exchange: &ExchangeSource<R>,
    ) -> Result<(TensorState<R>, ExchangeOutcome<R>), EngineError> {
        let client = &self.context.client;
        let zero = &self.context.zero;
        let area = self.domain().grid().cell_area();
        let links = &exchange.links;
        links.check_grid(self.domain().grid().cell_count())?;
        let inflow = &exchange.inflow_volume_m3;
        let withdrawal = &exchange.withdrawal_volume_m3;
        let [velocity_x, velocity_y] = &exchange.inflow_velocity_xy;
        self.finite(inflow, "surface exchange inflow volume")?;
        self.finite(withdrawal, "surface exchange withdrawal volume")?;
        self.finite(velocity_x, "surface exchange inflow velocity x")?;
        self.finite(velocity_y, "surface exchange inflow velocity y")?;
        let lane_zero = client.mul_scalar(inflow, 0.0)?;
        if self.any(&client.maximum(
            &client.lt(inflow, &lane_zero)?,
            &client.lt(withdrawal, &lane_zero)?,
        )?)? {
            return Err(EngineError::rejected(
                "surface exchange",
                format!(
                    "time={}: a linked inflow or withdrawal volume is negative; supply volumes of at least zero",
                    self.time()
                ),
            ));
        }
        let put = |values: &Tensor<R>| client.index_put(zero, 0, &links.ids, values);
        let linked = client.gt(&put(&client.add_scalar(&lane_zero, 1.0)?)?, zero)?;
        let inflow = put(inflow)?;
        let withdrawal = put(withdrawal)?;
        let inflow_depth = client.div_scalar(&inflow, area)?;
        let withdrawal_depth = client.div_scalar(&withdrawal, area)?;
        let underflow = client.maximum(
            &client.mul(&client.gt(&inflow, zero)?, &client.eq(&inflow_depth, zero)?)?,
            &client.mul(
                &client.gt(&withdrawal, zero)?,
                &client.eq(&withdrawal_depth, zero)?,
            )?,
        )?;
        if self.any(&underflow)? {
            return Err(EngineError::rejected(
                "surface exchange",
                format!(
                    "time={}, area={area}: a positive exchange volume over cell area underflows; rescale the exchange or grid",
                    self.time()
                ),
            ));
        }
        let h = &transported.h;
        let available = client.mul_scalar(h, area)?;
        let allowance = client.mul_scalar(&available, 1.0 + DONOR_ROUNDING_ULPS * f64::EPSILON)?;
        let excess = client.gt(&withdrawal, &allowance)?;
        if self.any(&excess)? {
            let excess = excess.try_to_vec::<f64>()?;
            let cell = excess.iter().position(|value| *value != 0.0);
            return Err(EngineError::rejected(
                "surface exchange",
                format!(
                    "time={}, cell={cell:?}: withdrawal exceeds the post-transport volume; limit the coordinator donor withdrawal to the stage available volume",
                    self.time()
                ),
            ));
        }
        let net_volume = client.sub(&inflow, &withdrawal)?;
        let raised = client.maximum(
            &client.add(h, &client.div_scalar(&net_volume, area)?)?,
            zero,
        )?;
        let next_h = client.where_cond(&linked, &raised, h)?;
        let one = client.add_scalar(zero, 1.0)?;
        let wet = client.gt(h, zero)?;
        let donor = client.where_cond(&wet, h, &one)?;
        let kept = client.maximum(
            &client.div(&client.sub(h, &withdrawal_depth)?, &donor)?,
            zero,
        )?;
        let kept = client.where_cond(&wet, &kept, zero)?;
        let next_wet = client.gt(&next_h, zero)?;
        let discharge = |q: &Tensor<R>, velocity: &Tensor<R>| -> Result<Tensor<R>, EngineError> {
            let carried = client.mul(&inflow_depth, &put(velocity)?)?;
            let exchanged = client.add(&client.mul(q, &kept)?, &carried)?;
            let exchanged = client.where_cond(&next_wet, &exchanged, zero)?;
            Ok(client.where_cond(&linked, &exchanged, q)?)
        };
        let next_qx = discharge(&transported.qx, velocity_x)?;
        let next_qy = discharge(&transported.qy, velocity_y)?;
        let gap = client.sub(
            &client.mul_scalar(&client.sub(&next_h, h)?, area)?,
            &net_volume,
        )?;
        let representation_error = client.max(&client.abs(&gap)?, &[0], false)?.item::<f64>()?;
        let inflow_volume = self.total(&exchange.inflow_volume_m3)?;
        let withdrawal_volume = self.total(&exchange.withdrawal_volume_m3)?;
        if ![inflow_volume, withdrawal_volume, representation_error]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err(EngineError::rejected(
                "surface exchange",
                format!(
                    "inflow={inflow_volume}, withdrawal={withdrawal_volume}, representation={representation_error}: rescale the exchange volumes"
                ),
            ));
        }
        let state = TensorState {
            h: next_h,
            qx: next_qx,
            qy: next_qy,
        };
        Ok((
            state,
            ExchangeOutcome {
                net_volume,
                depths: ExchangeDepths {
                    inflow: inflow_depth,
                    withdrawal: withdrawal_depth,
                },
                inflow_volume,
                withdrawal_volume,
                representation_error,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::contract::EngineError;
    use crate::solver::coupling::fixture::{AREA, TestResult, coupled_step, exchange, lake};
    use crate::solver::coupling::{CouplingCompletion, CouplingPreparation};

    fn close(actual: f64, expected: f64) -> bool {
        (actual - expected).abs() <= 1e-13 * expected.abs().max(1.0)
    }

    #[test]
    fn surface_exchange_conserves_and_removes_donor_momentum() -> TestResult {
        let solver = lake()?;
        let links = solver.coupling_links(&[0, 2])?;
        let end = 0.05;
        let CouplingPreparation::Ready(stage) = solver.coupling_prepare(end, 0, &links)? else {
            return Err("the lake step must prepare without retry".into());
        };
        let available = stage.available_volume_m3().try_to_vec::<f64>()?;
        let [ux, uy] = stage.velocity_xy();
        let (ux, uy) = (ux.try_to_vec::<f64>()?, uy.try_to_vec::<f64>()?);
        let withdrawal = [0.25 * available[0], 0.0];
        let inflow = [0.0, 0.1];
        let velocity = ([0.0, 0.3], [0.0, -0.2]);
        let operands = exchange(&links, &inflow, &withdrawal, [&velocity.0, &velocity.1])?;
        let CouplingCompletion::Accepted(next) =
            solver.coupling_complete(&stage, Some(&operands))?
        else {
            return Err("the lake step must complete without retry".into());
        };
        assert_eq!(
            solver.time(),
            0.0,
            "the calling engine keeps its accepted state"
        );
        let cells = next.state()?.cells().to_vec();
        // Donor: depth loses a quarter and keeps three quarters of its discharge.
        let donor = available[0] / AREA;
        assert!(close(cells[0].h, 0.75 * donor), "{:?}", cells[0]);
        assert!(close(cells[0].qx, 0.75 * ux[0] * donor), "{:?}", cells[0]);
        assert!(close(cells[0].qy, 0.75 * uy[0] * donor), "{:?}", cells[0]);
        // Receiver: depth gains inflow over area, and discharge gains that depth times its velocity.
        let receiver = available[1] / AREA;
        let gained = 0.1 / AREA;
        assert!(close(cells[2].h, receiver + gained), "{:?}", cells[2]);
        assert!(
            close(cells[2].qx, ux[1] * receiver + gained * 0.3),
            "{:?}",
            cells[2]
        );
        assert!(
            close(cells[2].qy, uy[1] * receiver - gained * 0.2),
            "{:?}",
            cells[2]
        );
        let plain = coupled_step(&solver, end, None, &links)?;
        let unlinked = plain.state()?.cells()[1];
        assert_eq!(
            [cells[1].h, cells[1].qx, cells[1].qy].map(f64::to_bits),
            [unlinked.h, unlinked.qx, unlinked.qy].map(f64::to_bits),
            "an unlinked cell keeps its standalone bits"
        );
        let ledger = next.ledger();
        let tolerance = 1e-9 + 1e-12 * ledger.storage;
        assert_eq!(ledger.exchange_in_m3, 0.1);
        assert_eq!(ledger.exchange_out_m3, withdrawal[0]);
        let change = ledger.storage - ledger.initial_storage;
        assert!(
            (change - (0.1 - withdrawal[0])).abs() <= tolerance,
            "{ledger:?}"
        );
        assert!(ledger.residual.abs() <= tolerance, "{ledger:?}");
        let step = next
            .last_step()?
            .ok_or("the coupled step must record diagnostics")?;
        assert_eq!(step.exchange_in_volume, 0.1);
        assert_eq!(step.exchange_out_volume, withdrawal[0]);
        assert!(step.exchange_representation_error <= tolerance);
        assert!(step.max_local_residual <= tolerance);
        assert_eq!(
            step.sources[0].exchange_withdrawal_depth_m,
            withdrawal[0] / AREA
        );
        assert_eq!(step.sources[2].exchange_inflow_depth_m, gained);
        assert_eq!(step.sources[1].exchange_inflow_depth_m, 0.0);
        assert_eq!(step.sources[1].exchange_withdrawal_depth_m, 0.0);
        // The restart checks replay the exchange through the coordinator import.
        let data = next.snapshot().data()?;
        let mut restored = lake()?;
        restored.coupling_restore_data(&data)?;
        assert_eq!(restored.snapshot().data()?, data);
        Ok(())
    }

    #[test]
    fn exchange_withdrawal_beyond_available_rejects() -> TestResult {
        let solver = lake()?;
        let links = solver.coupling_links(&[1])?;
        let CouplingPreparation::Ready(stage) = solver.coupling_prepare(0.05, 0, &links)? else {
            return Err("the lake step must prepare without retry".into());
        };
        let available = stage.available_volume_m3().try_to_vec::<f64>()?[0];
        let still = [&[0.0][..], &[0.0][..]];
        let beyond = exchange(&links, &[0.0], &[available * 1.001], still)?;
        match solver.coupling_complete(&stage, Some(&beyond)) {
            Err(EngineError::Rejected { operation, reason }) => {
                assert_eq!(operation, "surface exchange");
                assert!(reason.contains("donor withdrawal"), "{reason}");
            }
            Err(error) => return Err(format!("wrong error: {error}").into()),
            Ok(_) => return Err("withdrawal beyond available volume must reject".into()),
        }
        let negative = exchange(&links, &[-1e-3], &[0.0], still)?;
        assert!(matches!(
            solver.coupling_complete(&stage, Some(&negative)),
            Err(EngineError::Rejected {
                operation: "surface exchange",
                ..
            })
        ));
        // Withdrawing exactly the available volume empties the donor without negative depth.
        let all = exchange(&links, &[0.0], &[available], still)?;
        let CouplingCompletion::Accepted(next) = solver.coupling_complete(&stage, Some(&all))?
        else {
            return Err("an exact withdrawal must complete".into());
        };
        let cell = next.state()?.cells()[1];
        assert!(cell.h >= 0.0 && cell.h <= 1e-15, "{cell:?}");
        assert!(cell.qx.abs() <= 1e-15 && cell.qy.abs() <= 1e-15, "{cell:?}");
        assert_eq!(solver.time(), 0.0);
        Ok(())
    }
}
