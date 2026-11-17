use super::Solver;
use crate::contract::EngineError;
use crate::tensor::TensorState;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, HypotOps, IndexingOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

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
    pub(super) fn check_admission(&self, state: &TensorState<R>) -> Result<(), EngineError> {
        let client = &self.context.client;
        let zero = &self.context.zero;
        let one = client.add_scalar(zero, 1.0)?;
        let finite = |value: &Tensor<R>| -> Result<Tensor<R>, EngineError> {
            Ok(client.le(&client.abs(value)?, &client.add_scalar(zero, f64::MAX)?)?)
        };
        let wet = client.gt(&state.h, zero)?;
        let depth = client.where_cond(&wet, &state.h, &one)?;
        let mut admitted = client.mul(&finite(&state.h)?, &client.ge(&state.h, zero)?)?;
        for momentum in [&state.qx, &state.qy] {
            admitted = client.mul(&admitted, &finite(momentum)?)?;
            let dry_zero = client.maximum(&wet, &client.eq(momentum, zero)?)?;
            admitted = client.mul(&admitted, &dry_zero)?;
            admitted = client.mul(&admitted, &finite(&client.div(momentum, &depth)?)?)?;
        }
        for value in [&state.h, &state.qx, &state.qy] {
            let inactive_zero = client.maximum(&self.context.active, &client.eq(value, zero)?)?;
            admitted = client.mul(&admitted, &inactive_zero)?;
        }
        admitted = client.mul(
            &admitted,
            &finite(&client.add(&self.context.bed, &state.h)?)?,
        )?;
        let volume = client.mul_scalar(&state.h, self.domain().grid().cell_area())?;
        admitted = client.mul(&admitted, &finite(&volume)?)?;
        let represented =
            client.maximum(&client.eq(&state.h, zero)?, &client.gt(&volume, zero)?)?;
        admitted = client.mul(&admitted, &represented)?;
        let total = client.sum(&volume, &[0], false)?.item::<f64>()?;
        // Preserve the model's row-major running total in addition to accounting reduction.
        let running_total = client.cumsum(&volume, 0)?;
        let running_total = running_total
            .narrow(0, volume.numel() - 1, 1)?
            .item::<f64>()?;
        if client.min(&admitted, &[0], false)?.item::<f64>()? != 1.0
            || !total.is_finite()
            || !running_total.is_finite()
        {
            // Host projection preserves detailed model errors only for rejected candidates.
            state.to_flow(self.domain())?;
            return Err(EngineError::rejected(
                "accepted state",
                "tensor admission differs from physical state checks: inspect runtime arithmetic",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{BoundaryConditions, ForcingInterval, ForcingSchedule, SolverConfig};
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandModel, SoilState};
    use crate::solver::{InitialState, SolverSetup};
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use numr::tensor::Tensor;

    fn solver(area: f64, bed: f64) -> Result<Solver<CpuRuntime>, Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(2, 1, area, 1.0)?,
            vec![bed; 2],
            vec![0.0; 2],
            vec![true, false],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.0; 2],
                potential_evaporation: vec![0.0; 2],
                withdrawal: vec![0.0; 2],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(&domain, vec![CellState::default(); 2])?;
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        Ok(Solver::new(
            client,
            SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?)
    }

    fn candidate(
        solver: &Solver<CpuRuntime>,
        h: &[f64],
        qx: &[f64],
    ) -> Result<TensorState<CpuRuntime>, EngineError> {
        let device = solver.context.client.device();
        Ok(TensorState {
            h: Tensor::from_slice(h, &[2], device)?,
            qx: Tensor::from_slice(qx, &[2], device)?,
            qy: Tensor::from_slice(&[0.0; 2], &[2], device)?,
        })
    }

    #[test]
    fn device_admission_rejects_unrepresentable_and_unsupported_states()
    -> Result<(), Box<dyn std::error::Error>> {
        let solver = solver(1.0, 0.0)?;
        for (h, qx) in [
            ([f64::NAN, 0.0], [0.0, 0.0]),
            ([-1.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [1.0, 0.0]),
            ([1.0, 0.0], [f64::INFINITY, 0.0]),
            ([f64::from_bits(1), 0.0], [1.0, 0.0]),
            ([0.0, 1.0], [0.0, 0.0]),
        ] {
            assert!(matches!(
                solver.check_admission(&candidate(&solver, &h, &qx)?),
                Err(EngineError::Model(_))
            ));
        }
        solver.check_admission(&candidate(&solver, &[f64::from_bits(1), 0.0], &[0.0; 2])?)?;
        assert_eq!(solver.time(), 0.0);
        assert_eq!(solver.state()?.cells(), &[CellState::default(); 2]);
        Ok(())
    }

    #[test]
    fn device_admission_checks_elevation_and_cell_volume() -> Result<(), Box<dyn std::error::Error>>
    {
        let elevated = solver(1.0, f64::MAX)?;
        assert!(matches!(
            elevated.check_admission(&candidate(&elevated, &[f64::MAX, 0.0], &[0.0; 2])?),
            Err(EngineError::Model(_))
        ));
        let tiny = solver(f64::MIN_POSITIVE, 0.0)?;
        assert!(matches!(
            tiny.check_admission(&candidate(&tiny, &[f64::MIN_POSITIVE, 0.0], &[0.0; 2])?),
            Err(EngineError::Model(_))
        ));
        Ok(())
    }

    #[test]
    fn volume_scan_preserves_row_major_finite_overflow_admission()
    -> Result<(), Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(3, 1, 1.0, 1.0)?,
            vec![0.0; 3],
            vec![0.0; 3],
            vec![true; 3],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        let quarter_ulp = 2.0_f64.powi(969);
        for depths in [
            [f64::MAX, quarter_ulp, quarter_ulp],
            [f64::MAX, 2.0_f64.powi(971), 0.0],
            [f64::MAX / 4.0; 3],
        ] {
            let cells = depths
                .into_iter()
                .map(|h| CellState {
                    h,
                    qx: 0.0,
                    qy: 0.0,
                })
                .collect();
            let physical = FlowState::new(&domain, cells);
            let volume = Tensor::<CpuRuntime>::from_slice(&depths, &[3], client.device())?;
            let prefix = client.cumsum(&volume, 0)?;
            let total = prefix.narrow(0, 2, 1)?.item::<f64>()?;
            let serial = depths.into_iter().fold(0.0, |sum, value| sum + value);
            assert_eq!(total.to_bits(), serial.to_bits());
            assert_eq!(physical.is_ok(), total.is_finite());
        }
        Ok(())
    }
}
