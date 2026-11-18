//! Manning friction and tensor reductions shared by both trial phases.

use super::super::Solver;
use super::super::tensor::TensorState;
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, ReduceOps,
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
        + CorrectlyRoundedOps<R>
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
    pub(super) fn friction(
        &self,
        state: TensorState<R>,
        dt: f64,
    ) -> Result<TensorState<R>, EngineError> {
        let client = &self.context.client;
        let zero = &self.context.zero;
        let wet = client.gt(&state.h, zero)?;
        let depth = client.where_cond(&wet, &state.h, &client.add_scalar(zero, 1.0)?)?;
        let speed = client.hypot(&state.qx, &state.qy)?;
        self.finite(&speed, "Manning discharge magnitude")?;
        let active_drag = client.mul(
            &client.gt(&speed, zero)?,
            &client.gt(&self.context.manning, zero)?,
        )?;
        let power = client.cr_pow_scalar(&depth, 7.0 / 3.0)?;
        let safe_power = client.where_cond(&active_drag, &power, &client.add_scalar(zero, 1.0)?)?;
        let drag = client.mul(&self.context.manning, &self.context.manning)?;
        let drag = client.mul(&drag, &speed)?;
        let drag = client.mul_scalar(&drag, self.config().gravity * dt)?;
        let drag = client.div(&drag, &safe_power)?;
        let denominator = client.add_scalar(&drag, 1.0)?;
        self.finite(&denominator, "Manning denominator")?;
        Ok(TensorState {
            h: state.h,
            qx: client.div(&state.qx, &denominator)?,
            qy: client.div(&state.qy, &denominator)?,
        })
    }

    pub(super) fn total(&self, values: &Tensor<R>) -> Result<f64, EngineError> {
        Ok(self
            .context
            .client
            .sum(values, &[0], false)?
            .item::<f64>()?)
    }

    pub(super) fn any(&self, mask: &Tensor<R>) -> Result<bool, EngineError> {
        Ok(self
            .context
            .client
            .any(&self.context.client.cast(mask, DType::F64)?, &[0], false)?
            .item::<f64>()?
            != 0.0)
    }

    pub(super) fn finite(&self, values: &Tensor<R>, name: &'static str) -> Result<(), EngineError> {
        let client = &self.context.client;
        let invalid = client.maximum(
            &client.cast(&client.isnan(values)?, DType::F64)?,
            &client.cast(&client.isinf(values)?, DType::F64)?,
        )?;
        if self.any(&invalid)? {
            let values = values.try_to_vec::<f64>()?;
            let first = values
                .iter()
                .enumerate()
                .find(|(_, value)| !value.is_finite());
            return Err(EngineError::rejected(
                name,
                format!(
                    "time={}, first nonfinite input={first:?}: inspect physical scales and tensor operations",
                    self.time()
                ),
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
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn numr_friction_matches_exact_fixed_depth_vector_decay()
    -> Result<(), Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![0.03],
            vec![true],
            SpatialReference::new(
                "synthetic metric plane".into(),
                "synthetic datum".into(),
                [0.0; 2],
            )?,
        )?;
        let state = FlowState::new(
            &domain,
            vec![CellState {
                h: 1.0,
                qx: 2.0,
                qy: 1.0,
            }],
        )?;
        let walls = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &walls,
            vec![ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![0.0],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: vec![],
            }],
        )?;
        let land = crate::soil::LandModel::impervious(&domain);
        let soil = crate::soil::SoilState::initialized(&land)?;
        let solver = Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            crate::solver::SolverSetup {
                domain,
                land,
                boundaries: walls,
                forcing,
                config: SolverConfig::default(),
            },
            crate::solver::InitialState {
                time: 0.0,
                flow: state,
                soil,
            },
        )?;
        let result = solver
            .friction(solver.accepted.flow.clone(), 0.1)?
            .to_flow(solver.domain())?;
        let denominator = 1.0 + 9.81 * 0.03_f64.powi(2) * 5.0_f64.sqrt() * 0.1;
        assert!((result.cells()[0].qx - 2.0 / denominator).abs() <= 1e-12);
        assert!((result.cells()[0].qy - 1.0 / denominator).abs() <= 1e-12);
        assert_eq!(result.cells()[0].h, 1.0);
        Ok(())
    }
}
