//! Primitive source preparation followed by the shared conservative drainage core.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::check::finish;
use super::receipt::TensorReservoirStep;
use super::{algebra, drainage};
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::{TensorReservoirContext, TensorReservoirState};

pub(crate) fn advance<R>(
    context: &TensorReservoirContext<R>,
    state: &TensorReservoirState<R>,
    dt: f64,
    incoming: &Tensor<R>,
) -> Result<TensorReservoirStep<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let prepared = algebra::prepare(context, &state.depth, incoming, dt)?;
    let drained = drainage::drain(&prepared.hydraulic)?;
    finish(context, prepared, &state.depth, incoming, drained)
}

#[cfg(test)]
mod tests {
    use super::super::check_step;
    use super::super::fixture::{cpu_client, element};
    use super::*;
    use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirParameters, ReservoirState};
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn mixed_branches_retain_independent_tensor_receipts() -> Result<(), ReservoirError> {
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![
                element("finite", 1.0 / 32.0, 32.0)?,
                element("closed", 1.0 / 32.0, 0.0)?,
                element("instant", 0.0, 32.0)?,
            ],
            ReservoirConfig::default(),
        )?;
        let state = TensorReservoirState::<CpuRuntime>::new(
            &context,
            &[ReservoirState {
                depth_m: 0.25 * 0.3048,
            }; 3],
        )?;
        let step = advance(&context, &state, 1.0, &context.tensor(&[0.0; 3])?)?;
        check_step(&context, &step, 1.0)?;
        let depths = step.depth.try_to_vec::<f64>()?;
        let exports = step.export_volume.try_to_vec::<f64>()?;
        assert!(depths[0] > 0.125 * 0.3048 && depths[0] < 0.25 * 0.3048);
        assert_eq!(depths[1], 0.25 * 0.3048);
        assert_eq!(depths[2], 0.125 * 0.3048);
        assert!(exports[0] > 0.0);
        assert_eq!(exports[1], 0.0);
        assert!(exports[2] > 0.0);
        assert_eq!(step.constitutive_residual.try_to_vec::<f64>()?[2], 0.0);
        assert_eq!(step.constitutive_scale.try_to_vec::<f64>()?[2], 0.0);
        assert_eq!(step.endpoint_rate.try_to_vec::<f64>()?[2], 0.0);
        assert_eq!(state.depth.try_to_vec::<f64>()?, vec![0.25 * 0.3048; 3]);
        Ok(())
    }

    #[test]
    fn imported_receipt_rechecks_export_and_law_on_its_element() -> Result<(), ReservoirError> {
        let elements = (0..33)
            .map(|index| element(&format!("element-{index}"), 1.0 / 32.0, 32.0))
            .collect::<Result<Vec<_>, _>>()?;
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            elements,
            ReservoirConfig::default(),
        )?;
        let state = TensorReservoirState::<CpuRuntime>::new(
            &context,
            &vec![
                ReservoirState {
                    depth_m: 0.25 * 0.3048
                };
                33
            ],
        )?;
        let step = advance(&context, &state, 1.0, &context.tensor(&[0.0; 33])?)?;
        check_step(&context, &step, 1.0)?;
        for law_field in [false, true] {
            let mut altered = step.clone();
            let mut values = if law_field {
                altered.law_volume.try_to_vec::<f64>()?
            } else {
                altered.export_volume.try_to_vec::<f64>()?
            };
            values[32] += 1.0;
            if law_field {
                altered.law_volume = context.tensor(&values)?;
            } else {
                altered.export_volume = context.tensor(&values)?;
            }
            let error = check_step(&context, &altered, 1.0).err().ok_or_else(|| {
                ReservoirError::numerical("receipt algebra must reject altered physical volumes")
            })?;
            assert_eq!(error.element(), Some(32));
        }
        Ok(())
    }

    #[test]
    fn exhausted_batched_root_work_keeps_antecedent_tensor() -> Result<(), ReservoirError> {
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![element("finite", 1.0 / 32.0, 32.0)?],
            ReservoirConfig {
                root_iterations: 1,
                ..ReservoirConfig::default()
            },
        )?;
        let state = TensorReservoirState::<CpuRuntime>::new(
            &context,
            &[ReservoirState {
                depth_m: 0.25 * 0.3048,
            }],
        )?;
        let before = state.depth.item::<f64>()?.to_bits();
        let error = advance(&context, &state, 64.0, &context.tensor(&[0.0])?)
            .err()
            .ok_or_else(|| {
                ReservoirError::numerical("one round cannot resolve the implicit root")
            })?;
        assert!(matches!(
            error,
            ReservoirError::Numerical { element: None, .. }
        ));
        assert!(error.to_string().contains("solvr bisect_batched"));
        assert!(error.to_string().contains("max_expansions=0"));
        assert_eq!(state.depth.item::<f64>()?.to_bits(), before);
        Ok(())
    }

    #[test]
    fn positive_implicit_excess_cannot_collapse_into_depression() -> Result<(), ReservoirError> {
        let depression = 1e6f64;
        let parameters = ReservoirParameters::from_si(1.0, 1.0, 1.0, 1.0, 1e-8, depression)?;
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![ReservoirElement {
                id: "stiff".into(),
                parameters,
            }],
            ReservoirConfig::default(),
        )?;
        let state = TensorReservoirState::<CpuRuntime>::new(
            &context,
            &[ReservoirState {
                depth_m: f64::from_bits(depression.to_bits() + 1),
            }],
        )?;
        let error = advance(&context, &state, 1.0, &context.tensor(&[0.0])?)
            .err()
            .ok_or_else(|| {
                ReservoirError::numerical("represented implicit excess must remain positive")
            })?;
        assert_eq!(error.element(), Some(0));
        assert!(error.to_string().contains("represented implicit excess"));
        Ok(())
    }

    #[test]
    fn closed_and_instantaneous_lanes_skip_event_and_root_work() -> Result<(), ReservoirError> {
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![
                element("closed", 1.0 / 32.0, 0.0)?,
                element("instant", 0.0, 32.0)?,
            ],
            ReservoirConfig {
                root_iterations: 1,
                ..ReservoirConfig::default()
            },
        )?;
        let state = TensorReservoirState::<CpuRuntime>::new(
            &context,
            &[ReservoirState { depth_m: 0.0 }; 2],
        )?;
        let step = advance(
            &context,
            &state,
            256.0,
            &context.tensor(&[0.3048 / 1024.0; 2])?,
        )?;
        assert_eq!(step.root_iterations, 0);
        assert_eq!(step.active_duration.try_to_vec::<f64>()?, vec![0.0; 2]);
        assert_eq!(step.root_upper.try_to_vec::<f64>()?, vec![0.0; 2]);
        check_step(&context, &step, 256.0)?;
        Ok(())
    }

    #[test]
    fn positive_supply_absorbed_by_storage_rounding_rejects() -> Result<(), ReservoirError> {
        let context = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![element("closed", 1.0 / 32.0, 0.0)?],
            ReservoirConfig::default(),
        )?;
        let state =
            TensorReservoirState::<CpuRuntime>::new(&context, &[ReservoirState { depth_m: 1e16 }])?;
        let error = advance(&context, &state, 1.0, &context.tensor(&[1e-20])?)
            .err()
            .ok_or_else(|| {
                ReservoirError::numerical("positive incoming storage must remain represented")
            })?;
        assert_eq!(error.element(), Some(0));
        assert!(error.to_string().contains("absorbed by storage rounding"));
        Ok(())
    }
}
