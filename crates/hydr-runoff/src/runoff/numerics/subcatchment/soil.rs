//! Each persistent phase-role soil batch advances once from accepted memory.

use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::sources::Supply;
use super::step::TensorSubcatchmentSoilStep;
use crate::runoff::ReservoirError;
use crate::runoff::tensor::subcatchment::TensorSubareaGroup;
use crate::soil::{ModifiedGreenAmptBatch, SoilError};
use numr::dtype::DType;
use numr::runtime::Device;

pub(super) type SoilTrial<R> = (
    Option<TensorSubcatchmentSoilStep<R>>,
    Option<ModifiedGreenAmptBatch<R>>,
);

pub(super) fn trial<R>(
    group: &TensorSubareaGroup<R>,
    supply: &Supply<R>,
    accepted: Option<&ModifiedGreenAmptBatch<R>>,
    dt: f64,
) -> Result<SoilTrial<R>, SoilError>
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
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    if group.soil_slots.is_empty() {
        if accepted.is_some() {
            return Err(SoilError::input(
                "impervious phase unexpectedly contains soil memory",
            ));
        }
        return Ok((None, None));
    }
    let accepted = accepted.ok_or_else(|| {
        SoilError::input("pervious phase requires complete accepted native soil memory")
    })?;
    let c = &group.reservoir.client;
    let gather = |value| c.gather(value, 0, &group.soil_indices);
    let trial = accepted.trial_surface_tensors(
        dt,
        &gather(&supply.bridge.total_rate)?,
        &gather(&supply.old_depth)?,
        &gather(&supply.bridge.input_depth)?,
        &gather(&supply.bridge.available_depth)?,
    )?;
    let evidence = TensorSubcatchmentSoilStep {
        infiltration_depth: trial.infiltration_depth,
        adjustment_depth: trial.adjustment_depth,
        native_rate: trial.native_rate,
        native_available_depth: trial.native_available_depth,
        native_infiltration_depth: trial.native_infiltration_depth,
        native_excess_depth: trial.native_excess_depth,
        native_roundoff_bound: trial.native_roundoff_bound,
        bridge_roundoff_bound: trial.bridge_roundoff_bound,
        total_roundoff_bound: trial.total_roundoff_bound,
    };
    Ok((Some(evidence), Some(trial.next)))
}

/// Native/SI envelope algebra is checked by the shared soil import-boundary checker.
/// This tensor-only stage checks compact support and all physical evidence ranges.
pub(super) fn check_evidence<R>(
    group: &TensorSubareaGroup<R>,
    evidence: Option<&TensorSubcatchmentSoilStep<R>>,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    if group.soil_slots.is_empty() {
        return if evidence.is_none() {
            Ok(())
        } else {
            Err(ReservoirError::input(
                "impervious phase cannot contain soil evidence",
            ))
        };
    }
    let evidence = evidence.ok_or_else(|| {
        ReservoirError::input("pervious phase requires complete native source evidence")
    })?;
    let c = &group.reservoir.client;
    for (name, value) in fields(evidence) {
        if value.shape() != [group.soil_slots.len()]
            || value.dtype() != DType::F64
            || !value.device().is_same(group.reservoir.zero.device())
        {
            return Err(ReservoirError::input(format!(
                "{name}: compact pervious tensor support differs"
            )));
        }
        let zero = c.mul_scalar(&c.cast(&c.isnan(value)?, DType::F64)?, 0.0)?;
        let invalid = c.maximum(
            &c.lt(value, &zero)?,
            &c.maximum(
                &c.cast(&c.isnan(value)?, DType::F64)?,
                &c.cast(&c.isinf(value)?, DType::F64)?,
            )?,
        )?;
        if c.any(&invalid, &[], false)?.item::<f64>()? != 0.0 {
            let selected = c.argmax(&invalid, 0, false)?.item::<i64>()?;
            let lane = usize::try_from(selected)
                .ok()
                .filter(|lane| *lane < invalid.numel())
                .ok_or_else(|| {
                    ReservoirError::numerical(
                        "soil evidence rejection mask contradicts its reduction",
                    )
                })?;
            if invalid.narrow(0, lane, 1)?.item::<f64>()? == 0.0 {
                return Err(ReservoirError::numerical(
                    "soil evidence rejection mask contradicts its reduction",
                ));
            }
            let index = *group
                .soil_slots
                .get(lane)
                .ok_or_else(|| ReservoirError::numerical("soil evidence compact map differs"))?;
            return Err(ReservoirError::numerical(format!(
                "{name}: require finite nonnegative native source evidence"
            ))
            .at_element(index));
        }
    }
    Ok(())
}

pub(super) fn fields<R>(soil: &TensorSubcatchmentSoilStep<R>) -> [(&str, &Tensor<R>); 9]
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>,
{
    [
        ("infiltration depth", &soil.infiltration_depth),
        ("water adjustment depth", &soil.adjustment_depth),
        ("native rate", &soil.native_rate),
        ("native availability", &soil.native_available_depth),
        ("native infiltration", &soil.native_infiltration_depth),
        ("native excess", &soil.native_excess_depth),
        ("native bound", &soil.native_roundoff_bound),
        ("bridge bound", &soil.bridge_roundoff_bound),
        ("total bound", &soil.total_roundoff_bound),
    ]
}

#[cfg(test)]
#[path = "soil_rejection_tests.rs"]
mod rejection_tests;
