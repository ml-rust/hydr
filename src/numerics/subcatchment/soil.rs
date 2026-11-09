//! Each persistent phase-role soil batch advances once from accepted memory.

use super::sources::Supply;
use super::step::CpuSubcatchmentSoilStep;
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::CpuSubareaGroup;
use crate::runoff::ReservoirError;
use crate::soil::{ModifiedGreenAmptBatch, SoilError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Device;

pub(super) fn trial(
    group: &CpuSubareaGroup,
    supply: &Supply,
    accepted: Option<&ModifiedGreenAmptBatch>,
    dt: f64,
) -> Result<
    (
        Option<CpuSubcatchmentSoilStep>,
        Option<ModifiedGreenAmptBatch>,
    ),
    SoilError,
> {
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
    let evidence = CpuSubcatchmentSoilStep {
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
pub(super) fn check_evidence(
    group: &CpuSubareaGroup,
    evidence: Option<&CpuSubcatchmentSoilStep>,
) -> Result<(), ReservoirError> {
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
            let flags = invalid.try_to_vec::<f64>()?;
            let lane = flags.iter().position(|flag| *flag != 0.0).ok_or_else(|| {
                ReservoirError::numerical("soil evidence rejection mask contradicts its reduction")
            })?;
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

pub(super) fn fields(soil: &CpuSubcatchmentSoilStep) -> [(&str, &CpuTensor); 9] {
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
