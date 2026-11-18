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
mod tests {
    use super::*;
    use crate::runoff::tensor::subcatchment::TensorSubcatchmentContext;
    use crate::runoff::{
        Subarea, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentGeometry,
        SubcatchmentParameters, SubcatchmentRouting,
    };
    use crate::soil::ModifiedGreenAmptParameters;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn compact_soil_evidence_rejects_first_tied_lane_and_retains_subarea_mapping()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device).with_parallelism(ParallelismConfig::new(Some(1), None));
        let parameters = SubcatchmentParameters::new(
            SubcatchmentGeometry {
                total_area_m2: 1.0,
                width_m: 1.0,
                slope: 0.01,
                impervious_fraction: 0.5,
                zero_storage_impervious_fraction: 0.5,
                impervious_manning: 0.0,
                pervious_manning: 0.0,
                impervious_depression_m: 0.0,
                pervious_depression_m: 0.0,
            },
            Some(ModifiedGreenAmptParameters::from_native(
                0.125,
                1.0 / 1024.0,
                1.0,
            )?),
            SubcatchmentRouting::Direct,
        )?;
        let context = TensorSubcatchmentContext::<CpuRuntime>::new(
            client,
            (0..3)
                .map(|i| SubcatchmentDefinition {
                    id: format!("catchment-{i}"),
                    parameters: parameters.clone(),
                })
                .collect(),
            SubcatchmentConfig::default(),
        )?;
        let group = context
            .phase1
            .as_ref()
            .ok_or("Missing direct runoff phase")?;
        assert_eq!(group.soil_slots, vec![2, 5, 8]);
        let zero = group.reservoir.tensor(&[0.0, 0.0, 0.0])?;
        let evidence = TensorSubcatchmentSoilStep {
            infiltration_depth: group.reservoir.tensor(&[0.0, -1.0, -2.0])?,
            adjustment_depth: zero.clone(),
            native_rate: zero.clone(),
            native_available_depth: zero.clone(),
            native_infiltration_depth: zero.clone(),
            native_excess_depth: zero.clone(),
            native_roundoff_bound: zero.clone(),
            bridge_roundoff_bound: zero.clone(),
            total_roundoff_bound: zero,
        };
        let error = check_evidence(group, Some(&evidence))
            .err()
            .ok_or("Missing compact evidence rejection")?;
        assert_eq!(error.element(), Some(5));
        let physical = group.map_reservoir_error(error);
        assert_eq!(physical.catchment(), Some(1));
        assert_eq!(physical.subarea(), Some(Subarea::Pervious));
        Ok(())
    }
}
