//! Linear accepted-output assembly from each compact phase tensor exactly once.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::projection_hydraulic::{self, read};
use super::{Subarea, SubareaReceipt, SubcatchmentError, SubcatchmentPhaseDiagnostics};
use crate::runoff::numerics::subcatchment::TensorSubareaStep;
use crate::runoff::tensor::subcatchment::TensorSubareaGroup;
use crate::soil::SoilSourceReceipt;

pub(super) fn phase<R>(
    group: &TensorSubareaGroup<R>,
    step: &TensorSubareaStep<R>,
) -> Result<Vec<SubareaReceipt>, SubcatchmentError>
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
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let n = group.slots.len();
    let fields = [
        &step.old_depth,
        &step.rain_rate,
        &step.potential_evaporation,
        &step.external_volume,
        &step.received_by_donor[0],
        &step.received_by_donor[1],
        &step.received_by_donor[2],
        &step.infiltration_depth,
        &step.adjustment_depth,
        &step.evaporation_depth,
        &step.available_depth,
        &step.timing_rate,
        &step.direct_timing,
        &step.active_duration,
        &step.storage_before,
        &step.rain_volume,
        &step.internal_received_volume,
        &step.infiltration_volume,
        &step.evaporation_volume,
        &step.adjustment_volume,
        &step.water_residual,
        &step.water_scale,
        &step.rain_area_bound,
    ];
    let a = fields
        .into_iter()
        .map(|f| read(f, n))
        .collect::<Result<Vec<_>, _>>()?;
    let bridges = projection_hydraulic::bridges(step, n)?;
    let drainage = projection_hydraulic::drainage(group, step)?;
    let partition = projection_hydraulic::partitions(group, step)?;
    let soils = soil(group, step)?;
    group
        .slots
        .iter()
        .enumerate()
        .map(|(i, &slot)| {
            Ok(SubareaReceipt {
                catchment: slot / 3,
                subarea: Subarea::from_index(slot % 3).ok_or_else(|| {
                    SubcatchmentError::numerical("cached subarea index is invalid")
                })?,
                old_depth_m: a[0][i],
                rain_rate_m_per_s: a[1][i],
                potential_evaporation_m_per_s: a[2][i],
                external_volume_m3: a[3][i],
                received_by_donor_m3: [a[4][i], a[5][i], a[6][i]],
                bridge: bridges[i].clone(),
                soil: soils[i].clone(),
                infiltration_depth_m: a[7][i],
                adjustment_depth_m: a[8][i],
                evaporation_depth_m: a[9][i],
                available_depth_m: a[10][i],
                timing_rate_m_per_s: a[11][i],
                direct_timing: a[12][i] == 1.0,
                active_duration_s: a[13][i],
                drainage: drainage[i].clone(),
                partition: partition[i].clone(),
                storage_before_m3: a[14][i],
                rain_volume_m3: a[15][i],
                internal_received_volume_m3: a[16][i],
                infiltration_volume_m3: a[17][i],
                evaporation_volume_m3: a[18][i],
                adjustment_volume_m3: a[19][i],
                water_residual_m3: a[20][i],
                water_scale_m3: a[21][i],
                rain_area_bound_m3: a[22][i],
            })
        })
        .collect()
}

fn soil<R>(
    group: &TensorSubareaGroup<R>,
    step: &TensorSubareaStep<R>,
) -> Result<Vec<Option<SoilSourceReceipt>>, SubcatchmentError>
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
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let mut result = vec![None; group.slots.len()];
    let Some(s) = &step.soil else {
        return Ok(result);
    };
    let fields = [
        &s.native_rate,
        &s.native_available_depth,
        &s.native_infiltration_depth,
        &s.native_excess_depth,
        &s.native_roundoff_bound,
        &s.bridge_roundoff_bound,
        &s.total_roundoff_bound,
    ];
    let a = fields
        .into_iter()
        .map(|f| read(f, group.soil_slots.len()))
        .collect::<Result<Vec<_>, _>>()?;
    for (i, &slot) in group.soil_slots.iter().enumerate() {
        let destination = result.get_mut(slot).ok_or_else(|| {
            SubcatchmentError::numerical("soil projection map exceeds phase support")
        })?;
        *destination = Some(SoilSourceReceipt {
            native_rate_ft_per_s: a[0][i],
            native_available_depth_ft: a[1][i],
            native_infiltration_depth_ft: a[2][i],
            native_excess_depth_ft: a[3][i],
            native_roundoff_bound_ft: a[4][i],
            bridge_roundoff_bound_m: a[5][i],
            total_roundoff_bound_m: a[6][i],
        });
    }
    Ok(result)
}

pub(super) fn diagnostics<R>(step: &TensorSubareaStep<R>) -> SubcatchmentPhaseDiagnostics
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
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    SubcatchmentPhaseDiagnostics {
        root_iterations: step.drainage.iterations,
        soil_water_adjustment_abs_charge_m3: step.adjustment_charge,
        transfer_uncertainty_charge_m3: step.uncertainty_charge,
        physical_scale_lower_m3: step.physical_scale_lower,
    }
}
