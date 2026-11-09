//! Static role-group and complete drainage witness reconstruction.

use super::restore::require;
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::CpuSubareaGroup;
use crate::numerics::reservoir::drainage::DrainageResult;
use crate::numerics::subcatchment::CpuSubareaStep;
use crate::numerics::subcatchment::transfer::{Merged, Partition};
use crate::runoff::{SubareaReceipt, SubcatchmentError, SubcatchmentPhaseDiagnostics};

pub(super) fn import(
    group: Option<&CpuSubareaGroup>,
    diagnostics: Option<&SubcatchmentPhaseDiagnostics>,
    flat: &[Option<&SubareaReceipt>],
) -> Result<Option<CpuSubareaStep>, SubcatchmentError> {
    let (group, diagnostics) = match (group, diagnostics) {
        (None, None) => return Ok(None),
        (Some(group), Some(diagnostics)) => (group, diagnostics),
        _ => {
            return Err(SubcatchmentError::input(
                "checkpoint phase presence differs from immutable topological roles",
            ));
        }
    };
    let sources: Vec<_> = group
        .slots
        .iter()
        .map(|&slot| {
            flat.get(slot).and_then(|source| *source).ok_or_else(|| {
                SubcatchmentError::input("checkpoint phase lacks a required active source receipt")
            })
        })
        .collect::<Result<_, _>>()?;
    let tensor = |select| from_sources(group, &sources, select);
    let step = CpuSubareaStep {
        old_depth: tensor(|r| r.old_depth_m)?,
        rain_rate: tensor(|r| r.rain_rate_m_per_s)?,
        potential_evaporation: tensor(|r| r.potential_evaporation_m_per_s)?,
        external_volume: tensor(|r| r.external_volume_m3)?,
        received_by_donor: [
            tensor(|r| r.received_by_donor_m3[0])?,
            tensor(|r| r.received_by_donor_m3[1])?,
            tensor(|r| r.received_by_donor_m3[2])?,
        ],
        merged: Merged {
            volume: tensor(|r| r.bridge.merged_volume_m3)?,
            bound: tensor(|r| r.bridge.merge_bound_m3)?,
        },
        bridge: super::source::bridge(group, &sources)?,
        soil: super::source::soil(group, &sources)?,
        infiltration_depth: tensor(|r| r.infiltration_depth_m)?,
        adjustment_depth: tensor(|r| r.adjustment_depth_m)?,
        evaporation_depth: tensor(|r| r.evaporation_depth_m)?,
        available_depth: tensor(|r| r.available_depth_m)?,
        timing_rate: tensor(|r| r.timing_rate_m_per_s)?,
        direct_timing: tensor(|r| if r.direct_timing { 1.0 } else { 0.0 })?,
        active_duration: tensor(|r| r.active_duration_s)?,
        drainage: drainage(group, &sources, diagnostics.root_iterations)?,
        partition: Partition {
            routed: tensor(|r| r.partition.routed_volume_m3)?,
            outlet: tensor(|r| r.partition.outlet_volume_m3)?,
            residual: tensor(|r| r.partition.residual_m3)?,
            bound: tensor(|r| r.partition.bound_m3)?,
        },
        storage_before: tensor(|r| r.storage_before_m3)?,
        rain_volume: tensor(|r| r.rain_volume_m3)?,
        internal_received_volume: tensor(|r| r.internal_received_volume_m3)?,
        infiltration_volume: tensor(|r| r.infiltration_volume_m3)?,
        evaporation_volume: tensor(|r| r.evaporation_volume_m3)?,
        adjustment_volume: tensor(|r| r.adjustment_volume_m3)?,
        water_residual: tensor(|r| r.water_residual_m3)?,
        water_scale: tensor(|r| r.water_scale_m3)?,
        rain_area_bound: tensor(|r| r.rain_area_bound_m3)?,
        adjustment_charge: diagnostics.soil_water_adjustment_abs_charge_m3,
        uncertainty_charge: diagnostics.transfer_uncertainty_charge_m3,
        physical_scale_lower: diagnostics.physical_scale_lower_m3,
    };
    require(
        step.old_depth.numel() == group.slots.len(),
        "phase source support differs from cached role map",
    )?;
    Ok(Some(step))
}

pub(super) fn from_sources(
    group: &CpuSubareaGroup,
    sources: &[&SubareaReceipt],
    select: fn(&SubareaReceipt) -> f64,
) -> Result<CpuTensor, SubcatchmentError> {
    let values: Vec<_> = sources.iter().map(|&source| select(source)).collect();
    group
        .reservoir
        .tensor(&values)
        .map_err(|e| group.map_reservoir_error(e))
}

fn drainage(
    group: &CpuSubareaGroup,
    sources: &[&SubareaReceipt],
    iterations: usize,
) -> Result<DrainageResult, SubcatchmentError> {
    let tensor = |select| from_sources(group, sources, select);
    Ok(DrainageResult {
        lower: tensor(|r| r.drainage.root_lower_m)?,
        upper: tensor(|r| r.drainage.root_upper_m)?,
        depth: tensor(|r| r.drainage.depth_m)?,
        export_depth: tensor(|r| r.drainage.export_depth_m)?,
        storage_after: tensor(|r| r.drainage.storage_after_m3)?,
        export_volume: tensor(|r| r.drainage.export_volume_m3)?,
        endpoint_rate: tensor(|r| r.drainage.endpoint_rate_m_per_s.unwrap_or(0.0))?,
        endpoint_discharge: tensor(|r| r.drainage.endpoint_discharge_m3_per_s.unwrap_or(0.0))?,
        law_depth: tensor(|r| r.drainage.law_depth_m)?,
        law_volume: tensor(|r| r.drainage.law_volume_m3)?,
        constitutive_residual: tensor(|r| r.drainage.constitutive_residual_m3)?,
        constitutive_scale: tensor(|r| r.drainage.constitutive_scale_m3)?,
        iterations,
    })
}
