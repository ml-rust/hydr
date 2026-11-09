//! Complete ordered receipt support and full-grid tensor reconstruction.

use super::restore::require;
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::{CpuSubcatchmentContext, CpuSubcatchmentInput};
use crate::numerics::subcatchment::{CpuSubcatchmentStep, CpuSubcatchmentTotals, check_step};
use crate::runoff::{Subarea, SubareaReceipt, SubcatchmentError, SubcatchmentStepDiagnostics};
use crate::soil::{SoilReceiptInput, check_source_receipt};

pub(super) fn import(
    context: &CpuSubcatchmentContext,
    diagnostics: &SubcatchmentStepDiagnostics,
) -> Result<CpuSubcatchmentStep, SubcatchmentError> {
    let flat = check_support(context, diagnostics)?;
    let tensor = |select| flat_tensor(context, &flat, select);
    let input = CpuSubcatchmentInput::new(
        context,
        &diagnostics
            .receipts
            .iter()
            .map(|r| r.input)
            .collect::<Vec<_>>(),
    )?;
    let step = CpuSubcatchmentStep {
        old_depth: tensor(|r| r.old_depth_m)?,
        input,
        depth: tensor(|r| r.drainage.depth_m)?,
        phase1: super::phase::import(context.phase1.as_ref(), diagnostics.phase1.as_ref(), &flat)?,
        phase2: super::phase::import(context.phase2.as_ref(), diagnostics.phase2.as_ref(), &flat)?,
        received_by_donor: [
            tensor(|r| r.received_by_donor_m3[0])?,
            tensor(|r| r.received_by_donor_m3[1])?,
            tensor(|r| r.received_by_donor_m3[2])?,
        ],
        totals: totals(context, diagnostics, &flat)?,
    };
    check_step(context, &step, diagnostics.dt)?;
    Ok(step)
}

fn check_support<'a>(
    context: &CpuSubcatchmentContext,
    diagnostics: &'a SubcatchmentStepDiagnostics,
) -> Result<Vec<Option<&'a SubareaReceipt>>, SubcatchmentError> {
    require(
        diagnostics.receipts.len() == context.definitions.len(),
        "catchment receipts require complete ordered definition support",
    )?;
    let mut flat = Vec::with_capacity(context.zero.numel());
    for (index, (receipt, definition)) in diagnostics
        .receipts
        .iter()
        .zip(&context.definitions)
        .enumerate()
    {
        require(
            receipt.catchment == index,
            "catchment receipt identity differs from its position",
        )
        .map_err(|e| e.at_catchment(index))?;
        for subarea in Subarea::ALL {
            let parameters = &definition.parameters;
            let closure = parameters.closure(subarea);
            let source = receipt.subareas[subarea.index()].as_ref();
            require(
                closure.is_some() == source.is_some(),
                "active-subarea receipt presence differs from geometry",
            )
            .map_err(|e| e.at(index, Some(subarea)))?;
            if let (Some(closure), Some(source)) = (closure, source) {
                require(
                    source.catchment == index && source.subarea == subarea,
                    "source receipt physical identity differs from its ordered slot",
                )
                .map_err(|e| e.at(index, Some(subarea)))?;
                require(
                    source.rain_rate_m_per_s == receipt.input.rain_rate_m_per_s
                        && source.potential_evaporation_m_per_s
                            == receipt.input.potential_evaporation_m_per_s
                        && source.external_volume_m3
                            == receipt.input.external_runon_volumes_m3[subarea.index()],
                    "source receipt differs from the original caller input",
                )
                .map_err(|e| e.at(index, Some(subarea)))?;
                let fraction = if parameters.receiver().is_some() {
                    parameters.donor_fraction(subarea)
                } else {
                    0.0
                };
                require(
                    source.partition.fraction == fraction,
                    "source partition fraction differs from routing",
                )
                .map_err(|e| e.at(index, Some(subarea)))?;
                require(
                    source.drainage.endpoint_rate_m_per_s.is_none() == closure.is_instantaneous()
                        && source.drainage.endpoint_discharge_m3_per_s.is_none()
                            == closure.is_instantaneous(),
                    "endpoint rate/discharge presence differs from the physical closure",
                )
                .map_err(|e| e.at(index, Some(subarea)))?;
                require(
                    source.soil.is_some() == (subarea == Subarea::Pervious),
                    "native soil receipt presence differs from pervious support",
                )
                .map_err(|e| e.at(index, Some(subarea)))?;
                if let Some(soil) = &source.soil {
                    check_source_receipt(
                        &SoilReceiptInput {
                            dt_s: diagnostics.dt,
                            incoming_rate_m_per_s: source.bridge.total_rate_m_per_s,
                            ponded_depth_m: source.old_depth_m,
                            incoming_depth_m: source.bridge.input_depth_m,
                            available_depth_m: source.bridge.available_depth_m,
                            infiltration_depth_m: source.infiltration_depth_m,
                            adjustment_depth_m: source.adjustment_depth_m,
                        },
                        soil,
                    )
                    .map_err(|e| SubcatchmentError::from(e).at(index, Some(subarea)))?;
                }
            }
            flat.push(source);
        }
    }
    Ok(flat)
}

fn flat_tensor(
    context: &CpuSubcatchmentContext,
    flat: &[Option<&SubareaReceipt>],
    select: fn(&SubareaReceipt) -> f64,
) -> Result<CpuTensor, SubcatchmentError> {
    context.tensor(
        &flat
            .iter()
            .map(|receipt| receipt.map_or(0.0, select))
            .collect::<Vec<_>>(),
    )
}

fn totals(
    context: &CpuSubcatchmentContext,
    diagnostics: &SubcatchmentStepDiagnostics,
    flat: &[Option<&SubareaReceipt>],
) -> Result<CpuSubcatchmentTotals, SubcatchmentError> {
    let tensor = |select| flat_tensor(context, flat, select);
    let catchment = |select: fn(&crate::runoff::SubcatchmentReceipt) -> f64| {
        context.tensor(&diagnostics.receipts.iter().map(select).collect::<Vec<_>>())
    };
    Ok(CpuSubcatchmentTotals {
        storage_before: tensor(|r| r.storage_before_m3)?,
        storage_after: tensor(|r| r.drainage.storage_after_m3)?,
        applied_rain: tensor(|r| r.rain_volume_m3)?,
        external_runon: tensor(|r| r.external_volume_m3)?,
        infiltration: tensor(|r| r.infiltration_volume_m3)?,
        evaporation: tensor(|r| r.evaporation_volume_m3)?,
        outlet: tensor(|r| r.partition.outlet_volume_m3)?,
        soil_adjustment: tensor(|r| r.adjustment_volume_m3)?,
        internal_received: tensor(|r| r.internal_received_volume_m3)?,
        water_residual: tensor(|r| r.water_residual_m3)?,
        water_scale: tensor(|r| r.water_scale_m3)?,
        rain_area_bound: tensor(|r| r.rain_area_bound_m3)?,
        catchment_water_residual: catchment(|r| r.water_residual_m3)?,
        catchment_water_scale: catchment(|r| r.water_scale_m3)?,
        catchment_rain_area_difference: catchment(|r| r.rain_area_difference_m3)?,
        catchment_rain_area_bound: catchment(|r| r.rain_area_bound_m3)?,
        adjustment_charge: diagnostics.soil_water_adjustment_abs_charge_m3,
        uncertainty_charge: diagnostics.transfer_uncertainty_charge_m3,
        physical_scale_lower: diagnostics.physical_scale_lower_m3,
        root_iterations: diagnostics.root_iterations,
    })
}
