//! Complete represented receipt admission without soil-history or root replay.

use super::step::{CpuSubareaStep, CpuSubcatchmentStep, CpuSubcatchmentTotals};
use super::{accounting, bridge, phase, transfer, trial};
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::{CpuSubareaGroup, CpuSubcatchmentContext};
use crate::numerics::reservoir::drainage::DrainageResult;
use crate::runoff::SubcatchmentError;
use numr::dtype::DType;
use numr::ops::{CompareOps, IndexingOps};
use numr::runtime::Device;

/// Owned import also checks each native/SI receipt through the shared soil boundary checker.
pub(crate) fn check_step(
    ctx: &CpuSubcatchmentContext,
    step: &CpuSubcatchmentStep,
    dt: f64,
) -> Result<(), SubcatchmentError> {
    trial::check_inputs(ctx, &step.old_depth, &step.input, dt)?;
    let mut depth = step.old_depth.clone();
    let mut received: [CpuTensor; 3] = std::array::from_fn(|_| ctx.zero.clone());
    let first = admit_phase(
        ctx,
        ctx.phase1.as_ref(),
        step.phase1.as_ref(),
        step,
        &mut received,
        &mut depth,
        dt,
    )?;
    let second = admit_phase(
        ctx,
        ctx.phase2.as_ref(),
        step.phase2.as_ref(),
        step,
        &mut received,
        &mut depth,
        dt,
    )?;
    equal(ctx, &step.depth, &depth, "complete accepted depth")?;
    for (actual, wanted) in step.received_by_donor.iter().zip(&received) {
        equal(ctx, actual, wanted, "paired internal receiver volume")?;
    }
    let expected = accounting::totals(
        ctx,
        &step.input,
        [
            ctx.phase1.as_ref().zip(first.as_ref()),
            ctx.phase2.as_ref().zip(second.as_ref()),
        ],
        dt,
    )?;
    for ((name, actual), (_, wanted)) in totals_fields(&step.totals)
        .into_iter()
        .zip(totals_fields(&expected))
    {
        equal(ctx, actual, wanted, name)?;
    }
    for (name, actual, wanted) in [
        (
            "soil adjustment charge",
            step.totals.adjustment_charge,
            expected.adjustment_charge,
        ),
        (
            "routing uncertainty charge",
            step.totals.uncertainty_charge,
            expected.uncertainty_charge,
        ),
        (
            "step physical scale lower bound",
            step.totals.physical_scale_lower,
            expected.physical_scale_lower,
        ),
    ] {
        scalar_equal(actual, wanted, name)?;
    }
    if step.totals.root_iterations != expected.root_iterations {
        return Err(SubcatchmentError::numerical(
            "complete root work differs from admitted phase work",
        ));
    }
    Ok(())
}

fn admit_phase(
    ctx: &CpuSubcatchmentContext,
    group: Option<&CpuSubareaGroup>,
    actual: Option<&CpuSubareaStep>,
    complete: &CpuSubcatchmentStep,
    received: &mut [CpuTensor; 3],
    depth: &mut CpuTensor,
    dt: f64,
) -> Result<Option<CpuSubareaStep>, SubcatchmentError> {
    match (group, actual) {
        (None, None) => Ok(None),
        (Some(group), Some(actual)) => {
            let supply = trial::gather_supply(
                ctx,
                group,
                &complete.old_depth,
                &complete.input,
                received,
                dt,
            )?;
            let expected = phase::admit(group, supply, actual, dt)?;
            check_phase(group, actual, &expected, dt)?;
            trial::route(ctx, group, &expected, received)?;
            *depth = ctx
                .client
                .scatter(depth, 0, &group.indices, &expected.drainage.depth)?;
            Ok(Some(expected))
        }
        _ => Err(SubcatchmentError::input(
            "phase receipts differ from immutable positive-area role support",
        )),
    }
}

fn check_phase(
    group: &CpuSubareaGroup,
    actual: &CpuSubareaStep,
    expected: &CpuSubareaStep,
    dt: f64,
) -> Result<(), SubcatchmentError> {
    let ctx = &group.reservoir;
    transfer::check_merge(
        ctx,
        &[
            &actual.external_volume,
            &actual.received_by_donor[0],
            &actual.received_by_donor[1],
            &actual.received_by_donor[2],
        ],
        &actual.merged,
    )
    .map_err(|e| group.map_reservoir_error(e))?;
    bridge::check_bridge(
        ctx,
        &actual.old_depth,
        &actual.rain_rate,
        &actual.merged,
        dt,
        &actual.bridge,
    )
    .map_err(|e| group.map_reservoir_error(e))?;
    transfer::check_partition(
        ctx,
        &actual.drainage.export_volume,
        &group.route_fraction,
        &actual.partition,
    )
    .map_err(|e| group.map_reservoir_error(e))?;
    for ((name, actual), (_, wanted)) in
        phase_fields(actual).into_iter().zip(phase_fields(expected))
    {
        transfer::equal(ctx, actual, wanted, name).map_err(|e| group.map_reservoir_error(e))?;
    }
    for (actual, wanted) in actual
        .received_by_donor
        .iter()
        .zip(&expected.received_by_donor)
    {
        transfer::equal(ctx, actual, wanted, "phase paired receiver volume")
            .map_err(|e| group.map_reservoir_error(e))?;
    }
    for ((name, actual), (_, wanted)) in drainage_fields(&actual.drainage)
        .into_iter()
        .zip(drainage_fields(&expected.drainage))
    {
        transfer::equal(ctx, actual, wanted, name).map_err(|e| group.map_reservoir_error(e))?;
    }
    for (name, actual, wanted) in [
        (
            "phase adjustment charge",
            actual.adjustment_charge,
            expected.adjustment_charge,
        ),
        (
            "phase routing charge",
            actual.uncertainty_charge,
            expected.uncertainty_charge,
        ),
        (
            "phase physical scale lower bound",
            actual.physical_scale_lower,
            expected.physical_scale_lower,
        ),
    ] {
        scalar_equal(actual, wanted, name)?;
    }
    if actual.drainage.iterations != expected.drainage.iterations {
        return Err(SubcatchmentError::numerical(
            "phase root work differs from admitted work",
        ));
    }
    Ok(())
}

fn equal(
    ctx: &CpuSubcatchmentContext,
    actual: &CpuTensor,
    expected: &CpuTensor,
    quantity: &str,
) -> Result<(), SubcatchmentError> {
    if actual.shape() != expected.shape()
        || actual.dtype() != DType::F64
        || !actual.device().is_same(ctx.zero.device())
    {
        return Err(SubcatchmentError::input(format!(
            "{quantity}: receipt tensor metadata differs from complete support"
        )));
    }
    ctx.finite(actual, quantity)?;
    ctx.reject_mask(
        &ctx.client.ne(actual, expected)?,
        quantity,
        &[("actual", actual), ("expected", expected)],
        &[],
        "Receipt differs from complete source, transfer, or storage algebra",
    )
}

fn scalar_equal(actual: f64, expected: f64, quantity: &str) -> Result<(), SubcatchmentError> {
    if actual.is_finite() && actual == expected {
        Ok(())
    } else {
        Err(SubcatchmentError::numerical(format!(
            "{quantity}: stored={actual}, expected={expected}. Retain the admitted complete accounting"
        )))
    }
}

fn phase_fields(step: &CpuSubareaStep) -> [(&str, &CpuTensor); 21] {
    [
        ("phase initial depth", &step.old_depth),
        ("phase rain", &step.rain_rate),
        ("phase potential evaporation", &step.potential_evaporation),
        ("phase external volume", &step.external_volume),
        ("infiltration depth", &step.infiltration_depth),
        ("water adjustment depth", &step.adjustment_depth),
        ("evaporation depth", &step.evaporation_depth),
        ("authoritative post-source depth", &step.available_depth),
        ("event timing rate", &step.timing_rate),
        ("direct timing mode", &step.direct_timing),
        ("active drainage duration", &step.active_duration),
        ("phase initial storage", &step.storage_before),
        ("applied rain volume", &step.rain_volume),
        ("internal received volume", &step.internal_received_volume),
        ("infiltration volume", &step.infiltration_volume),
        ("evaporation volume", &step.evaporation_volume),
        ("water adjustment volume", &step.adjustment_volume),
        ("phase water residual", &step.water_residual),
        ("phase physical scale", &step.water_scale),
        ("rain-area bound", &step.rain_area_bound),
        ("constructed total rate", &step.bridge.total_rate),
    ]
}

fn drainage_fields(step: &DrainageResult) -> [(&str, &CpuTensor); 12] {
    [
        ("root lower", &step.lower),
        ("root upper", &step.upper),
        ("accepted depth", &step.depth),
        ("export depth", &step.export_depth),
        ("accepted storage", &step.storage_after),
        ("integrated export", &step.export_volume),
        ("endpoint rate", &step.endpoint_rate),
        ("endpoint discharge", &step.endpoint_discharge),
        ("constitutive depth", &step.law_depth),
        ("constitutive volume", &step.law_volume),
        ("constitutive residual", &step.constitutive_residual),
        ("constitutive scale", &step.constitutive_scale),
    ]
}

fn totals_fields(step: &CpuSubcatchmentTotals) -> [(&str, &CpuTensor); 16] {
    [
        ("complete initial storage", &step.storage_before),
        ("complete accepted storage", &step.storage_after),
        ("complete applied rain", &step.applied_rain),
        ("complete external runon", &step.external_runon),
        ("complete infiltration", &step.infiltration),
        ("complete evaporation", &step.evaporation),
        ("complete outlet", &step.outlet),
        ("complete soil adjustment", &step.soil_adjustment),
        ("complete internal traffic", &step.internal_received),
        ("complete subarea water residual", &step.water_residual),
        ("complete subarea physical scale", &step.water_scale),
        ("complete rain-area bound", &step.rain_area_bound),
        ("catchment water residual", &step.catchment_water_residual),
        ("catchment physical scale", &step.catchment_water_scale),
        (
            "catchment rain-area difference",
            &step.catchment_rain_area_difference,
        ),
        (
            "catchment rain-area envelope",
            &step.catchment_rain_area_bound,
        ),
    ]
}
