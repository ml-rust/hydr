//! One authoritative supply, native infiltration, and availability-limited evaporation.

use super::bridge::Bridge;
use super::step::CpuSubcatchmentSoilStep;
use super::transfer::Merged;
use super::{bounds, bridge, transfer};
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::CpuSubareaGroup;
use crate::numerics::reservoir::drainage;
use crate::runoff::ReservoirError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps};

pub(super) struct Supply {
    pub old_depth: CpuTensor,
    pub rain_rate: CpuTensor,
    pub potential_evaporation: CpuTensor,
    pub external_volume: CpuTensor,
    pub received_by_donor: [CpuTensor; 3],
    pub merged: Merged,
    pub bridge: Bridge,
    pub storage_before: CpuTensor,
    pub rain_volume: CpuTensor,
}

pub(super) struct Sources {
    pub supply: Supply,
    pub soil: Option<CpuSubcatchmentSoilStep>,
    pub infiltration_depth: CpuTensor,
    pub adjustment_depth: CpuTensor,
    pub evaporation_depth: CpuTensor,
    pub available_depth: CpuTensor,
    pub direct_timing: CpuTensor,
    pub infiltration_volume: CpuTensor,
    pub evaporation_volume: CpuTensor,
    pub adjustment_volume: CpuTensor,
    pub internal_received_volume: CpuTensor,
    pub rain_area_bound: CpuTensor,
    pub adjustment_charge: f64,
    pub physical_scale_lower: f64,
}

pub(super) fn supply(
    group: &CpuSubareaGroup,
    old_depth: CpuTensor,
    rain_rate: CpuTensor,
    potential_evaporation: CpuTensor,
    external_volume: CpuTensor,
    received_by_donor: [CpuTensor; 3],
    dt: f64,
) -> Result<Supply, ReservoirError> {
    let ctx = &group.reservoir;
    bounds::check_nonnegative(ctx, &potential_evaporation, "potential evaporation rate")?;
    let merged = transfer::merge(
        ctx,
        &[
            &external_volume,
            &received_by_donor[0],
            &received_by_donor[1],
            &received_by_donor[2],
        ],
    )?;
    let bridge = bridge::prepare(ctx, &old_depth, &rain_rate, &merged, dt)?;
    let storage_before = drainage::volume(ctx, &old_depth, "initial subarea storage")?;
    let rain_volume = drainage::volume(ctx, &bridge.rain_depth, "applied subarea rainfall")?;
    drainage::volume(ctx, &bridge.input_depth, "constructed source volume")?;
    drainage::volume(ctx, &bridge.available_depth, "pre-infiltration storage")?;
    Ok(Supply {
        old_depth,
        rain_rate,
        potential_evaporation,
        external_volume,
        received_by_donor,
        merged,
        bridge,
        storage_before,
        rain_volume,
    })
}

pub(super) fn finish(
    group: &CpuSubareaGroup,
    supply: Supply,
    soil: Option<CpuSubcatchmentSoilStep>,
    dt: f64,
) -> Result<Sources, ReservoirError> {
    let ctx = &group.reservoir;
    let c = &ctx.client;
    super::soil::check_evidence(group, soil.as_ref())?;
    let (infiltration_depth, adjustment_depth) = match &soil {
        Some(soil) => (
            c.scatter(&ctx.zero, 0, &group.soil_indices, &soil.infiltration_depth)?,
            c.scatter(&ctx.zero, 0, &group.soil_indices, &soil.adjustment_depth)?,
        ),
        None => (ctx.zero.clone(), ctx.zero.clone()),
    };
    let after_soil = c.where_cond(
        &c.gt(&adjustment_depth, &ctx.zero)?,
        &ctx.zero,
        &c.sub(&supply.bridge.available_depth, &infiltration_depth)?,
    )?;
    bounds::check_nonnegative(ctx, &after_soil, "post-infiltration subarea depth")?;
    let potential = c.mul_scalar(&supply.potential_evaporation, dt)?;
    bounds::check_nonnegative(ctx, &potential, "potential evaporation depth")?;
    ctx.reject_mask(
        &c.mul(
            &c.gt(&after_soil, &ctx.zero)?,
            &c.mul(
                &c.gt(&supply.potential_evaporation, &ctx.zero)?,
                &c.eq(&potential, &ctx.zero)?,
            )?,
        )?,
        "wet potential evaporation underflow",
        &[
            ("rate", &supply.potential_evaporation),
            ("available", &after_soil),
        ],
        &[("dt", dt)],
        "Rescale positive evaporation or interval",
    )?;
    let evaporation_depth =
        c.where_cond(&c.lt(&potential, &after_soil)?, &potential, &after_soil)?;
    let available_depth = c.sub(&after_soil, &evaporation_depth)?;
    bounds::check_nonnegative(ctx, &available_depth, "post-source storage budget")?;
    let direct_timing = c.mul(
        &c.eq(&supply.merged.volume, &ctx.zero)?,
        &c.mul(
            &c.eq(&infiltration_depth, &ctx.zero)?,
            &c.mul(
                &c.eq(&evaporation_depth, &ctx.zero)?,
                &c.eq(&adjustment_depth, &ctx.zero)?,
            )?,
        )?,
    )?;
    let infiltration_volume =
        drainage::volume(ctx, &infiltration_depth, "subarea infiltration volume")?;
    let evaporation_volume =
        drainage::volume(ctx, &evaporation_depth, "subarea evaporation volume")?;
    let adjustment_volume = drainage::volume(ctx, &adjustment_depth, "subarea water adjustment")?;
    let (_, adjustment_charge) = group
        .adjustment_reduction
        .totals_with_areas(&ctx.client, &adjustment_depth, &ctx.area)
        .map_err(|error| {
            let index = error.cell();
            let mapped = match error {
                crate::soil::SoilError::InvalidInput { context, .. } => {
                    ReservoirError::input(context)
                }
                crate::soil::SoilError::Numerical { context, .. } => {
                    ReservoirError::numerical(context)
                }
                crate::soil::SoilError::Runtime { context, .. } => ReservoirError::Runtime {
                    element: None,
                    context,
                },
            };
            if let Some(index) = index {
                mapped.at_element(index)
            } else {
                mapped
            }
        })?;
    let internal_received_volume = c.add(
        &c.add(&supply.received_by_donor[0], &supply.received_by_donor[1])?,
        &supply.received_by_donor[2],
    )?;
    bounds::check_nonnegative(ctx, &internal_received_volume, "internal received volume")?;
    let rain_area_bound =
        bounds::rain_area_bound(ctx, &supply.bridge.rain_depth, &group.area_bound)?;
    let mut physical_scale_lower = bounds::sum_lower(ctx, &supply.rain_volume)?;
    for value in [
        &supply.external_volume,
        &supply.received_by_donor[0],
        &supply.received_by_donor[1],
        &supply.received_by_donor[2],
    ] {
        physical_scale_lower =
            bounds::add_scale_lower(physical_scale_lower, bounds::sum_lower(ctx, value)?)?;
    }
    Ok(Sources {
        supply,
        soil,
        infiltration_depth,
        adjustment_depth,
        evaporation_depth,
        available_depth,
        direct_timing,
        infiltration_volume,
        evaporation_volume,
        adjustment_volume,
        internal_received_volume,
        rain_area_bound,
        adjustment_charge,
        physical_scale_lower,
    })
}
