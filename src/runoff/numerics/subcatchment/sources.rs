//! One authoritative supply, native infiltration, and availability-limited evaporation.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::bridge::Bridge;
use super::step::TensorSubcatchmentSoilStep;
use super::transfer::Merged;
use super::{bounds, bridge, transfer};
use crate::runoff::ReservoirError;
use crate::runoff::numerics::reservoir::drainage;
use crate::runoff::tensor::subcatchment::TensorSubareaGroup;

pub(super) struct Supply<R: Runtime<DType = DType>> {
    pub old_depth: Tensor<R>,
    pub rain_rate: Tensor<R>,
    pub potential_evaporation: Tensor<R>,
    pub external_volume: Tensor<R>,
    pub received_by_donor: [Tensor<R>; 3],
    pub merged: Merged<R>,
    pub bridge: Bridge<R>,
    pub storage_before: Tensor<R>,
    pub rain_volume: Tensor<R>,
}

pub(super) struct Sources<R: Runtime<DType = DType>> {
    pub supply: Supply<R>,
    pub soil: Option<TensorSubcatchmentSoilStep<R>>,
    pub infiltration_depth: Tensor<R>,
    pub adjustment_depth: Tensor<R>,
    pub evaporation_depth: Tensor<R>,
    pub available_depth: Tensor<R>,
    pub direct_timing: Tensor<R>,
    pub infiltration_volume: Tensor<R>,
    pub evaporation_volume: Tensor<R>,
    pub adjustment_volume: Tensor<R>,
    pub internal_received_volume: Tensor<R>,
    pub rain_area_bound: Tensor<R>,
    pub adjustment_charge: f64,
    pub physical_scale_lower: f64,
}

pub(super) fn supply<R>(
    group: &TensorSubareaGroup<R>,
    old_depth: Tensor<R>,
    rain_rate: Tensor<R>,
    potential_evaporation: Tensor<R>,
    external_volume: Tensor<R>,
    received_by_donor: [Tensor<R>; 3],
    dt: f64,
) -> Result<Supply<R>, ReservoirError>
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

pub(super) fn finish<R>(
    group: &TensorSubareaGroup<R>,
    supply: Supply<R>,
    soil: Option<TensorSubcatchmentSoilStep<R>>,
    dt: f64,
) -> Result<Sources<R>, ReservoirError>
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
