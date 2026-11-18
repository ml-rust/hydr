//! One phase advances its disjoint physical lanes and accepted soil memory once.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::sources::{Sources, Supply};
use super::step::TensorSubareaStep;
use super::{bounds, sources, timing, transfer};
use crate::runoff::numerics::reservoir::drainage::{self, DrainageResult, DrainageRoot};
use crate::runoff::tensor::subcatchment::TensorSubareaGroup;
use crate::runoff::{ReservoirError, SubcatchmentError};
use crate::soil::ModifiedGreenAmptBatch;

pub(super) fn advance<R>(
    group: &TensorSubareaGroup<R>,
    supply: Supply<R>,
    accepted_soil: Option<&ModifiedGreenAmptBatch<R>>,
    dt: f64,
) -> Result<(TensorSubareaStep<R>, Option<ModifiedGreenAmptBatch<R>>), SubcatchmentError>
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
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let (soil, next_soil) = super::soil::trial(group, &supply, accepted_soil, dt)
        .map_err(|error| group.map_soil_error(error))?;
    let sources = sources::finish(group, supply, soil, dt)
        .map_err(|error| group.map_reservoir_error(error))?;
    let (budget, timing_rate) = timing::prepare(&group.reservoir, &sources, dt)
        .map_err(|error| group.map_reservoir_error(error))?;
    let active_duration = budget.active_duration().clone();
    let drained = drainage::drain(&budget).map_err(|error| group.map_reservoir_error(error))?;
    let step = finish(group, sources, timing_rate, active_duration, drained)
        .map_err(|error| group.map_reservoir_error(error))?;
    Ok((step, next_soil))
}

/// Check represented source and root witnesses without soil transition or root replay.
pub(super) fn admit<R>(
    group: &TensorSubareaGroup<R>,
    supply: Supply<R>,
    step: &TensorSubareaStep<R>,
    dt: f64,
) -> Result<TensorSubareaStep<R>, SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let sources = sources::finish(group, supply, step.soil.clone(), dt)
        .map_err(|error| group.map_reservoir_error(error))?;
    let (budget, timing_rate) = timing::prepare(&group.reservoir, &sources, dt)
        .map_err(|error| group.map_reservoir_error(error))?;
    let active_duration = budget.active_duration().clone();
    let drained = drainage::admit_drainage(
        &budget,
        DrainageRoot {
            lower: step.drainage.lower.clone(),
            upper: step.drainage.upper.clone(),
            depth: step.drainage.depth.clone(),
            iterations: step.drainage.iterations,
        },
    )
    .map_err(|error| group.map_reservoir_error(error))?;
    finish(group, sources, timing_rate, active_duration, drained)
        .map_err(|error| group.map_reservoir_error(error))
}

fn finish<R>(
    group: &TensorSubareaGroup<R>,
    sources: Sources<R>,
    timing_rate: Tensor<R>,
    active_duration: Tensor<R>,
    drainage: DrainageResult<R>,
) -> Result<TensorSubareaStep<R>, ReservoirError>
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
    let partition = transfer::partition(ctx, &drainage.export_volume, &group.route_fraction)?;
    let inflow = c.add(&sources.supply.rain_volume, &sources.supply.merged.volume)?;
    let credited = c.add(&inflow, &sources.adjustment_volume)?;
    let losses = c.add(
        &c.add(&sources.infiltration_volume, &sources.evaporation_volume)?,
        &drainage.export_volume,
    )?;
    ctx.finite(&credited, "subarea credited volume")?;
    ctx.finite(&losses, "subarea losses and export")?;
    let water_residual = c.sub(
        &c.sub(&drainage.storage_after, &sources.supply.storage_before)?,
        &c.sub(&credited, &losses)?,
    )?;
    let water_scale = c.maximum(
        &c.maximum(&sources.supply.storage_before, &drainage.storage_after)?,
        &c.maximum(&credited, &losses)?,
    )?;
    drainage::check_residual(
        ctx,
        &water_residual,
        &water_scale,
        &ctx.one,
        "subarea source water residual",
    )?;
    let positive_uncertainty = c.maximum(
        &c.gt(&sources.supply.bridge.uncertainty, &ctx.zero)?,
        &c.gt(&partition.bound, &ctx.zero)?,
    )?;
    let uncertainty = c.where_cond(
        &positive_uncertainty,
        &bounds::upward_add(ctx, &sources.supply.bridge.uncertainty, &partition.bound)?,
        &ctx.zero,
    )?;
    let uncertainty_charge = bounds::sum_upper(ctx, &uncertainty)?;
    Ok(TensorSubareaStep {
        old_depth: sources.supply.old_depth,
        rain_rate: sources.supply.rain_rate,
        potential_evaporation: sources.supply.potential_evaporation,
        external_volume: sources.supply.external_volume,
        received_by_donor: sources.supply.received_by_donor,
        merged: sources.supply.merged,
        bridge: sources.supply.bridge,
        soil: sources.soil,
        infiltration_depth: sources.infiltration_depth,
        adjustment_depth: sources.adjustment_depth,
        evaporation_depth: sources.evaporation_depth,
        available_depth: sources.available_depth,
        timing_rate,
        direct_timing: sources.direct_timing,
        active_duration,
        storage_before: sources.supply.storage_before,
        rain_volume: sources.supply.rain_volume,
        internal_received_volume: sources.internal_received_volume,
        infiltration_volume: sources.infiltration_volume,
        evaporation_volume: sources.evaporation_volume,
        adjustment_volume: sources.adjustment_volume,
        water_residual,
        water_scale,
        rain_area_bound: sources.rain_area_bound,
        adjustment_charge: sources.adjustment_charge,
        uncertainty_charge,
        physical_scale_lower: sources.physical_scale_lower,
        drainage,
        partition,
    })
}
