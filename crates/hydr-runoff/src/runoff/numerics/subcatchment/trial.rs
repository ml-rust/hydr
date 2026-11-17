//! Atomic two-phase advancement with persistent topology and paired interface volumes.

use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::sources::Supply;
use super::step::{TensorSubareaStep, TensorSubcatchmentStep};
use super::{accounting, bounds, phase, sources};
use crate::runoff::SubcatchmentError;
use crate::runoff::numerics::reservoir::drainage;
use crate::runoff::tensor::subcatchment::{
    TensorSubareaGroup, TensorSubcatchmentContext, TensorSubcatchmentInput, TensorSubcatchmentState,
};
use numr::dtype::DType;
use numr::runtime::Device;

pub(crate) fn advance<R>(
    ctx: &TensorSubcatchmentContext<R>,
    state: &TensorSubcatchmentState<R>,
    dt: f64,
    input: &TensorSubcatchmentInput<R>,
) -> Result<(TensorSubcatchmentState<R>, TensorSubcatchmentStep<R>), SubcatchmentError>
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
    check_inputs(ctx, &state.depth, input, dt)?;
    let mut received: [Tensor<R>; 3] = std::array::from_fn(|_| ctx.zero.clone());
    let mut depth = state.depth.clone();
    let (first, phase1_soil) = match &ctx.phase1 {
        Some(group) => {
            let supply = gather_supply(ctx, group, &state.depth, input, &received, dt)?;
            let (step, soil) = phase::advance(group, supply, state.phase1_soil.as_ref(), dt)?;
            route(ctx, group, &step, &mut received)?;
            depth = ctx
                .client
                .scatter(&depth, 0, &group.indices, &step.drainage.depth)?;
            (Some(step), soil)
        }
        None if state.phase1_soil.is_none() => (None, None),
        None => {
            return Err(SubcatchmentError::input(
                "absent first phase cannot contain native soil memory",
            ));
        }
    };
    let (second, phase2_soil) = match &ctx.phase2 {
        Some(group) => {
            let supply = gather_supply(ctx, group, &state.depth, input, &received, dt)?;
            let (step, soil) = phase::advance(group, supply, state.phase2_soil.as_ref(), dt)?;
            route(ctx, group, &step, &mut received)?;
            depth = ctx
                .client
                .scatter(&depth, 0, &group.indices, &step.drainage.depth)?;
            (Some(step), soil)
        }
        None if state.phase2_soil.is_none() => (None, None),
        None => {
            return Err(SubcatchmentError::input(
                "absent second phase cannot contain native soil memory",
            ));
        }
    };
    let totals = accounting::totals(
        ctx,
        input,
        [
            ctx.phase1.as_ref().zip(first.as_ref()),
            ctx.phase2.as_ref().zip(second.as_ref()),
        ],
        dt,
    )?;
    let next = TensorSubcatchmentState {
        depth: depth.clone(),
        phase1_soil,
        phase2_soil,
    };
    let step = TensorSubcatchmentStep {
        old_depth: state.depth.clone(),
        input: input.clone(),
        depth,
        phase1: first,
        phase2: second,
        received_by_donor: received,
        totals,
    };
    Ok((next, step))
}

pub(crate) fn initial_storage_scale_lower<R>(
    ctx: &TensorSubcatchmentContext<R>,
    state: &TensorSubcatchmentState<R>,
) -> Result<f64, SubcatchmentError>
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
    let mut lower = 0.0;
    for group in [ctx.phase1.as_ref(), ctx.phase2.as_ref()]
        .into_iter()
        .flatten()
    {
        let depth = ctx.client.gather(&state.depth, 0, &group.indices)?;
        let volume = drainage::volume(
            &group.reservoir,
            &depth,
            "initial uncertainty-scale storage",
        )
        .map_err(|error| group.map_reservoir_error(error))?;
        let group_lower = bounds::sum_lower(&group.reservoir, &volume)
            .map_err(|error| group.map_reservoir_error(error))?;
        lower = bounds::add_scale_lower(lower, group_lower)?;
    }
    Ok(lower)
}

pub(super) fn gather_supply<R>(
    ctx: &TensorSubcatchmentContext<R>,
    group: &TensorSubareaGroup<R>,
    old: &Tensor<R>,
    input: &TensorSubcatchmentInput<R>,
    received: &[Tensor<R>; 3],
    dt: f64,
) -> Result<Supply<R>, SubcatchmentError>
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
    let gather = |value| ctx.client.gather(value, 0, &group.indices);
    sources::supply(
        group,
        gather(old)?,
        gather(&input.rain)?,
        gather(&input.evaporation)?,
        gather(&input.external)?,
        [
            gather(&received[0])?,
            gather(&received[1])?,
            gather(&received[2])?,
        ],
        dt,
    )
    .map_err(|error| group.map_reservoir_error(error))
}

pub(super) fn route<R>(
    ctx: &TensorSubcatchmentContext<R>,
    group: &TensorSubareaGroup<R>,
    step: &TensorSubareaStep<R>,
    received: &mut [Tensor<R>; 3],
) -> Result<(), SubcatchmentError>
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
    for (destination, map) in received.iter_mut().zip(&group.routes) {
        if let Some(map) = map {
            let shared_volume =
                ctx.client
                    .gather(&step.partition.routed, 0, &map.source_indices)?;
            *destination =
                ctx.client
                    .scatter(destination, 0, &map.destination_indices, &shared_volume)?;
        }
    }
    Ok(())
}

pub(super) fn check_inputs<R>(
    ctx: &TensorSubcatchmentContext<R>,
    old: &Tensor<R>,
    input: &TensorSubcatchmentInput<R>,
    dt: f64,
) -> Result<(), SubcatchmentError>
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
    if !dt.is_finite() || dt <= 0.0 {
        return Err(SubcatchmentError::input(format!(
            "dt={dt}: require a positive finite complete interval"
        )));
    }
    let c = &ctx.client;
    for (name, value) in [
        ("initial subarea depth", old),
        ("rain rate", &input.rain),
        ("potential evaporation", &input.evaporation),
        ("external runon", &input.external),
    ] {
        metadata(ctx, value, name)?;
        ctx.finite(value, name)?;
        ctx.reject_mask(
            &c.lt(value, &ctx.zero)?,
            name,
            &[("value", value)],
            &[],
            "Supply finite nonnegative complete component inputs",
        )?;
    }
    let absent = c.eq(&ctx.area, &ctx.zero)?;
    for (name, value) in [
        ("absent subarea depth", old),
        ("absent external target", &input.external),
    ] {
        ctx.reject_mask(
            &c.mul(&absent, &c.ne(value, &ctx.zero)?)?,
            name,
            &[("value", value)],
            &[],
            "Absent physical support requires explicit zero bookkeeping",
        )?;
    }
    for (name, forcing) in [
        ("shared rainfall", &input.rain),
        ("shared potential evaporation", &input.evaporation),
    ] {
        let by_catchment = c.gather(forcing, 0, &ctx.catchment_slots)?;
        let repeated = c.gather(&by_catchment, 0, &ctx.catchment_indices)?;
        ctx.reject_mask(
            &c.ne(forcing, &repeated)?,
            name,
            &[("value", forcing), ("expected", &repeated)],
            &[],
            "Each catchment uses one common rate across all three subarea slots",
        )?;
    }
    Ok(())
}

pub(super) fn metadata<R>(
    ctx: &TensorSubcatchmentContext<R>,
    value: &Tensor<R>,
    quantity: &str,
) -> Result<(), SubcatchmentError>
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
    if value.shape() != ctx.zero.shape()
        || value.dtype() != DType::F64
        || !value.device().is_same(ctx.zero.device())
    {
        return Err(SubcatchmentError::input(format!(
            "{quantity}: require configured CPU F64 flat support {:?}",
            ctx.zero.shape()
        )));
    }
    Ok(())
}
