//! Frozen-area rain, capped bottom demand, and capped evaporation before exchanges.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use crate::storage::LevelPoolError;
use crate::storage::tensor::{
    TensorPoolContext, TensorPoolSourceInput, TensorPoolSourceState, TensorPoolSourceStep,
    TensorPoolState,
};

fn product<R>(
    context: &TensorPoolContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    quantity: &str,
) -> Result<Tensor<R>, LevelPoolError>
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let c = &context.client;
    let result = c.mul(a, b)?;
    context.nonnegative(&result, quantity)?;
    context.reject(
        &c.mul(&c.gt(a, &context.zero)?, &c.eq(&result, &context.zero)?)?,
        quantity,
    )?;
    Ok(result)
}

pub(crate) type PreparedSources<R> = (TensorPoolSourceStep<R>, TensorPoolSourceState<R>);

pub(crate) fn prepare<R>(
    context: &TensorPoolContext<R>,
    state: &TensorPoolState<R>,
    dt: f64,
    input: &TensorPoolSourceInput<R>,
) -> Result<Option<PreparedSources<R>>, LevelPoolError>
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
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let c = &context.client;
    let active = c.maximum(
        &c.gt(&input.rain, &context.zero)?,
        &c.maximum(
            &c.gt(&input.evaporation, &context.zero)?,
            &c.gt(&input.bottom_loss, &context.zero)?,
        )?,
    )?;
    if !context.any(&active)? {
        return Ok(None);
    }
    let previous = state.sources.as_ref().ok_or_else(|| {
        LevelPoolError::input("source-aware trial requires immutable source policies")
    })?;
    super::curves::evaluate(context, &state.volume_m3)?;
    let (area, segment) = super::source_area::select(context, &state.volume_m3)?;
    let rain_rate = product(
        context,
        &input.rain,
        &area,
        "positive rain rate-area product is unrepresentable",
    )?;
    let evap_rate = product(
        context,
        &input.evaporation,
        &area,
        "positive potential evaporation rate-area product is unrepresentable",
    )?;
    let duration = c.add_scalar(&context.zero, dt)?;
    let rain = product(
        context,
        &rain_rate,
        &duration,
        "positive interval rain product is unrepresentable",
    )?;
    context.reject(&c.gt(&rain,&context.maximum)?,"actual source volume exceeds capacity-based source-step resolution. Use a shorter interval without clipping")?;
    let evap = product(
        context,
        &evap_rate,
        &duration,
        "positive potential evaporation volume is unrepresentable",
    )?;
    let bottom = product(
        context,
        &input.bottom_loss,
        &duration,
        "positive potential bottom-demand volume is unrepresentable",
    )?;
    let (after_rain, rain_witness) =
        super::trial::exact_zero_add(context, &state.volume_m3, &rain)?;
    context.reject(
        &c.mul(
            &c.gt(&rain, &context.zero)?,
            &c.eq(&after_rain, &state.volume_m3)?,
        )?,
        "positive rain is absorbed. Rescale storage/source magnitudes or choose an interval with resolvable process volume",
    )?;
    let (bottom_cap, after_bottom, bottom_witness) =
        super::source_caps::cap(context, &bottom, &after_rain)?;
    let (evap_cap, post_source, evap_witness) =
        super::source_caps::cap(context, &evap, &after_bottom)?;
    for volume in [&bottom_cap.actual_volume_m3, &evap_cap.actual_volume_m3] {
        context.reject(&c.gt(volume,&context.maximum)?,"actual source volume exceeds capacity-based source-step resolution. Use a shorter interval without clipping")?;
    }
    let stage = super::curves::evaluate(context, &post_source)?;
    let rain_counter =
        super::source_caps::counter(context, &previous.rain, &previous.rain_bound, &rain)?;
    let evaporation_counter = super::source_caps::counter(
        context,
        &previous.evaporation,
        &previous.evaporation_bound,
        &evap_cap.actual_volume_m3,
    )?;
    let bottom_loss_counter = super::source_caps::counter(
        context,
        &previous.bottom_loss,
        &previous.bottom_loss_bound,
        &bottom_cap.actual_volume_m3,
    )?;
    let bottom_loss_demand_counter = super::source_caps::counter(
        context,
        &previous.bottom_loss_demand,
        &previous.bottom_loss_demand_bound,
        &bottom,
    )?;
    let next = TensorPoolSourceState {
        rain: rain_counter.after_m3.clone(),
        evaporation: evaporation_counter.after_m3.clone(),
        bottom_loss: bottom_loss_counter.after_m3.clone(),
        bottom_loss_demand: bottom_loss_demand_counter.after_m3.clone(),
        rain_bound: rain_counter.bound_after_m3.clone(),
        evaporation_bound: evaporation_counter.bound_after_m3.clone(),
        bottom_loss_bound: bottom_loss_counter.bound_after_m3.clone(),
        bottom_loss_demand_bound: bottom_loss_demand_counter.bound_after_m3.clone(),
    };
    Ok(Some((
        TensorPoolSourceStep {
            frozen_area_m2: area,
            segment,
            active,
            rain_rate_m_per_s: input.rain.clone(),
            potential_evaporation_rate_m_per_s: input.evaporation.clone(),
            bottom_loss_rate_m3_per_s: input.bottom_loss.clone(),
            rain_rate_volume_m3_per_s: rain_rate,
            rain_volume_m3: rain,
            potential_evaporation_rate_volume_m3_per_s: evap_rate,
            potential_evaporation_volume_m3: evap,
            after_rain_volume_m3: after_rain,
            after_bottom_loss_volume_m3: after_bottom,
            post_source_volume_m3: post_source,
            rain_witness_m3: rain_witness,
            bottom_loss_witness_m3: bottom_witness,
            evaporation_witness_m3: evap_witness,
            post_source_stage_m: stage.stage,
            post_source_reconstructed_volume_m3: stage.reconstructed,
            post_source_inverse_residual_m3: stage.residual,
            bottom_loss_cap: bottom_cap,
            evaporation_cap: evap_cap,
            rain_counter,
            evaporation_counter,
            bottom_loss_counter,
            bottom_loss_demand_counter,
        },
        next,
    )))
}
