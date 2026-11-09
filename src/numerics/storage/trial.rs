//! Atomic authoritative-volume exchanges with Numr TwoSum witnesses.

use crate::cpu::{
    CpuTensor,
    level_pool::{
        CpuPoolContext, CpuPoolSourceInput, CpuPoolSourceState, CpuPoolSourceStep, CpuPoolState,
        CpuPoolStep,
    },
};
use crate::storage::LevelPoolError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, UnaryOps};

pub(crate) fn exact_zero_add(
    context: &CpuPoolContext,
    a: &CpuTensor,
    b: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), LevelPoolError> {
    let c = &context.client;
    let (high, low) = c.two_sum(a, b)?;
    let zero = c.eq(b, &context.zero)?;
    let high = c.where_cond(&zero, a, &high)?;
    let low = c.where_cond(&zero, &context.zero, &low)?;
    context.finite(&high, "TwoSum high component")?;
    context.finite(&low, "TwoSum signed witness")?;
    context.reject(
        &c.gt(&c.abs(&low)?, &super::bounds::rounding(context, &high)?)?,
        "TwoSum witness magnitude",
    )?;
    Ok((high, low))
}

pub(crate) fn advance(
    context: &CpuPoolContext,
    state: &CpuPoolState,
    incoming: &CpuTensor,
    outgoing: &CpuTensor,
    steps: usize,
) -> Result<(CpuPoolState, CpuPoolStep), LevelPoolError> {
    advance_prepared(context, state, incoming, outgoing, steps, None)
}

pub(crate) fn advance_with_sources(
    context: &CpuPoolContext,
    state: &CpuPoolState,
    incoming: &CpuTensor,
    outgoing: &CpuTensor,
    steps: usize,
    dt: f64,
    input: &CpuPoolSourceInput,
) -> Result<(CpuPoolState, CpuPoolStep), LevelPoolError> {
    let prepared = super::sources::prepare(context, state, dt, input)?;
    advance_prepared(context, state, incoming, outgoing, steps, prepared)
}

fn advance_prepared(
    context: &CpuPoolContext,
    state: &CpuPoolState,
    incoming: &CpuTensor,
    outgoing: &CpuTensor,
    steps: usize,
    prepared: Option<(CpuPoolSourceStep, CpuPoolSourceState)>,
) -> Result<(CpuPoolState, CpuPoolStep), LevelPoolError> {
    let c = &context.client;
    let base = prepared
        .as_ref()
        .map_or(&state.volume_m3, |(s, _)| &s.post_source_volume_m3);
    context.nonnegative(incoming, "original incoming volume")?;
    context.nonnegative(outgoing, "original outgoing volume")?;
    context.reject(
        &c.gt(outgoing, base)?,
        if prepared.is_some() {
            "outgoing volume exceeds post-source donor storage without external incoming credit"
        } else {
            "outgoing volume exceeds initial donor storage without incoming credit"
        },
    )?;
    let (remaining, outgoing_witness) = exact_zero_add(context, base, &c.neg(outgoing)?)?;
    context.reject(
        &c.mul(&c.gt(outgoing, &context.zero)?, &c.eq(&remaining, base)?)?,
        "positive outgoing volume is absorbed into unchanged storage",
    )?;
    let (volume, incoming_witness) = exact_zero_add(context, &remaining, incoming)?;
    context.reject(
        &c.mul(&c.gt(incoming, &context.zero)?, &c.eq(&volume, &remaining)?)?,
        "positive incoming volume is absorbed into unchanged storage",
    )?;
    context.range(&volume, "candidate volume outside closed curve support")?;
    let stage = super::curves::evaluate(context, &volume)?;
    let (cum_in, in_witness) =
        exact_zero_add(context, &state.cumulative_incoming_volume_m3, incoming)?;
    let (cum_out, out_witness) =
        exact_zero_add(context, &state.cumulative_outgoing_volume_m3, outgoing)?;
    let in_bound =
        super::bounds::accumulate(context, &state.incoming_roundoff_bound_m3, &in_witness)?;
    let out_bound =
        super::bounds::accumulate(context, &state.outgoing_roundoff_bound_m3, &out_witness)?;
    let a = c.sub(&volume, &state.volume_m3)?;
    let b = c.sub(&a, incoming)?;
    let residual = c.add(&b, outgoing)?;
    let witness_sum = c.add(&outgoing_witness, &incoming_witness)?;
    let consistency = c.add(&residual, &witness_sum)?;
    let mut bound = super::bounds::rounding(context, &a)?;
    for value in [&b, &residual, &witness_sum, &consistency] {
        bound = super::bounds::add(context, &bound, &super::bounds::rounding(context, value)?)?;
    }
    let scale = c.maximum(
        &c.maximum(&state.volume_m3, &volume)?,
        &c.maximum(incoming, outgoing)?,
    )?;
    let legacy = super::source_accounting::Interval {
        residual,
        consistency,
        bound,
        scale,
    };
    let checked = if let Some((s, _)) = &prepared {
        super::source_accounting::interval(
            context,
            s,
            &a,
            &b,
            &legacy,
            &outgoing_witness,
            &incoming_witness,
        )?
    } else {
        legacy
    };
    let super::source_accounting::Interval {
        residual,
        consistency,
        bound,
        scale,
    } = checked;
    super::accounting::check_residual(
        context,
        &residual,
        &scale,
        "per-element interval physical residual",
    )?;
    context.reject(
        &c.gt(&c.abs(&consistency)?, &bound)?,
        "storage witness consistency residual",
    )?;
    let next = CpuPoolState {
        sources: prepared
            .as_ref()
            .map(|(_, state)| state.clone())
            .or_else(|| state.sources.clone()),
        volume_m3: volume.clone(),
        initial_volume_m3: state.initial_volume_m3.clone(),
        cumulative_incoming_volume_m3: cum_in.clone(),
        cumulative_outgoing_volume_m3: cum_out.clone(),
        incoming_roundoff_bound_m3: in_bound.clone(),
        outgoing_roundoff_bound_m3: out_bound.clone(),
    };
    super::accounting::check_state(context, &next, steps)?;
    let step = CpuPoolStep {
        source: prepared.map(|(step, _)| step),
        old_volume_m3: state.volume_m3.clone(),
        incoming_volume_m3: incoming.clone(),
        outgoing_volume_m3: outgoing.clone(),
        remaining_volume_m3: remaining,
        volume_m3: volume,
        outgoing_witness_m3: outgoing_witness,
        incoming_witness_m3: incoming_witness,
        water_residual_m3: residual,
        witness_consistency_m3: consistency,
        witness_consistency_bound_m3: bound,
        physical_scale_m3: scale,
        stage_m: stage.stage,
        reconstructed_volume_m3: stage.reconstructed,
        inverse_residual_m3: stage.residual,
        cumulative_incoming_before_m3: state.cumulative_incoming_volume_m3.clone(),
        cumulative_incoming_after_m3: cum_in,
        cumulative_incoming_witness_m3: in_witness,
        cumulative_outgoing_before_m3: state.cumulative_outgoing_volume_m3.clone(),
        cumulative_outgoing_after_m3: cum_out,
        cumulative_outgoing_witness_m3: out_witness,
        incoming_bound_before_m3: state.incoming_roundoff_bound_m3.clone(),
        incoming_bound_after_m3: in_bound,
        outgoing_bound_before_m3: state.outgoing_roundoff_bound_m3.clone(),
        outgoing_bound_after_m3: out_bound,
    };
    Ok((next, step))
}
