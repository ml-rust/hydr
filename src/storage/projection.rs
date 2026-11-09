//! Explicit projection boundaries. Numerical continuation never consumes projected arrays.

use super::accounting::read;
use super::batch::StoredStep;
use super::{
    LevelPoolElementLedger, LevelPoolError, LevelPoolOutput, LevelPoolReceipt,
    LevelPoolStepDiagnostics,
};
use crate::cpu::level_pool::{CpuPoolContext, CpuPoolState};

pub(crate) fn outputs(
    context: &CpuPoolContext,
    state: &CpuPoolState,
) -> Result<Vec<LevelPoolOutput>, LevelPoolError> {
    let stage = crate::numerics::storage::curves::evaluate(context, &state.volume_m3)?;
    let n = context.elements.len();
    let volume = read(&state.volume_m3, n)?;
    let heights = read(&stage.stage, n)?;
    Ok((0..n)
        .map(|element| LevelPoolOutput {
            element,
            volume_m3: volume[element],
            stage_m: if volume[element] == 0.0 {
                None
            } else {
                Some(heights[element])
            },
        })
        .collect())
}

pub(crate) fn ledgers(
    context: &CpuPoolContext,
    state: &CpuPoolState,
) -> Result<Vec<LevelPoolElementLedger>, LevelPoolError> {
    let n = context.elements.len();
    let sources = super::source_projection::ledgers(context, state.sources.as_ref())?;
    let initial_volume_m3 = read(&state.initial_volume_m3, n)?;
    let volume_m3 = read(&state.volume_m3, n)?;
    let cumulative_incoming_volume_m3 = read(&state.cumulative_incoming_volume_m3, n)?;
    let cumulative_outgoing_volume_m3 = read(&state.cumulative_outgoing_volume_m3, n)?;
    let incoming_roundoff_bound_m3 = read(&state.incoming_roundoff_bound_m3, n)?;
    let outgoing_roundoff_bound_m3 = read(&state.outgoing_roundoff_bound_m3, n)?;
    let residual = read(
        &crate::numerics::storage::accounting::residual(context, state)?,
        n,
    )?;
    Ok((0..n)
        .map(|element| LevelPoolElementLedger {
            sources: sources.as_ref().map(|s| s[element].clone()),
            element,
            initial_volume_m3: initial_volume_m3[element],
            volume_m3: volume_m3[element],
            cumulative_incoming_volume_m3: cumulative_incoming_volume_m3[element],
            cumulative_outgoing_volume_m3: cumulative_outgoing_volume_m3[element],
            incoming_roundoff_bound_m3: incoming_roundoff_bound_m3[element],
            outgoing_roundoff_bound_m3: outgoing_roundoff_bound_m3[element],
            residual_m3: residual[element],
        })
        .collect())
}

pub(crate) fn step(
    context: &CpuPoolContext,
    stored: &StoredStep,
) -> Result<LevelPoolStepDiagnostics, LevelPoolError> {
    crate::numerics::storage::check_step(context, &stored.step)?;
    let totals = super::accounting::step_totals(context, &stored.step)?;
    let source = super::source_projection::receipts(context, stored.step.source.as_ref())?;
    let source_totals =
        super::source_projection::step_totals(context, stored.step.source.as_ref())?;
    let n = context.elements.len();
    let old_volume_m3 = read(&stored.step.old_volume_m3, n)?;
    let incoming_volume_m3 = read(&stored.step.incoming_volume_m3, n)?;
    let outgoing_volume_m3 = read(&stored.step.outgoing_volume_m3, n)?;
    let remaining_volume_m3 = read(&stored.step.remaining_volume_m3, n)?;
    let volume_m3 = read(&stored.step.volume_m3, n)?;
    let outgoing_witness_m3 = read(&stored.step.outgoing_witness_m3, n)?;
    let incoming_witness_m3 = read(&stored.step.incoming_witness_m3, n)?;
    let water_residual_m3 = read(&stored.step.water_residual_m3, n)?;
    let witness_consistency_m3 = read(&stored.step.witness_consistency_m3, n)?;
    let witness_consistency_bound_m3 = read(&stored.step.witness_consistency_bound_m3, n)?;
    let physical_scale_m3 = read(&stored.step.physical_scale_m3, n)?;
    let stage_m = read(&stored.step.stage_m, n)?;
    let reconstructed_volume_m3 = read(&stored.step.reconstructed_volume_m3, n)?;
    let inverse_residual_m3 = read(&stored.step.inverse_residual_m3, n)?;
    let cumulative_incoming_before_m3 = read(&stored.step.cumulative_incoming_before_m3, n)?;
    let cumulative_incoming_after_m3 = read(&stored.step.cumulative_incoming_after_m3, n)?;
    let cumulative_incoming_witness_m3 = read(&stored.step.cumulative_incoming_witness_m3, n)?;
    let cumulative_outgoing_before_m3 = read(&stored.step.cumulative_outgoing_before_m3, n)?;
    let cumulative_outgoing_after_m3 = read(&stored.step.cumulative_outgoing_after_m3, n)?;
    let cumulative_outgoing_witness_m3 = read(&stored.step.cumulative_outgoing_witness_m3, n)?;
    let incoming_bound_before_m3 = read(&stored.step.incoming_bound_before_m3, n)?;
    let incoming_bound_after_m3 = read(&stored.step.incoming_bound_after_m3, n)?;
    let outgoing_bound_before_m3 = read(&stored.step.outgoing_bound_before_m3, n)?;
    let outgoing_bound_after_m3 = read(&stored.step.outgoing_bound_after_m3, n)?;
    let receipts = (0..n)
        .map(|element| LevelPoolReceipt {
            source: source[element].clone(),
            element,
            old_volume_m3: old_volume_m3[element],
            incoming_volume_m3: incoming_volume_m3[element],
            outgoing_volume_m3: outgoing_volume_m3[element],
            remaining_volume_m3: remaining_volume_m3[element],
            volume_m3: volume_m3[element],
            outgoing_witness_m3: outgoing_witness_m3[element],
            incoming_witness_m3: incoming_witness_m3[element],
            water_residual_m3: water_residual_m3[element],
            witness_consistency_m3: witness_consistency_m3[element],
            witness_consistency_bound_m3: witness_consistency_bound_m3[element],
            physical_scale_m3: physical_scale_m3[element],
            stage_m: if volume_m3[element] == 0.0 {
                None
            } else {
                Some(stage_m[element])
            },
            reconstructed_volume_m3: reconstructed_volume_m3[element],
            inverse_residual_m3: inverse_residual_m3[element],
            cumulative_incoming_before_m3: cumulative_incoming_before_m3[element],
            cumulative_incoming_after_m3: cumulative_incoming_after_m3[element],
            cumulative_incoming_witness_m3: cumulative_incoming_witness_m3[element],
            cumulative_outgoing_before_m3: cumulative_outgoing_before_m3[element],
            cumulative_outgoing_after_m3: cumulative_outgoing_after_m3[element],
            cumulative_outgoing_witness_m3: cumulative_outgoing_witness_m3[element],
            incoming_bound_before_m3: incoming_bound_before_m3[element],
            incoming_bound_after_m3: incoming_bound_after_m3[element],
            outgoing_bound_before_m3: outgoing_bound_before_m3[element],
            outgoing_bound_after_m3: outgoing_bound_after_m3[element],
        })
        .collect();
    Ok(LevelPoolStepDiagnostics {
        rain_volume_m3: source_totals.rain_volume_m3,
        evaporation_volume_m3: source_totals.evaporation_volume_m3,
        bottom_loss_volume_m3: source_totals.bottom_loss_volume_m3,
        bottom_loss_demand_m3: source_totals.bottom_loss_demand_m3,
        unmet_bottom_loss_m3: source_totals.unmet_bottom_loss_m3,
        potential_evaporation_volume_m3: source_totals.potential_evaporation_volume_m3,
        unmet_evaporation_volume_m3: source_totals.unmet_evaporation_volume_m3,

        start: stored.start,
        end: stored.end,
        dt: stored.end - stored.start,
        volume_before_m3: totals.before,
        volume_after_m3: totals.after,
        incoming_volume_m3: totals.incoming,
        outgoing_volume_m3: totals.outgoing,
        water_residual_m3: totals.residual,
        physical_scale_m3: totals.scale,
        reduction_bound_m3: totals.bound,
        receipts,
    })
}
