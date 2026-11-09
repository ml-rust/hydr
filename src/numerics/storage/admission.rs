//! Independent stage, witness, and per-element receipt gates.

use crate::cpu::level_pool::{CpuPoolContext, CpuPoolStep};
use crate::storage::LevelPoolError;
use numr::ops::{CompareOps, UnaryOps};

pub(crate) fn check_step(
    context: &CpuPoolContext,
    step: &CpuPoolStep,
) -> Result<(), LevelPoolError> {
    let c = &context.client;
    context.range(&step.old_volume_m3, "receipt original storage")?;
    context.range(&step.volume_m3, "receipt accepted storage")?;
    context.nonnegative(&step.incoming_volume_m3, "receipt incoming exchange")?;
    context.nonnegative(&step.outgoing_volume_m3, "receipt outgoing exchange")?;
    context.reject(
        &c.gt(
            &step.outgoing_volume_m3,
            step.source
                .as_ref()
                .map_or(&step.old_volume_m3, |s| &s.post_source_volume_m3),
        )?,
        "receipt donor budget",
    )?;
    super::accounting::check_residual(
        context,
        &step.water_residual_m3,
        &step.physical_scale_m3,
        "receipt independent physical residual",
    )?;
    context.nonnegative(
        &step.witness_consistency_bound_m3,
        "receipt witness consistency bound",
    )?;
    context.reject(
        &c.gt(
            &c.abs(&step.witness_consistency_m3)?,
            &step.witness_consistency_bound_m3,
        )?,
        "receipt witness consistency",
    )?;
    for (witness, high) in [
        (&step.outgoing_witness_m3, &step.remaining_volume_m3),
        (&step.incoming_witness_m3, &step.volume_m3),
        (
            &step.cumulative_incoming_witness_m3,
            &step.cumulative_incoming_after_m3,
        ),
        (
            &step.cumulative_outgoing_witness_m3,
            &step.cumulative_outgoing_after_m3,
        ),
    ] {
        context.finite(witness, "receipt signed addition witness")?;
        context.reject(
            &c.gt(&c.abs(witness)?, &super::bounds::rounding(context, high)?)?,
            "receipt signed witness magnitude",
        )?;
    }
    Ok(())
}
