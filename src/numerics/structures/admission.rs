//! Known-support physical eligibility before equal-head or ideal-flap shortcuts.

use crate::cpu::structures::{CpuStructureContext, CpuStructureInput};
use crate::structures::StructureError;
use numr::ops::{BinaryOps, CompareOps};

pub(super) fn check(
    context: &CpuStructureContext,
    input: &CpuStructureInput,
) -> Result<(), StructureError> {
    let c = &context.client;
    let wet = c.mul(&input.wet_a, &input.wet_b)?;
    let eligible = c.mul(
        &wet,
        &c.mul(
            &c.gt(&input.stage_a, &context.crown)?,
            &c.gt(&input.stage_b, &context.crown)?,
        )?,
    )?;
    context.reject_support(
        &c.mul(&context.orifice, &c.sub(&context.one, &eligible)?)?,
        "fully submerged orifice requires both known wet stages strictly above crown",
    )?;
    let dry_a = c.sub(&context.one, &input.wet_a)?;
    let dry_b = c.sub(&context.one, &input.wet_b)?;
    let invalid_dry = c.maximum(
        &c.mul(&dry_a, &c.gt(&input.bed_a, &context.crest)?)?,
        &c.mul(&dry_b, &c.gt(&input.bed_b, &context.crest)?)?,
    )?;
    context.reject_support(
        &c.mul(&context.weir, &invalid_dry)?,
        "known dry weir bed must lie at or below crest without a fabricated water stage",
    )
}
