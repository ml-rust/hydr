//! Known-support physical eligibility before equal-head or ideal-flap shortcuts.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::{TensorStructureContext, TensorStructureInput};

pub(super) fn check<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
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
