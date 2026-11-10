//! Physical direction and regime from original support, before rounded depth arithmetic.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::ScalarOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::{TensorStructureContext, TensorStructureInput, codes};
use numr::tensor::Tensor;

pub(super) struct Classification<R: Runtime<DType = DType>> {
    pub active: Tensor<R>,
    pub direction: Tensor<R>,
    pub regime: Tensor<R>,
    pub status: Tensor<R>,
    pub orifice: Tensor<R>,
    pub free_weir: Tensor<R>,
    pub submerged_weir: Tensor<R>,
    pub high: Tensor<R>,
    pub low: Tensor<R>,
}

pub(super) fn classify<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
) -> Result<Classification<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let a_high = c.gt(&input.stage_a, &input.stage_b)?;
    let b_high = c.gt(&input.stage_b, &input.stage_a)?;
    let equal = c.eq(&input.stage_a, &input.stage_b)?;
    let above_a = c.mul(&input.wet_a, &c.gt(&input.stage_a, &context.crest)?)?;
    let above_b = c.mul(&input.wet_b, &c.gt(&input.stage_b, &context.crest)?)?;
    let above_both = c.mul(&above_a, &above_b)?;
    let above_any = c.maximum(&above_a, &above_b)?;
    let weir_a = c.mul(
        &context.weir,
        &c.mul(
            &above_a,
            &c.add(
                &c.sub(&context.one, &input.wet_b)?,
                &c.mul(&input.wet_b, &a_high)?,
            )?,
        )?,
    )?;
    let weir_b = c.mul(
        &context.weir,
        &c.mul(
            &above_b,
            &c.add(
                &c.sub(&context.one, &input.wet_a)?,
                &c.mul(&input.wet_a, &b_high)?,
            )?,
        )?,
    )?;
    let nominal_a = c.add(&c.mul(&context.orifice, &a_high)?, &weir_a)?;
    let nominal_b = c.add(&c.mul(&context.orifice, &b_high)?, &weir_b)?;
    let nominal = c.add(&nominal_a, &nominal_b)?;
    let allowed = c.add(
        &c.mul(&nominal_a, &context.allow_a)?,
        &c.mul(&nominal_b, &context.allow_b)?,
    )?;
    let active = c.mul(&nominal, &allowed)?;
    let blocked = c.mul(&nominal, &c.sub(&context.one, &allowed)?)?;
    let orifice = c.mul(&context.orifice, &c.sub(&context.one, &equal)?)?;
    let submerged = c.mul(
        &context.weir,
        &c.mul(&above_both, &c.sub(&context.one, &equal)?)?,
    )?;
    let free = c.mul(
        &context.weir,
        &c.mul(&above_any, &c.sub(&context.one, &above_both)?)?,
    )?;
    let below = c.mul(&context.weir, &c.sub(&context.one, &above_any)?)?;
    let equal_regime = c.mul(
        &equal,
        &c.add(&context.orifice, &c.mul(&context.weir, &above_both)?)?,
    )?;
    let regime = c.add(
        &c.add(
            &c.mul_scalar(&orifice, codes::ORIFICE)?,
            &c.mul_scalar(&free, codes::FREE_WEIR)?,
        )?,
        &c.add(
            &c.add(
                &c.mul_scalar(&submerged, codes::SUBMERGED_WEIR)?,
                &c.mul_scalar(&below, codes::BELOW_CREST)?,
            )?,
            &c.mul_scalar(&equal_regime, codes::EQUAL_HEADS)?,
        )?,
    )?;
    Ok(Classification {
        direction: c.sub(&nominal_a, &nominal_b)?,
        regime,
        status: c.add(
            &c.mul_scalar(&active, codes::ACTIVE)?,
            &c.mul_scalar(&blocked, codes::BLOCKED)?,
        )?,
        orifice: c.mul(&orifice, &active)?,
        free_weir: c.mul(&free, &active)?,
        submerged_weir: c.mul(&submerged, &active)?,
        high: c.where_cond(&nominal_a, &input.stage_a, &input.stage_b)?,
        low: c.where_cond(&nominal_a, &input.stage_b, &input.stage_a)?,
        active,
    })
}
