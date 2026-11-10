//! Exact supported-head topology and crown-expansion comparisons before flow arithmetic.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::ScalarOps;
use numr::ops::TwoSumOps;
use numr::ops::TypeConversionOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::{
    TensorStructureContext, TensorStructureInput, codes,
    gate::{TensorGateContext, TensorGateInput},
};
use numr::tensor::Tensor;

pub(super) struct Classification<R: Runtime<DType = DType>> {
    pub active: Tensor<R>,
    pub free: Tensor<R>,
    pub submerged: Tensor<R>,
    pub fully_submerged: Tensor<R>,
    pub donor_full: Tensor<R>,
    pub receiver_above: Tensor<R>,
    pub upper_positive: Tensor<R>,
    pub donor: Tensor<R>,
    pub receiver: Tensor<R>,
    pub crown_high: Tensor<R>,
    pub crown_low: Tensor<R>,
    pub crown_overflow: Tensor<R>,
    pub regime: Tensor<R>,
    pub status: Tensor<R>,
    pub direction: Tensor<R>,
}

fn at_or_above<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    stage: &Tensor<R>,
    high: &Tensor<R>,
    low: &Tensor<R>,
    overflow: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    Ok(c.mul(
        &c.sub(&context.one, overflow)?,
        &c.add(
            &c.gt(stage, high)?,
            &c.mul(&c.eq(stage, high)?, &c.le(low, &context.zero)?)?,
        )?,
    )?)
}

pub(super) fn classify<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    gate: &TensorGateContext<R>,
    input: &TensorStructureInput<R>,
    operation: &TensorGateInput<R>,
) -> Result<Classification<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let open = c.mul(&gate.mask, &c.gt(&operation.opening, &context.zero)?)?;
    let invalid_dry = c.maximum(
        &c.mul(
            &c.sub(&context.one, &input.wet_a)?,
            &c.gt(&input.bed_a, &gate.sill)?,
        )?,
        &c.mul(
            &c.sub(&context.one, &input.wet_b)?,
            &c.gt(&input.bed_b, &gate.sill)?,
        )?,
    )?;
    context.reject_support(
        &c.mul(&open, &invalid_dry)?,
        "positive-opening gate requires each known dry bed at or below its sill",
    )?;
    let above_a = c.mul(&input.wet_a, &c.gt(&input.stage_a, &gate.sill)?)?;
    let above_b = c.mul(&input.wet_b, &c.gt(&input.stage_b, &gate.sill)?)?;
    let nominal_a = c.mul(
        &open,
        &c.mul(
            &above_a,
            &c.add(
                &c.sub(&context.one, &input.wet_b)?,
                &c.mul(&input.wet_b, &c.gt(&input.stage_a, &input.stage_b)?)?,
            )?,
        )?,
    )?;
    let nominal_b = c.mul(
        &open,
        &c.mul(
            &above_b,
            &c.add(
                &c.sub(&context.one, &input.wet_a)?,
                &c.mul(&input.wet_a, &c.gt(&input.stage_b, &input.stage_a)?)?,
            )?,
        )?,
    )?;
    let nominal = c.add(&nominal_a, &nominal_b)?;
    let active = c.add(
        &c.mul(&nominal_a, &context.allow_a)?,
        &c.mul(&nominal_b, &context.allow_b)?,
    )?;
    let blocked = c.sub(&nominal, &active)?;
    let equal = c.mul(
        &open,
        &c.mul(
            &c.mul(&above_a, &above_b)?,
            &c.eq(&input.stage_a, &input.stage_b)?,
        )?,
    )?;
    let below = c.sub(&c.sub(&open, &nominal)?, &equal)?;
    let closed = c.sub(&gate.mask, &open)?;
    let donor = c.where_cond(&nominal_a, &input.stage_a, &input.stage_b)?;
    let receiver = c.where_cond(&nominal_a, &input.stage_b, &input.stage_a)?;
    let receiver_wet = c.where_cond(&nominal_a, &input.wet_b, &input.wet_a)?;
    let receiver_above = c.mul(&receiver_wet, &c.gt(&receiver, &gate.sill)?)?;

    let used_sill = c.where_cond(&open, &gate.sill, &context.zero)?;
    let raw_crown = c.add(&used_sill, &operation.opening)?;
    let overflow = c.cast(&c.isinf(&raw_crown)?, raw_crown.dtype())?;
    let finite_crown = c.mul(&open, &c.sub(&context.one, &overflow)?)?;
    let (crown_high, crown_low) = c.two_sum(
        &c.where_cond(&finite_crown, &gate.sill, &context.zero)?,
        &c.where_cond(&finite_crown, &operation.opening, &context.zero)?,
    )?;
    context.finite(&crown_high, "finite gate crown expansion high part")?;
    context.finite(&crown_low, "finite gate crown expansion low part")?;
    let donor_full = at_or_above(context, &donor, &crown_high, &crown_low, &overflow)?;
    let receiver_full = c.mul(
        &receiver_wet,
        &at_or_above(context, &receiver, &crown_high, &crown_low, &overflow)?,
    )?;
    let free = c.mul(&active, &c.sub(&context.one, &receiver_full)?)?;
    let submerged = c.mul(&active, &receiver_above)?;
    let fully_submerged = c.mul(&active, &receiver_full)?;
    let exact_crown = c.mul(
        &c.eq(&donor, &crown_high)?,
        &c.eq(&crown_low, &context.zero)?,
    )?;
    let upper_positive = c.mul(
        &free,
        &c.mul(&donor_full, &c.sub(&context.one, &exact_crown)?)?,
    )?;
    let free_regime = c.mul(&nominal, &c.sub(&context.one, &receiver_above)?)?;
    let mixed_regime = c.mul(
        &nominal,
        &c.mul(&receiver_above, &c.sub(&context.one, &receiver_full)?)?,
    )?;
    let submerged_regime = c.mul(&nominal, &receiver_full)?;
    let regime = c.add(
        &c.add(
            &c.mul_scalar(&closed, codes::GATE_CLOSED)?,
            &c.mul_scalar(&below, codes::GATE_BELOW_SILL)?,
        )?,
        &c.add(
            &c.add(
                &c.mul_scalar(&equal, codes::EQUAL_HEADS)?,
                &c.mul_scalar(&free_regime, codes::FREE_GATE)?,
            )?,
            &c.add(
                &c.mul_scalar(&mixed_regime, codes::MIXED_GATE)?,
                &c.mul_scalar(&submerged_regime, codes::SUBMERGED_GATE)?,
            )?,
        )?,
    )?;
    Ok(Classification {
        active,
        free,
        submerged,
        fully_submerged,
        donor_full,
        receiver_above,
        upper_positive,
        donor,
        receiver,
        crown_high,
        crown_low,
        crown_overflow: overflow,
        regime,
        status: c.add(
            &c.mul_scalar(&nominal, codes::ACTIVE)?,
            &c.mul_scalar(&blocked, codes::BLOCKED - codes::ACTIVE)?,
        )?,
        direction: c.sub(&nominal_a, &nominal_b)?,
    })
}
