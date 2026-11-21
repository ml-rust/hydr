//! Donor, direction and zero-flow class of every culvert lane, from the supported heads.
//!
//! A side supports flow when it is wet and stands above its invert. The donor is the side that
//! supports flow and stands higher, or that faces a dry receiver. Equal wet stages with either
//! side above its invert carry no water. A receiver that is dry or below its invert is a free
//! outfall, so its tailwater is zero.

use crate::structures::tensor::culvert::{TensorBarrelContext, TensorCulvertLaw};
use crate::structures::tensor::{TensorStructureContext, TensorStructureInput, codes};
use crate::structures::{CulvertClient, StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Per-lane heads and classes, each `[m]`.
pub(crate) struct Lanes<R: Runtime<DType = DType>> {
    /// 1 where the culvert carries the donor's water, 0 elsewhere.
    pub(crate) active: Tensor<R>,
    /// Nominal direction: 1 for A to B, -1 for B to A, 0 without a donor.
    pub(crate) direction: Tensor<R>,
    /// Status code of lanes outside `active`.
    pub(crate) status: Tensor<R>,
    /// Regime code of lanes outside `active`.
    pub(crate) regime: Tensor<R>,
    /// Oriented row of each lane, I64: `j` from A, `m + j` from B.
    pub(crate) rows: Tensor<R>,
    /// Donor stage above the inlet invert. Inactive lanes hold half the rise.
    pub(crate) headwater: Tensor<R>,
    /// Receiver stage above the outlet invert, zero for a free outfall and on inactive lanes.
    pub(crate) tailwater: Tensor<R>,
    /// Donor stage minus receiver stage on active lanes with a wet receiver, else zero.
    pub(crate) gap: Tensor<R>,
}

/// Classifies every lane and rejects a known dry end whose bed lies above its invert.
///
/// `rise` is the barrel context of the forward rows, which shares each lane's rise.
pub(crate) fn classify<R>(
    law: &TensorCulvertLaw<R>,
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
    rise: &TensorBarrelContext<R>,
) -> Result<Lanes<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = &context.client;
    let take = |value: &Tensor<R>| c.index_select(value, 0, &law.lanes);
    let [stage_a, stage_b] = [take(&input.stage_a)?, take(&input.stage_b)?];
    let [wet_a, wet_b] = [take(&input.wet_a)?, take(&input.wet_b)?];
    let [bed_a, bed_b] = [take(&input.bed_a)?, take(&input.bed_b)?];
    let [unknown_a, unknown_b] = [take(&input.unknown_a)?, take(&input.unknown_b)?];
    let [invert_a, invert_b] = &law.invert;
    let one = &rise.one;
    let not = |mask: &Tensor<R>| c.sub(one, mask);

    let dry_a = c.mul(&not(&wet_a)?, &not(&unknown_a)?)?;
    let dry_b = c.mul(&not(&wet_b)?, &not(&unknown_b)?)?;
    let buried = c.maximum(
        &c.mul(&dry_a, &c.gt(&bed_a, invert_a)?)?,
        &c.mul(&dry_b, &c.gt(&bed_b, invert_b)?)?,
    )?;
    let wide = c.index_put(&context.zero, 0, &law.lanes, &buried)?;
    context.reject_input(
        &wide,
        StructureInputErrorKind::UnsupportedRegime,
        "a known dry culvert end needs its bed at or below its invert",
    )?;

    let above_a = c.mul(&wet_a, &c.gt(&stage_a, invert_a)?)?;
    let above_b = c.mul(&wet_b, &c.gt(&stage_b, invert_b)?)?;
    let a_high = c.gt(&stage_a, &stage_b)?;
    let b_high = c.gt(&stage_b, &stage_a)?;
    let equal = c.eq(&stage_a, &stage_b)?;
    let nominal_a = c.mul(&above_a, &c.add(&not(&wet_b)?, &c.mul(&wet_b, &a_high)?)?)?;
    let nominal_b = c.mul(&above_b, &c.add(&not(&wet_a)?, &c.mul(&wet_a, &b_high)?)?)?;
    let nominal = c.add(&nominal_a, &nominal_b)?;
    let [allow_a, allow_b] = &law.allow;
    let active = c.add(&c.mul(&nominal_a, allow_a)?, &c.mul(&nominal_b, allow_b)?)?;
    let blocked = c.sub(&nominal, &active)?;
    let level = c.mul(
        &c.mul(&c.mul(&wet_a, &wet_b)?, &equal)?,
        &c.maximum(&above_a, &above_b)?,
    )?;
    let code = |value: f64| rise.constant(value);
    let regime = c.where_cond(
        &level,
        &code(codes::EQUAL_HEADS)?,
        &c.where_cond(
            &blocked,
            &code(codes::CULVERT_UNDECLARED_END)?,
            &code(codes::CULVERT_BELOW_INVERT)?,
        )?,
    )?;
    let status = c.where_cond(
        &blocked,
        &code(codes::BLOCKED)?,
        &code(codes::PHYSICAL_ZERO)?,
    )?;

    let offset = c.mul_scalar(&nominal_b, law.width as f64)?;
    let rows = c.cast(&c.add(&law.forward, &offset)?, DType::I64)?;
    let donor = c.where_cond(&nominal_b, &stage_b, &stage_a)?;
    let receiver = c.where_cond(&nominal_b, &stage_a, &stage_b)?;
    let receiver_wet = c.where_cond(&nominal_b, &wet_a, &wet_b)?;
    let inlet_invert = c.where_cond(&nominal_b, invert_b, invert_a)?;
    let outlet_invert = c.where_cond(&nominal_b, invert_a, invert_b)?;
    let headwater = c.where_cond(&active, &c.sub(&donor, &inlet_invert)?, &rise.half_rise)?;
    let submerged = c.mul(&active, &receiver_wet)?;
    let tailwater = c.where_cond(
        &submerged,
        &c.maximum(&c.sub(&receiver, &outlet_invert)?, &rise.zero)?,
        &rise.zero,
    )?;
    let gap = c.where_cond(&submerged, &c.sub(&donor, &receiver)?, &rise.zero)?;
    Ok(Lanes {
        direction: c.sub(&nominal_a, &nominal_b)?,
        active,
        status,
        regime,
        rows,
        headwater,
        tailwater,
        gap,
    })
}
