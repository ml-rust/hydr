//! Original-opening strip geometry with retained physical-expression TwoSum witnesses.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::TwoSumOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use super::{classification::Classification, geometry_bounds as bounds};
use crate::structures::StructureError;
use crate::tensor::structures::{
    TensorStructureContext,
    gate::{TensorGateContext, TensorGateGeometry, TensorGateInput},
};

pub(super) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    gate: &TensorGateContext<R>,
    operation: &TensorGateInput<R>,
    class: &Classification<R>,
) -> Result<TensorGateGeometry<R>, StructureError>
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
    let partial_submerged = c.sub(&class.submerged, &class.fully_submerged)?;
    let (depth, depth_low) = c.two_sum(
        &c.where_cond(&partial_submerged, &class.receiver, &context.zero)?,
        &c.neg(&c.where_cond(&partial_submerged, &gate.sill, &context.zero)?)?,
    )?;
    context.finite(&depth, "active gate submerged height overflows")?;
    context.finite(
        &depth_low,
        "active gate submerged-height witness is nonfinite",
    )?;
    let submerged_height = c.where_cond(&class.fully_submerged, &operation.opening, &depth)?;
    let submerged_height_bound = c.abs(&depth_low)?;
    let (gap, gap_low) = c.two_sum(
        &c.where_cond(&class.submerged, &class.donor, &context.zero)?,
        &c.neg(&c.where_cond(&class.submerged, &class.receiver, &context.zero)?)?,
    )?;
    context.finite(&gap, "active gate original-stage pressure gap overflows")?;
    context.finite(&gap_low, "active gate pressure-gap witness is nonfinite")?;
    let gap_bound = c.abs(&gap_low)?;

    let full_free = c.mul(&class.free, &class.donor_full)?;
    let complement = c.mul(&full_free, &class.receiver_above)?;
    let (complement_high, complement_low) = c.two_sum(
        &c.where_cond(&complement, &operation.opening, &context.zero)?,
        &c.neg(&c.where_cond(&complement, &depth, &context.zero)?)?,
    )?;
    context.finite(
        &complement_high,
        "gate complementary-height subtraction overflows",
    )?;
    context.finite(
        &complement_low,
        "gate complementary-height first witness is nonfinite",
    )?;
    let (complement_height, complement_tail) = c.two_sum(
        &complement_high,
        &c.neg(&c.where_cond(&complement, &depth_low, &context.zero)?)?,
    )?;
    context.finite(
        &complement_height,
        "gate complementary free height overflows",
    )?;
    context.finite(
        &complement_tail,
        "gate complementary-height final witness is nonfinite",
    )?;
    let complement_bound =
        bounds::add(context, &c.abs(&complement_low)?, &c.abs(&complement_tail)?)?;

    let partial_free = c.sub(&class.free, &full_free)?;
    let lower_plane = c.where_cond(&class.receiver_above, &class.receiver, &gate.sill)?;
    let (partial_height, partial_low) = c.two_sum(
        &c.where_cond(&partial_free, &class.donor, &context.zero)?,
        &c.neg(&c.where_cond(&partial_free, &lower_plane, &context.zero)?)?,
    )?;
    context.finite(&partial_height, "gate partly wetted free height overflows")?;
    context.finite(
        &partial_low,
        "gate partly wetted free-height witness is nonfinite",
    )?;
    let free_height = c.where_cond(
        &complement,
        &complement_height,
        &c.where_cond(&full_free, &operation.opening, &partial_height)?,
    )?;
    let free_height_bound = c.where_cond(&complement, &complement_bound, &c.abs(&partial_low)?)?;

    let (pressure_high, pressure_low) = c.two_sum(
        &c.where_cond(&class.upper_positive, &class.donor, &context.zero)?,
        &c.neg(&c.where_cond(&class.upper_positive, &class.crown_high, &context.zero)?)?,
    )?;
    context.finite(&pressure_high, "gate crown-pressure subtraction overflows")?;
    context.finite(
        &pressure_low,
        "gate crown-pressure first witness is nonfinite",
    )?;
    let (upper, pressure_tail) = c.two_sum(
        &pressure_high,
        &c.neg(&c.where_cond(&class.upper_positive, &class.crown_low, &context.zero)?)?,
    )?;
    context.finite(&upper, "active upper gate free pressure head overflows")?;
    context.finite(
        &pressure_tail,
        "gate crown-pressure final witness is nonfinite",
    )?;
    let upper_bound = bounds::add(context, &c.abs(&pressure_low)?, &c.abs(&pressure_tail)?)?;
    let (lower, lower_low) = c.two_sum(&free_height, &upper)?;
    context.finite(&lower, "active lower gate free pressure head overflows")?;
    context.finite(
        &lower_low,
        "gate lower-pressure reconstruction witness is nonfinite",
    )?;
    let lower_bound = bounds::add(
        context,
        &bounds::add(context, &free_height_bound, &upper_bound)?,
        &c.abs(&lower_low)?,
    )?;
    bounds::resolved_positive(
        context,
        &class.submerged,
        &submerged_height,
        &submerged_height_bound,
    )?;
    bounds::resolved_positive(context, &class.submerged, &gap, &gap_bound)?;
    bounds::resolved_positive(context, &class.free, &free_height, &free_height_bound)?;
    bounds::resolved_positive(context, &class.free, &lower, &lower_bound)?;
    bounds::resolved_positive(context, &class.upper_positive, &upper, &upper_bound)?;

    let partial_donor = c.mul(&class.active, &c.sub(&context.one, &class.donor_full)?)?;
    let (target_partial, target_low) = c.two_sum(
        &c.where_cond(&partial_donor, &class.donor, &context.zero)?,
        &c.neg(&c.where_cond(&partial_donor, &gate.sill, &context.zero)?)?,
    )?;
    context.finite(
        &target_partial,
        "gate wetted-height closure target overflows",
    )?;
    context.finite(
        &target_low,
        "gate wetted-height closure target witness is nonfinite",
    )?;
    let target = c.where_cond(
        &c.mul(&class.active, &class.donor_full)?,
        &operation.opening,
        &target_partial,
    )?;
    let (sum, sum_low) = c.two_sum(&submerged_height, &free_height)?;
    context.finite(&sum, "gate represented strip sum overflows")?;
    context.finite(&sum_low, "gate represented strip-sum witness is nonfinite")?;
    let (closure_residual, residual_low) = c.two_sum(&sum, &c.neg(&target)?)?;
    context.finite(
        &closure_residual,
        "gate strip-closure discrepancy overflows",
    )?;
    context.finite(
        &residual_low,
        "gate strip-closure discrepancy witness is nonfinite",
    )?;
    let closure_uncertainty = bounds::add(context, &c.abs(&sum_low)?, &c.abs(&residual_low)?)?;
    let closure_allowance = bounds::add(
        context,
        &submerged_height_bound,
        &bounds::add(context, &free_height_bound, &c.abs(&target_low)?)?,
    )?;
    let closure_envelope = bounds::add(context, &closure_allowance, &closure_uncertainty)?;
    context.reject(
        &c.gt(&c.abs(&closure_residual)?, &closure_envelope)?,
        "gate strip-closure interval is disjoint from independent physical-length allowance",
    )?;
    Ok(TensorGateGeometry {
        opening: c.where_cond(&class.active, &operation.opening, &context.zero)?,
        crown_high: c.where_cond(&class.active, &class.crown_high, &context.zero)?,
        crown_low: c.where_cond(&class.active, &class.crown_low, &context.zero)?,
        crown_overflow: c.mul(&class.active, &class.crown_overflow)?,
        submerged_height,
        submerged_height_bound,
        free_height,
        free_height_bound,
        gap,
        gap_bound,
        lower,
        lower_bound,
        upper,
        upper_bound,
        closure_residual,
        closure_uncertainty,
        closure_allowance,
    })
}
