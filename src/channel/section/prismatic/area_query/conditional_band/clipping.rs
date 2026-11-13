use super::types::LocalEndpoint;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct ClippedGap<R: Runtime<DType = DType>> {
    pub retained: CertifiedExpansion<R>,
    pub physical: Interval<R>,
    pub active: Tensor<R>,
    pub zero: Tensor<R>,
}

pub(super) fn clip<R>(
    a: &mut Arithmetic<'_, R>,
    above: &Interval<R>,
    below: &Interval<R>,
    extent: &Interval<R>,
    start_identity: &Tensor<R>,
    end_identity: &Tensor<R>,
    zero: &Tensor<R>,
) -> HydraulicResult<LocalEndpoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let at_start = c.maximum(start_identity, &c.le(above.upper(), zero)?)?;
    let at_end = c.maximum(end_identity, &c.le(below.upper(), zero)?)?;
    let distance = |a: &mut Arithmetic<'_, R>,
                    value: &Interval<R>,
                    here: &Tensor<R>,
                    opposite: &Tensor<R>|
     -> HydraulicResult<Interval<R>> {
        let low = c.maximum(&c.minimum(value.lower(), extent.lower())?, zero)?;
        let high = c.maximum(&c.minimum(value.upper(), extent.upper())?, zero)?;
        Ok(a.bounds(
            &c.where_cond(here, zero, &c.where_cond(opposite, extent.lower(), &low)?)?,
            &c.where_cond(here, zero, &c.where_cond(opposite, extent.upper(), &high)?)?,
        )?)
    };
    Ok(LocalEndpoint {
        above_start: distance(a, above, &at_start, &at_end)?,
        below_end: distance(a, below, &at_end, &at_start)?,
        at_start,
        at_end,
    })
}

pub(super) fn gap<R>(
    a: &mut Arithmetic<'_, R>,
    valid: &Tensor<R>,
    start: &LocalEndpoint<R>,
    end: &LocalEndpoint<R>,
    delta: &CertifiedExpansion<R>,
    unclipped: &Tensor<R>,
    zero: &Tensor<R>,
) -> HydraulicResult<ClippedGap<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let above_gap = a.sub(&end.above_start, &start.above_start)?;
    let below_gap = a.sub(&start.below_end, &end.below_end)?;
    let complete = a.expansion_interval(delta)?;
    // Clipping is monotone and one-Lipschitz, preserving the complete gap's physical sign and scale.
    let lower = c.maximum(
        &c.maximum(above_gap.lower(), below_gap.lower())?,
        &c.minimum(complete.lower(), zero)?,
    )?;
    let upper = c.minimum(
        &c.minimum(above_gap.upper(), below_gap.upper())?,
        &c.maximum(complete.upper(), zero)?,
    )?;
    let lower = c.where_cond(unclipped, &c.maximum(&lower, complete.lower())?, &lower)?;
    let upper = c.where_cond(unclipped, &c.minimum(&upper, complete.upper())?, &upper)?;
    let complete_zero = c.mul(
        &c.eq(delta.high(), zero)?,
        &c.mul(&c.eq(delta.low(), zero)?, &c.eq(delta.abs_error(), zero)?)?,
    )?;
    let shared_boundary = c.maximum(
        &c.mul(&start.at_start, &end.at_start)?,
        &c.mul(&start.at_end, &end.at_end)?,
    )?;
    let zero_gap = c.maximum(
        &c.eq(valid, zero)?,
        &c.maximum(&shared_boundary, &complete_zero)?,
    )?;
    let active = c.mul(valid, &c.eq(&zero_gap, zero)?)?;
    let physical = a.bounds(
        &c.where_cond(&active, &lower, zero)?,
        &c.where_cond(&active, &upper, zero)?,
    )?;
    let enclosed = a.expansion_from_interval(&physical)?;
    let retained = |value: &Tensor<R>, full: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.where_cond(&active, &c.where_cond(unclipped, full, value)?, zero)?)
    };
    let retained = a.expansion_bounds(
        &retained(enclosed.high(), delta.high())?,
        &retained(enclosed.low(), delta.low())?,
        &retained(enclosed.abs_error(), delta.abs_error())?,
    )?;
    Ok(ClippedGap {
        retained,
        physical,
        active,
        zero: zero_gap,
    })
}
