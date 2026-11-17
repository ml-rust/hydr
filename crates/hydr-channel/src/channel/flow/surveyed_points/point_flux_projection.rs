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

pub(super) struct Projection<R: Runtime<DType = DType>> {
    pub(super) proposal: CertifiedExpansion<R>,
    pub(super) contained: Tensor<R>,
    pub(super) chosen: CertifiedExpansion<R>,
    pub(super) change: CertifiedExpansion<R>,
}

pub(super) fn center<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(value.high().shape(), a.dtype(), a.client().device())?;
    Ok(a.expansion_bounds(value.high(), value.low(), &zero)?)
}

/// Selects the proposal at the center of `raw`, with its ball dropped.
///
/// The selected point is the proposal, so its error scales with the face jump.
/// `contained` marks proposals inside `physical`. Captured identities count as contained.
/// A proposal whose sign `physical` excludes moves to the nearest physical endpoint.
/// That keeps the certified water-flux sign and an exact zero endpoint, such as wall water.
/// The joint bound hulls the selected point with `physical`.
/// The flux acceptance mask then rejects a proposal that lies outside by more than the allowance.
pub(super) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    raw: &CertifiedExpansion<R>,
    physical: &Interval<R>,
    captured_identity: Option<&Tensor<R>>,
) -> HydraulicResult<Projection<R>>
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
    let proposal = center(a, raw)?;
    let lower = a.expansion_point(physical.lower())?;
    let upper = a.expansion_point(physical.upper())?;
    // Retained subtraction tests both limbs before projecting their signs.
    let above = a.expansion_sub(&proposal, &lower)?;
    let below = a.expansion_sub(&upper, &proposal)?;
    let above = a.expansion_interval(&above)?;
    let below = a.expansion_interval(&below)?;
    let c = a.client();
    let zero = Tensor::zeros(proposal.high().shape(), a.dtype(), c.device())?;
    let mut contained = c.mul(&c.ge(above.lower(), &zero)?, &c.ge(below.lower(), &zero)?)?;
    if let Some(identity) = captured_identity {
        contained = c.maximum(&contained, identity)?;
    }
    let repair = sign_repair(a, &proposal, physical, &contained)?;
    // Physical lower is nonnegative unless physical upper is nonpositive. Adding +0 clears -0.
    let target = c.where_cond(
        &c.le(physical.upper(), &zero)?,
        physical.upper(),
        physical.lower(),
    )?;
    let target = c.add(&target, &zero)?;
    let chosen = a.expansion_bounds(
        &c.where_cond(&repair, &target, proposal.high())?,
        &c.where_cond(&repair, &zero, proposal.low())?,
        &zero,
    )?;
    let change = a.expansion_sub(&chosen, &proposal)?;
    Ok(Projection {
        proposal,
        contained,
        chosen,
        change,
    })
}

/// Zero-one mask of uncontained proposals whose sign `physical` excludes.
///
/// `physical` admits a negative value when its lower end is negative, zero when it holds 0,
/// and a positive value when its upper end is positive.
fn sign_repair<R>(
    a: &mut Arithmetic<'_, R>,
    proposal: &CertifiedExpansion<R>,
    physical: &Interval<R>,
    contained: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(proposal.high().shape(), a.dtype(), c.device())?;
    let (high, low) = (proposal.high(), proposal.low());
    // A zero high limb leaves the sign to the low limb.
    let flat = c.eq(high, &zero)?;
    let negative = c.maximum(&c.lt(high, &zero)?, &c.mul(&flat, &c.lt(low, &zero)?)?)?;
    let positive = c.maximum(&c.gt(high, &zero)?, &c.mul(&flat, &c.gt(low, &zero)?)?)?;
    let exact_zero = c.mul(&flat, &c.eq(low, &zero)?)?;
    let (lower, upper) = (physical.lower(), physical.upper());
    let holds_zero = c.mul(&c.le(lower, &zero)?, &c.ge(upper, &zero)?)?;
    let excluded = |sign: &Tensor<R>, admitted: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(c.mul(sign, &c.eq(admitted, &zero)?)?)
    };
    let repair = c.maximum(
        &excluded(&negative, &c.lt(lower, &zero)?)?,
        &c.maximum(
            &excluded(&positive, &c.gt(upper, &zero)?)?,
            &excluded(&exact_zero, &holds_zero)?,
        )?,
    )?;
    Ok(c.mul(&repair, &c.eq(contained, &zero)?)?)
}
