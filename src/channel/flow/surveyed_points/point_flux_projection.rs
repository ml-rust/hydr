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
    let fallback = a.expansion_from_interval(physical)?;
    let fallback = center(a, &fallback)?;
    let chosen = a.expansion_bounds(
        &c.where_cond(&contained, proposal.high(), fallback.high())?,
        &c.where_cond(&contained, proposal.low(), fallback.low())?,
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
