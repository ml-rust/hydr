use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Area, discharge, and chosen pressure are exact numerical points. Evaluation uncertainty remains outside this ledger.
pub(super) fn momentum<R>(
    a: &mut Arithmetic<'_, R>,
    area: &CertifiedExpansion<R>,
    discharge: &CertifiedExpansion<R>,
    pressure: &CertifiedExpansion<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (velocity, advective) = advection(a, area, discharge)?;
    let momentum = a.expansion_add(&advective, pressure)?;
    Ok((velocity, momentum))
}

/// Exact numerical operands retain every quotient and product arithmetic radius.
pub(super) fn advection<R>(
    a: &mut Arithmetic<'_, R>,
    area: &CertifiedExpansion<R>,
    discharge: &CertifiedExpansion<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let velocity = a.expansion_div_positive(discharge, area)?;
    let advective = a.expansion_mul(discharge, &velocity)?;
    Ok((velocity, advective))
}

pub(super) fn intersect_advection<R>(
    a: &mut Arithmetic<'_, R>,
    ordinary: &Interval<R>,
    retained: &CertifiedExpansion<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let retained = a.expansion_interval(retained)?;
    let (intersection, valid) = super::wet_star::intersection(a, ordinary, &retained)?;
    let zero = numr::tensor::Tensor::zeros(valid.shape(), a.dtype(), a.client().device())?;
    if a.client().nonzero(&a.client().eq(&valid, &zero)?)?.numel() != 0 {
        return Err(super::point_root_session::invalid(
            "ordinary and retained advection certificates are disjoint. Discard the inconsistent point",
        ));
    }
    Ok(intersection)
}
