use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use crate::channel::flow::surveyed_points::analytic_bottom::AnalyticSource;
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Independent leaf results retain arithmetic uncertainty separately from physical restrictions.
pub(in crate::channel::flow::surveyed_points) struct PotentialValue<R: Runtime<DType = DType>> {
    pub(in crate::channel::flow::surveyed_points) retained: CertifiedExpansion<R>,
    pub(in crate::channel::flow::surveyed_points) physical: Interval<R>,
    /// Only eligible rows carry analytic certificates. Other rows remain unavailable.
    pub(in crate::channel::flow::surveyed_points) analytic: Option<AnalyticSource<R>>,
}

pub(in crate::channel::flow::surveyed_points) fn zero_value<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
) -> HydraulicResult<PotentialValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(&[count], DType::F64, a.client().device())?;
    Ok(PotentialValue {
        retained: a.expansion_point(&zero)?,
        physical: a.point(&zero)?,
        analytic: None,
    })
}

pub(in crate::channel::flow::surveyed_points) fn select_expansion<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    rows: &Tensor<R>,
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
    let c = a.client();
    Ok(a.expansion_bounds(
        &c.index_select(value.high(), 0, rows)?,
        &c.index_select(value.low(), 0, rows)?,
        &c.index_select(value.abs_error(), 0, rows)?,
    )?)
}

pub(in crate::channel::flow::surveyed_points) fn scatter<R>(
    a: &mut Arithmetic<'_, R>,
    dense: PotentialValue<R>,
    rows: &Tensor<R>,
    value: &PotentialValue<R>,
) -> HydraulicResult<PotentialValue<R>>
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
    Ok(PotentialValue {
        retained: a.expansion_bounds(
            &c.index_put(dense.retained.high(), 0, rows, value.retained.high())?,
            &c.index_put(dense.retained.low(), 0, rows, value.retained.low())?,
            &c.index_put(
                dense.retained.abs_error(),
                0,
                rows,
                value.retained.abs_error(),
            )?,
        )?,
        physical: a.bounds(
            &c.index_put(dense.physical.lower(), 0, rows, value.physical.lower())?,
            &c.index_put(dense.physical.upper(), 0, rows, value.physical.upper())?,
        )?,
        analytic: dense.analytic,
    })
}

pub(in crate::channel::flow::surveyed_points) fn from_interval<R>(
    a: &mut Arithmetic<'_, R>,
    physical: Interval<R>,
) -> HydraulicResult<PotentialValue<R>>
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
    Ok(PotentialValue {
        retained: a.expansion_from_interval(&physical)?,
        physical,
        analytic: None,
    })
}
