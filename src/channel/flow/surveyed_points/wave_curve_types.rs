use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Unavailable rows contain placeholders, never hydraulic certificates.
pub(super) struct CurveValue<R: Runtime<DType = DType>> {
    pub(super) retained: CertifiedExpansion<R>,
    pub(super) physical: Interval<R>,
    pub(super) available: Tensor<R>,
}

pub(super) struct CurveClassification<R: Runtime<DType = DType>> {
    pub(super) shock: Tensor<R>,
    pub(super) rarefaction: Tensor<R>,
    pub(super) equality: Tensor<R>,
    pub(super) uncertain: Tensor<R>,
    pub(super) dry_source: Tensor<R>,
}

/// Restricted physical operands belong to this shock certificate, not an unrestricted wave-speed claim.
pub(super) struct ShockDomain<R: Runtime<DType = DType>> {
    pub(super) source_area: Interval<R>,
    pub(super) target_area: Interval<R>,
    pub(super) retained_separation: CertifiedExpansion<R>,
    pub(super) separation: Interval<R>,
    pub(super) contraction: Interval<R>,
}

pub(super) struct ShockOffset<R: Runtime<DType = DType>> {
    pub(super) value: CurveValue<R>,
    pub(super) domain: ShockDomain<R>,
}

pub(super) struct EndpointCurve<R: Runtime<DType = DType>> {
    pub(super) curve: CurveValue<R>,
    pub(super) classification: CurveClassification<R>,
    pub(super) actual_shock: ShockOffset<R>,
    pub(super) conditional_shock: ShockOffset<R>,
    pub(super) equality_limit: CurveValue<R>,
}

pub(super) struct CurveRange<R: Runtime<DType = DType>> {
    pub(super) lower: EndpointCurve<R>,
    pub(super) upper: EndpointCurve<R>,
    pub(super) curve: CurveValue<R>,
}

pub(super) fn unavailable<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(&[count], a.dtype(), a.client().device())?;
    Ok(CurveValue {
        retained: a.expansion_point(&zero)?,
        physical: a.point(&zero)?,
        available: zero,
    })
}

pub(super) fn empty_offset<R>(
    a: &mut Arithmetic<'_, R>,
    count: usize,
) -> HydraulicResult<ShockOffset<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(&[count], a.dtype(), a.client().device())?;
    Ok(ShockOffset {
        value: unavailable(a, count)?,
        domain: ShockDomain {
            source_area: a.point(&zero)?,
            target_area: a.point(&zero)?,
            retained_separation: a.expansion_point(&zero)?,
            separation: a.point(&zero)?,
            contraction: a.point(&zero)?,
        },
    })
}

pub(super) fn scatter_value<R>(
    a: &mut Arithmetic<'_, R>,
    dense: CurveValue<R>,
    rows: &Tensor<R>,
    value: &CurveValue<R>,
) -> HydraulicResult<CurveValue<R>>
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
    Ok(CurveValue {
        retained: scatter_expansion(a, &dense.retained, rows, &value.retained)?,
        physical: scatter_interval(a, &dense.physical, rows, &value.physical)?,
        available: c.index_put(&dense.available, 0, rows, &value.available)?,
    })
}

pub(super) fn scatter_offset<R>(
    a: &mut Arithmetic<'_, R>,
    dense: ShockOffset<R>,
    rows: &Tensor<R>,
    value: &ShockOffset<R>,
) -> HydraulicResult<ShockOffset<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(ShockOffset {
        value: scatter_value(a, dense.value, rows, &value.value)?,
        domain: ShockDomain {
            source_area: scatter_interval(
                a,
                &dense.domain.source_area,
                rows,
                &value.domain.source_area,
            )?,
            target_area: scatter_interval(
                a,
                &dense.domain.target_area,
                rows,
                &value.domain.target_area,
            )?,
            retained_separation: scatter_expansion(
                a,
                &dense.domain.retained_separation,
                rows,
                &value.domain.retained_separation,
            )?,
            separation: scatter_interval(
                a,
                &dense.domain.separation,
                rows,
                &value.domain.separation,
            )?,
            contraction: scatter_interval(
                a,
                &dense.domain.contraction,
                rows,
                &value.domain.contraction,
            )?,
        },
    })
}

fn scatter_interval<R>(
    a: &mut Arithmetic<'_, R>,
    dense: &Interval<R>,
    rows: &Tensor<R>,
    value: &Interval<R>,
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
    let c = a.client();
    Ok(a.bounds(
        &c.index_put(dense.lower(), 0, rows, value.lower())?,
        &c.index_put(dense.upper(), 0, rows, value.upper())?,
    )?)
}

fn scatter_expansion<R>(
    a: &mut Arithmetic<'_, R>,
    dense: &CertifiedExpansion<R>,
    rows: &Tensor<R>,
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
    let c = a.client();
    Ok(a.expansion_bounds(
        &c.index_put(dense.high(), 0, rows, value.high())?,
        &c.index_put(dense.low(), 0, rows, value.low())?,
        &c.index_put(dense.abs_error(), 0, rows, value.abs_error())?,
    )?)
}
