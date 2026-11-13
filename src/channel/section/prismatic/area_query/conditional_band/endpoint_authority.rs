use super::types::LocalEndpoint;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) struct StoredEndpointArea<R: Runtime<DType = DType>> {
    pub value: CertifiedExpansion<R>,
    pub available: Tensor<R>,
}

pub(super) fn retain<R>(
    a: &mut Arithmetic<'_, R>,
    input: &CertifiedExpansion<R>,
    boundaries: (&CertifiedExpansion<R>, &CertifiedExpansion<R>),
    clip: &LocalEndpoint<R>,
    inside: &Tensor<R>,
    dry: &Tensor<R>,
    valid: &Tensor<R>,
) -> HydraulicResult<StoredEndpointArea<R>>
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
    let zero = Tensor::zeros(input.high().shape(), DType::F64, c.device())?;
    let choose =
        |value: &Tensor<R>, start: &Tensor<R>, end: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            Ok(c.where_cond(
                valid,
                &c.where_cond(
                    dry,
                    &zero,
                    &c.where_cond(
                        &clip.at_start,
                        start,
                        &c.where_cond(&clip.at_end, end, value)?,
                    )?,
                )?,
                &zero,
            )?)
        };
    let value = a.expansion_bounds(
        &choose(input.high(), boundaries.0.high(), boundaries.1.high())?,
        &choose(input.low(), boundaries.0.low(), boundaries.1.low())?,
        &choose(
            input.abs_error(),
            boundaries.0.abs_error(),
            boundaries.1.abs_error(),
        )?,
    )?;
    let available = c.mul(
        valid,
        &c.maximum(
            inside,
            &c.maximum(dry, &c.maximum(&clip.at_start, &clip.at_end)?)?,
        )?,
    )?;
    let m = input.high().numel();
    Ok(StoredEndpointArea {
        value: a.expansion_bounds(
            &value.high().reshape(&[m])?,
            &value.low().reshape(&[m])?,
            &value.abs_error().reshape(&[m])?,
        )?,
        available: available.reshape(&[m])?,
    })
}

pub(super) fn select<R>(
    a: &mut Arithmetic<'_, R>,
    value: &StoredEndpointArea<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<StoredEndpointArea<R>>
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
    Ok(StoredEndpointArea {
        value: a.expansion_bounds(
            &c.index_select(value.value.high(), 0, rows)?,
            &c.index_select(value.value.low(), 0, rows)?,
            &c.index_select(value.value.abs_error(), 0, rows)?,
        )?,
        available: c.index_select(&value.available, 0, rows)?,
    })
}
