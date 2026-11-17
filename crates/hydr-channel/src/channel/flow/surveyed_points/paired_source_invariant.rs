use super::paired_sources::{PairedSources, SourceInvariant};
use super::potential_range::select_expansion;
use super::wave_curve_types::{CurveValue, scatter_value, unavailable};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

#[cfg(test)]
pub(super) fn evaluate<R>(parent: &mut PairedSources<'_, R>) -> HydraulicResult<SourceInvariant<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = parent.geometry.client();
    let n = parent.original_rows.numel();
    let rows = c.nonzero(&parent.velocity.both_wet)?.flatten()?;
    let source = parent.source_values([&rows, &rows])?;
    let value = checked(c, "paired source invariant cancellation", |a| {
        let du = select_expansion(a, &parent.velocity.difference, &rows)?;
        let retained = a.expansion_sub(&du, &source[0].retained)?;
        let retained = a.expansion_sub(&retained, &source[1].retained)?;
        let physical = a.expansion_interval(&retained)?;
        let value = CurveValue {
            retained,
            physical,
            available: Tensor::ones(rows.shape(), a.dtype(), c.device())?,
        };
        let empty = unavailable(a, n)?;
        scatter_value(a, empty, &rows, &value)
    })?;
    checked::<R, _>(c, "paired source invariant classification", |a| {
        let input = parent.velocity.input();
        classify(a, n, value, &input.left_dry, &input.right_dry)
    })
}

pub(super) fn evaluate_on_rows<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
) -> HydraulicResult<SourceInvariant<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = parent.geometry.client();
    let selected = checked::<R, _>(c, "paired source invariant row admission", |a| {
        parent.select_sources(a, rows)
    })?;
    let n = selected.rows.numel();
    let owner = c.nonzero(&selected.velocity.both_wet)?.flatten()?;
    let rows = c.index_select(&selected.rows, 0, &owner)?;
    let source = parent.source_values([&rows, &rows])?;
    let value = checked::<R, _>(c, "paired selected source invariant cancellation", |a| {
        let du = select_expansion(a, &selected.velocity.difference, &owner)?;
        let retained = a.expansion_sub(&du, &source[0].retained)?;
        let retained = a.expansion_sub(&retained, &source[1].retained)?;
        let physical = a.expansion_interval(&retained)?;
        let value = CurveValue {
            retained,
            physical,
            available: Tensor::ones(owner.shape(), a.dtype(), c.device())?,
        };
        let empty = unavailable(a, n)?;
        scatter_value(a, empty, &owner, &value)
    })?;
    checked::<R, _>(c, "paired selected source invariant classification", |a| {
        let input = selected.velocity.input();
        classify(a, n, value, &input.left_dry, &input.right_dry)
    })
}

fn classify<R>(
    a: &mut crate::channel::flow::surveyed::error::Arithmetic<'_, R>,
    n: usize,
    value: CurveValue<R>,
    left_dry: &Tensor<R>,
    right_dry: &Tensor<R>,
) -> HydraulicResult<SourceInvariant<R>>
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
    let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
    let vacuum = c.mul(&value.available, &c.gt(value.physical.lower(), &zero)?)?;
    let wet_candidate = c.mul(&value.available, &c.lt(value.physical.upper(), &zero)?)?;
    let touching = c.mul(
        &value.available,
        &c.mul(
            &c.eq(value.retained.high(), &zero)?,
            &c.mul(
                &c.eq(value.retained.low(), &zero)?,
                &c.eq(value.retained.abs_error(), &zero)?,
            )?,
        )?,
    )?;
    #[cfg(test)]
    let known = c.maximum(&touching, &c.maximum(&vacuum, &wet_candidate)?)?;
    #[cfg(test)]
    let unresolved = c.mul(&value.available, &c.eq(&known, &zero)?)?;
    let both_dry = c.mul(left_dry, right_dry)?;
    Ok(SourceInvariant {
        value,
        vacuum,
        touching,
        wet_candidate,
        #[cfg(test)]
        unresolved,
        both_dry,
    })
}
