use super::paired_sources::{PairedSources, SourceInvariant};
use super::potential_range::select_expansion;
use super::wave_curve_types::{CurveValue, scatter_value, unavailable};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

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
        let known = c.maximum(&touching, &c.maximum(&vacuum, &wet_candidate)?)?;
        let unresolved = c.mul(&value.available, &c.eq(&known, &zero)?)?;
        let input = parent.velocity.input();
        let both_dry = c.mul(&input.left_dry, &input.right_dry)?;
        Ok(SourceInvariant {
            value,
            vacuum,
            touching,
            wet_candidate,
            unresolved,
            both_dry,
        })
    })
}
