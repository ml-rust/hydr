use super::{
    paired_sources::PairedSources, point_candidate_types::ExteriorPoint,
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn exterior<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    owners: &Tensor<R>,
    parent: &Tensor<R>,
    side: SonicSide,
) -> HydraulicResult<ExteriorPoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let selected = sources.select_sources(a, parent)?;
    let query = super::potential_range::source_query(
        sources.geometry,
        &selected.bands,
        &selected.sources[side.index()],
        &selected.original_rows,
    )?;
    let input = selected.velocity.input();
    let area = if side == SonicSide::Left {
        &input.left_area
    } else {
        &input.right_area
    };
    let discharge = if side == SonicSide::Left {
        &input.left_discharge
    } else {
        &input.right_discharge
    };
    let velocity = if side == SonicSide::Left {
        &selected.velocity.left
    } else {
        &selected.velocity.right
    };
    let area = a.expansion_bounds(area.high(), area.low(), area.abs_error())?;
    let discharge = a.expansion_bounds(discharge.high(), discharge.low(), discharge.abs_error())?;
    let velocity = a.expansion_bounds(velocity.high(), velocity.low(), velocity.abs_error())?;
    let [left, right] = selected.sources;
    let recovery = if side == SonicSide::Left { left } else { right };
    Ok(ExteriorPoint {
        invocation_owner: copy(a, owners)?,
        parent_rows: copy(a, parent)?,
        side,
        query,
        area,
        discharge,
        velocity,
        recovery,
    })
}
fn copy<R>(a: &Arithmetic<'_, R>, value: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), a.client().device())?;
    a.client().copy_into(&owned, value)?;
    Ok(owned)
}
