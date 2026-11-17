//! Right-sided accepted-state secant area through existing Numr lookup primitives.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

use crate::storage::LevelPoolError;
use crate::storage::tensor::TensorPoolContext;

pub(super) fn select<R>(
    context: &TensorPoolContext<R>,
    volume: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), LevelPoolError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    let c = &context.client;
    let mut areas = Vec::with_capacity(context.groups.len());
    let mut segments = Vec::with_capacity(context.groups.len());
    for group in &context.groups {
        let query = c.index_select(volume, 0, &group.indices)?;
        let insertion = c.searchsorted(&group.volumes, &query, true)?;
        let zero = c.sub(&group.index_one, &group.index_one)?;
        let segment = c.maximum(
            &zero,
            &c.minimum(&c.sub(&insertion, &group.index_one)?, &group.index_last)?,
        )?;
        areas.push(c.index_select(&group.areas, 0, &segment)?);
        segments.push(segment);
    }
    let area = c.index_select(
        &c.cat(&areas.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    let segment = c.index_select(
        &c.cat(&segments.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    Ok((area, segment))
}
