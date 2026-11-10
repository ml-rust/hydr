//! Grouped Solvr Linear interpolation and independent inverse-volume admission.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::storage::LevelPoolError;
use crate::tensor::level_pool::TensorPoolContext;
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};

pub(crate) struct PoolStage<R: Runtime<DType = DType>> {
    pub stage: Tensor<R>,
    pub reconstructed: Tensor<R>,
    pub residual: Tensor<R>,
}

pub(crate) fn evaluate<R>(
    context: &TensorPoolContext<R>,
    volume: &Tensor<R>,
) -> Result<PoolStage<R>, LevelPoolError>
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
    context.range(volume, "storage curve query outside closed volume support")?;
    let c = &context.client;
    let mut stages = Vec::with_capacity(context.groups.len());
    let mut inverses = Vec::with_capacity(context.groups.len());
    for group in &context.groups {
        let query = c.index_select(volume, 0, &group.indices)?;
        let height = c.interp1d(&group.volumes, &group.stages, &query, InterpMethod::Linear)?;
        let inverse = c.interp1d(&group.stages, &group.volumes, &height, InterpMethod::Linear)?;
        stages.push(height);
        inverses.push(inverse);
    }
    let stage = c.index_select(
        &c.cat(&stages.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    let reconstructed = c.index_select(
        &c.cat(&inverses.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    context.finite(&stage, "reported pool stage")?;
    context.finite(&reconstructed, "inverse reconstructed volume")?;
    let wet = c.gt(volume, &context.zero)?;
    context.reject(
        &c.mul(&wet, &c.le(&stage, &context.origin)?)?,
        "positive storage cannot resolve wet stage above its origin",
    )?;
    context.reject(
        &c.mul(&wet, &c.le(&reconstructed, &context.zero)?)?,
        "positive storage cannot resolve inverse volume",
    )?;
    let residual = c.sub(&reconstructed, volume)?;
    super::accounting::check_residual(
        context,
        &residual,
        &c.maximum(volume, &reconstructed)?,
        "stage inverse-volume residual",
    )?;
    Ok(PoolStage {
        stage,
        reconstructed,
        residual,
    })
}
