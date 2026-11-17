use super::super::{config, hydraulic_geometry::GatheredBands};
#[cfg(test)]
use super::chart::AreaAnchor;
use crate::channel::{
    flow::surveyed::error::{
        Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
    },
    section::CrossSectionError,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps, TypeConversionOps, UnaryOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) fn matching_model(matches: bool) -> HydraulicResult<()> {
    if !matches {
        return Err(CrossSectionError::input(
            "area query inputs use different compiled models. Rebuild inputs from one prismatic batch",
        ).into());
    }
    Ok(())
}

pub(super) fn reject_geometry<R>(
    a: &Arithmetic<'_, R>,
    query: &super::chart::AreaQuery<R>,
    mask: &Tensor<R>,
    message: &str,
) -> HydraulicResult<()>
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
    let rows = c.nonzero(mask)?.flatten()?;
    if rows.numel() != 0 {
        let first = rows.narrow(0, 0, 1)?;
        let original = c
            .index_select(&query.original_rows, 0, &first)?
            .item::<i64>()?;
        let owner = c.index_select(&query.owners, 0, &first)?.item::<i64>()?;
        return Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::GeometryUnresolved,
            query: usize::try_from(original).ok(),
            section_owner: usize::try_from(owner).ok(),
            context: message.into(),
        });
    }
    Ok(())
}

pub(in crate::channel::section::prismatic) fn metadata<R>(
    a: &Arithmetic<'_, R>,
    value: &Tensor<R>,
    shape: &[usize],
    dtype: DType,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if value.shape() != shape
        || value.dtype() != dtype
        || !value.device().is_same(a.client().device())
    {
        return Err(CrossSectionError::input(format!("area query shape={:?}, dtype={:?}, device={}: supply shape={shape:?}, dtype={dtype:?}, device={}", value.shape(), value.dtype(), value.device().id(), a.client().device().id())).into());
    }
    Ok(())
}

pub(super) fn reject<R>(
    a: &Arithmetic<'_, R>,
    mask: &Tensor<R>,
    message: &str,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if mask.numel() != 0
        && a.client()
            .max(&mask.flatten()?, &[0], false)?
            .item::<f64>()?
            != 0.0
    {
        return Err(CrossSectionError::input(message).into());
    }
    Ok(())
}

fn finite<R>(a: &Arithmetic<'_, R>, value: &Tensor<R>) -> HydraulicResult<()>
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
    reject(
        a,
        &c.maximum(
            &c.cast(&c.isnan(value)?, DType::F64)?,
            &c.cast(&c.isinf(value)?, DType::F64)?,
        )?,
        "area query contains nonfinite values. Supply finite components and parameter endpoints",
    )
}

pub(super) fn matching_owners<R>(
    a: &Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    owners: &Tensor<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    metadata(a, owners, bands.section_owner.shape(), DType::I64)?;
    reject(
        a,
        &a.client()
            .cast(&a.client().ne(owners, &bands.section_owner)?, DType::F64)?,
        "area query owners differ from the cached geometry rows. Select matching query and table rows",
    )
}

pub(super) fn nonnegative_parameter<R>(
    a: &Arithmetic<'_, R>,
    parameter: &Interval<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(&[], DType::F64, a.client().device())?;
    reject(
        a,
        &a.client().lt(parameter.lower(), &zero)?,
        "cap deficit parameter is negative. Supply a nonnegative deficit range",
    )
}

pub(super) fn chart<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    #[cfg(test)] anchor: &AreaAnchor<R>,
    scale: &CertifiedExpansion<R>,
    parameter: &Interval<R>,
) -> HydraulicResult<()>
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
    let shape = bands.section_owner.shape();
    config::check_work(bands.section_owner.numel(), bands.band_count + 1)?;
    for value in [
        scale.high(),
        scale.low(),
        scale.abs_error(),
        parameter.lower(),
        parameter.upper(),
    ] {
        metadata(a, value, shape, DType::F64)?;
        finite(a, value)?;
    }
    let zero = Tensor::zeros(&[], DType::F64, c.device())?;
    reject(
        a,
        &c.ne(scale.abs_error(), &zero)?,
        "area query scale radius is nonzero. Supply an exact root scale",
    )?;
    let normalized = a.expansion_bounds(scale.high(), scale.low(), scale.abs_error())?;
    let negative = c.maximum(
        &c.lt(normalized.high(), &zero)?,
        &c.mul(
            &c.eq(normalized.high(), &zero)?,
            &c.lt(normalized.low(), &zero)?,
        )?,
    )?;
    reject(
        a,
        &negative,
        "area query scale is negative. Supply a nonnegative root scale",
    )?;
    reject(
        a,
        &c.gt(parameter.lower(), parameter.upper())?,
        "area query parameter endpoints are unordered. Supply an ordered interval",
    )?;
    #[cfg(test)]
    if let AreaAnchor::BandStart(index) | AreaAnchor::BandEnd(index) = anchor {
        metadata(a, index, shape, DType::I64)?;
        let zero_index = Tensor::from_slice(&[0_i64], &[], c.device())?;
        let count = i64::try_from(bands.band_count).map_err(|_| HydraulicError::Numerical {
            context: "area query band count exceeds I64. Split geometry".into(),
        })?;
        let limit = Tensor::from_slice(&[count], &[], c.device())?;
        let invalid = c.maximum(&c.lt(index, &zero_index)?, &c.ge(index, &limit)?)?;
        reject(
            a,
            &c.cast(&invalid, DType::F64)?,
            "area query anchor band is outside compiled columns. Supply a valid physical band",
        )?;
        let valid = c
            .gather(&bands.valid, 1, &index.reshape(&[index.numel(), 1])?)?
            .reshape(shape)?;
        reject(
            a,
            &c.eq(&valid, &zero)?,
            "area query anchor band is inactive. Supply a valid physical band",
        )?;
    }
    Ok(())
}
