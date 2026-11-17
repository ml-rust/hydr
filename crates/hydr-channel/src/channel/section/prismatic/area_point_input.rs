use super::super::CrossSectionError;
use super::{
    context::Context,
    query_input::{self, Arithmetic, QueryResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct Inputs<R: Runtime<DType = DType>> {
    pub owners: Tensor<R>,
    pub areas: CertifiedExpansion<R>,
    pub wet: Tensor<R>,
    pub dry: Tensor<R>,
}

pub(super) fn metadata<R: Runtime<DType = DType>>(
    context: &Context<R>,
    owners: &Tensor<R>,
    areas: &CertifiedExpansion<R>,
) -> Result<usize, CrossSectionError> {
    let count = query_input::metadata(context, owners, areas.high())?;
    query_input::metadata(context, owners, areas.low())?;
    query_input::metadata(context, owners, areas.abs_error())?;
    Ok(count)
}

pub(super) fn capture<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    owners: &Tensor<R>,
    areas: &CertifiedExpansion<R>,
) -> QueryResult<Inputs<R>>
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
    let zero = &context.source.zero;
    let placeholders = Tensor::zeros(owners.shape(), DType::F64, c.device())?;
    let captured = query_input::capture(context, owners, &placeholders, "area_m2")?;
    if owners.numel() != 0 {
        for value in [areas.high(), areas.low(), areas.abs_error()] {
            let invalid = c.maximum(
                &c.cast(&c.isnan(value)?, DType::F64)?,
                &c.cast(&c.isinf(value)?, DType::F64)?,
            )?;
            query_input::reject(
                context,
                &captured.owners,
                &invalid,
                false,
                "area point contains nonfinite components. Supply finite F64 components",
            )?;
        }
        query_input::reject(
            context,
            &captured.owners,
            &c.ne(areas.abs_error(), zero)?,
            false,
            "area point radius is nonzero. Supply an exact selected area point",
        )?;
    }
    let areas = a.expansion_bounds(areas.high(), areas.low(), areas.abs_error())?;
    let negative = c.maximum(
        &c.lt(areas.high(), zero)?,
        &c.mul(&c.eq(areas.high(), zero)?, &c.lt(areas.low(), zero)?)?,
    )?;
    if owners.numel() != 0 {
        query_input::reject(
            context,
            &captured.owners,
            &negative,
            false,
            "requested area point is negative. Supply a nonnegative area",
        )?;
    }
    let dry = c.mul(&c.eq(areas.high(), zero)?, &c.eq(areas.low(), zero)?)?;
    let wet = c.eq(&dry, zero)?;
    Ok(Inputs {
        owners: captured.owners,
        areas,
        wet,
        dry,
    })
}

/// Subtract interval endpoints before projecting the requested expansion.
pub(super) fn difference<R>(
    a: &mut Arithmetic<'_, R>,
    point: &CertifiedExpansion<R>,
    interval: &Interval<R>,
    reverse: bool,
) -> QueryResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let lower = a.expansion_point(interval.lower())?;
    let upper = a.expansion_point(interval.upper())?;
    let (lower, upper) = if reverse {
        (
            a.expansion_sub(&lower, point)?,
            a.expansion_sub(&upper, point)?,
        )
    } else {
        (
            a.expansion_sub(point, &upper)?,
            a.expansion_sub(point, &lower)?,
        )
    };
    let lower = a.expansion_interval(&lower)?;
    let upper = a.expansion_interval(&upper)?;
    Ok(a.bounds(lower.lower(), upper.upper())?)
}
