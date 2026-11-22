//! Certified closed-conduit barrel geometry inside checked interval calls.
//!
//! Every quantity is a solvr `Interval`. Callers open one `with_interval_arithmetic` call and
//! pass its context to these functions, so no function here adds a probe or a host read.

pub(crate) mod arcs;
#[cfg(test)]
mod certificate_rows;
pub(crate) mod constants;
pub(crate) mod critical;
pub(crate) mod inlet;
mod inversion;
pub(crate) mod normal;
pub(crate) mod outlet;
mod roots;
pub(crate) mod section;

pub(crate) use roots::BarrelDepths;

use crate::structures::{CulvertBarrel, StructureError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, NextRepresentableOps, ScalarOps,
    TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, IntervalContext};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::CertifiedBisectionAlgorithms;

/// Interval context on the barrel batch's client.
pub(crate) type Arithmetic<'a, R> = IntervalContext<'a, R, <R as Runtime>::Client>;

/// Client operations that barrel geometry needs.
pub trait BarrelClient<R: Runtime<DType = DType>>:
    TensorOps<R>
    + CompareOps<R>
    + NextRepresentableOps<R>
    + ScalarOps<R>
    + TwoSumOps<R>
    + CorrectlyRoundedOps<R>
    + RuntimeClient<R>
{
}

impl<R, C> BarrelClient<R> for C
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + CorrectlyRoundedOps<R>
        + RuntimeClient<R>,
{
}

/// Host barrels that name rows in errors, through an optional device map of row indices.
pub(crate) struct RowNames<'a, R: Runtime<DType = DType>> {
    barrels: &'a [CulvertBarrel],
    rows: Option<&'a Tensor<R>>,
}

impl<R: Runtime<DType = DType>> Clone for RowNames<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: Runtime<DType = DType>> Copy for RowNames<'_, R> {}

impl<'a, R: Runtime<DType = DType>> RowNames<'a, R> {
    /// Row `i` is `barrels[i]`.
    pub(crate) fn new(barrels: &'a [CulvertBarrel]) -> Self {
        Self {
            barrels,
            rows: None,
        }
    }

    /// Row `i` is `barrels[rows[i]]`, with `rows` an I64 tensor.
    pub(crate) fn mapped(barrels: &'a [CulvertBarrel], rows: &'a Tensor<R>) -> Self {
        Self {
            barrels,
            rows: Some(rows),
        }
    }

    /// Barrel of `row`. A mapped list reads its map to the host, so only error paths call this.
    pub(crate) fn barrel(&self, row: usize) -> Result<Option<CulvertBarrel>, StructureError> {
        let index = match self.rows {
            None => Some(row),
            Some(rows) => rows
                .try_to_vec::<i64>()?
                .get(row)
                .and_then(|&index| usize::try_from(index).ok()),
        };
        Ok(index.and_then(|index| self.barrels.get(index)).copied())
    }
}

/// Client operations a structure batch needs beyond its scalar laws: the culvert law's barrel
/// geometry, certified roots and certified quadrature.
///
/// Every client that implements those traits implements this one.
pub trait CulvertClient<R: Runtime<DType = DType>>:
    BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>
{
}

impl<R, C> CulvertClient<R> for C
where
    R: Runtime<DType = DType>,
    C: BarrelClient<R> + CertifiedBisectionAlgorithms<R> + CertifiedSimpsonAlgorithms<R>,
{
}

/// Intersects `x` with `[low, high]`, collapsing an interval wholly outside to the nearer end.
pub(crate) fn clip<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    low: f64,
    high: f64,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = x.lower();
    let floor = Tensor::<R>::full_scalar(like.shape(), DType::F64, low, like.device())?;
    let ceiling = Tensor::<R>::full_scalar(like.shape(), DType::F64, high, like.device())?;
    let lower = c.minimum(&c.maximum(x.lower(), &floor)?, &ceiling)?;
    let upper = c.maximum(&c.minimum(x.upper(), &ceiling)?, &floor)?;
    a.bounds(&lower, &upper)
}

/// Takes `on` where `mask` is 1 and `off` where it is 0.
pub(crate) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    on: &Interval<R>,
    off: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.where_cond(mask, on.lower(), off.lower())?;
    let upper = c.where_cond(mask, on.upper(), off.upper())?;
    a.bounds(&lower, &upper)
}
