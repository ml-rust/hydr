use super::super::{
    band_geometry::{BandRows, hull, intersect, masked},
    compilation_bounds::nonnegative,
    context::Context,
    query_input::{self, Arithmetic, QueryResult, reject},
};
use super::compilation::ContactTable;
use crate::channel::section::CrossSectionError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

#[derive(Debug)]
pub(super) struct Fields<R: Runtime<DType = DType>> {
    pub perimeter: Interval<R>,
    pub radius: Interval<R>,
}

pub(super) fn evaluate<R>(
    context: &Context<R>,
    table: &ContactTable<R>,
    owners: &Tensor<R>,
    depth: &Interval<R>,
    dry: &Tensor<R>,
    area: &Interval<R>,
) -> Result<Fields<R>, CrossSectionError>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    query_input::checked(context, "prismatic depth perimeter", |a| {
        fields(a, context, table, owners, depth, dry, area)
    })
}

pub(super) fn fields<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    table: &ContactTable<R>,
    owners: &Tensor<R>,
    depth: &Interval<R>,
    dry: &Tensor<R>,
    area: &Interval<R>,
) -> QueryResult<Fields<R>>
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
    let count = owners.numel();
    if count == 0 {
        return Ok(Fields {
            perimeter: a.point(depth.lower())?,
            radius: a.point(depth.lower())?,
        });
    }
    let zero = &context.source.zero;
    let one = &context.source.one;
    let wet = c.eq(dry, zero)?;
    let rows = BandRows::gather(a, context, owners)?;
    let gather = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| {
        a.bounds(
            &c.index_select(value.lower(), 0, owners)?,
            &c.index_select(value.upper(), 0, owners)?,
        )
    };
    let start = gather(a, &table.start)?;
    let thickness = gather(a, &table.thickness)?;
    let left = gather(a, &table.left)?;
    let right = gather(a, &table.right)?;
    let cap = gather(a, &table.cap)?;
    let query = a.bounds(
        &depth.lower().reshape(&[count, 1])?,
        &depth.upper().reshape(&[count, 1])?,
    )?;
    let candidate = c.mul(
        &rows.valid,
        &c.mul(
            &wet.reshape(&[count, 1])?,
            &c.mul(
                &c.ge(query.upper(), rows.start_depth.lower())?,
                &c.le(query.lower(), rows.end_depth.upper())?,
            )?,
        )?,
    )?;
    let query = masked(a, &candidate, &query, zero)?;
    let start_depth = masked(a, &candidate, &rows.start_depth, zero)?;
    let local = a.sub(&query, &start_depth)?;
    let (local, candidate) = intersect(a, &candidate, &local, zero, thickness.upper(), zero)?;
    let divisor = masked(a, &candidate, &thickness, one)?;
    let fraction = a.div_positive_interval(&local, &divisor)?;
    let (fraction, candidate) = intersect(a, &candidate, &fraction, zero, one, zero)?;
    let left = masked(a, &candidate, &left, zero)?;
    let right = masked(a, &candidate, &right, zero)?;
    let increment = a.add(&left, &right)?;
    let partial = a.mul(&fraction, &increment)?;
    let start = masked(a, &candidate, &start, zero)?;
    let perimeter = a.add(&start, &partial)?;
    let (perimeter, candidate) = intersect(
        a,
        &candidate,
        &perimeter,
        zero,
        &cap.upper().reshape(&[count, 1])?,
        zero,
    )?;
    reject(
        context,
        owners,
        &c.mul(&wet, &c.eq(&c.max(&candidate, &[1], false)?, zero)?)?,
        true,
        "no contact band encloses this depth. Check profile conditioning",
    )?;
    let perimeter = hull(a, &candidate, &perimeter, &wet, cap.upper(), zero)?;
    reject(
        context,
        owners,
        &c.mul(&wet, &c.le(perimeter.lower(), zero)?)?,
        true,
        "wet perimeter has no represented positive lower bound. Check profile conditioning",
    )?;
    let denominator = masked(a, &wet, &perimeter, one)?;
    let radius = a.div_positive_interval(area, &denominator)?;
    let radius = nonnegative(a, radius, zero)?;
    let radius = masked(a, &wet, &radius, zero)?;
    Ok(Fields { perimeter, radius })
}
