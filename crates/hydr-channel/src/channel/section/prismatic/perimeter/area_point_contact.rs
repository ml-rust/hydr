use super::super::{
    area_point_query::PrismaticAreaPointRecovery,
    band_geometry::{intersect, masked},
    compilation_bounds::nonnegative,
    query_input::{Arithmetic, QueryResult, reject},
};
use super::{compilation::ContactTable, contact::Fields, local_contact::local_contact};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
    SortingOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(super) fn fields<R>(
    a: &mut Arithmetic<'_, R>,
    table: &ContactTable<R>,
    geometry: &PrismaticAreaPointRecovery<R>,
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
    let context = &geometry.context;
    let count = geometry.owners().numel();
    let zero = &context.source.zero;
    let one = &context.source.one;
    let zeros = Tensor::zeros(&[count], DType::F64, c.device())?;
    if count == 0 {
        return Ok(Fields {
            perimeter: a.point(&zeros)?,
            radius: a.point(&zeros)?,
        });
    }
    let wet = c.eq(geometry.dry_mask(), zero)?;
    let coordinates = geometry.band_coordinates();
    let mut present = zeros.clone();
    let mut lower = zeros.clone();
    let mut upper = zeros;
    for band in 0..context.maximum_bands {
        let candidate = c.mul(&wet, &column(coordinates.candidate_mask(), band)?)?;
        let query_rows = c.nonzero(&candidate)?.flatten()?;
        if query_rows.numel() == 0 {
            continue;
        }
        let owners = c.index_select(geometry.owners(), 0, &query_rows)?;
        let gather = |a: &mut Arithmetic<'_, R>,
                      value: &Interval<R>,
                      indices: &Tensor<R>|
         -> QueryResult<Interval<R>> {
            Ok(a.bounds(
                &c.index_select(&column(value.lower(), band)?, 0, indices)?,
                &c.index_select(&column(value.upper(), band)?, 0, indices)?,
            )?)
        };
        let thickness = gather(a, &table.thickness, &owners)?;
        reject(
            context,
            &owners,
            &c.le(thickness.lower(), zero)?,
            true,
            "contact thickness has no positive lower bound. Review profile conditioning",
        )?;
        let start = gather(a, &table.start, &owners)?;
        let left = gather(a, &table.left, &owners)?;
        let right = gather(a, &table.right, &owners)?;
        let h = gather(a, coordinates.depth_above_start_m(), &query_rows)?;
        let e = gather(a, coordinates.depth_below_end_m(), &query_rows)?;
        let left_partial = local_contact(a, &h, &left, &thickness)?;
        let right_partial = local_contact(a, &h, &right, &thickness)?;
        let start_left = a.add(&start, &left_partial)?;
        let from_start = a.add(&start_left, &right_partial)?;
        let left_remaining = local_contact(a, &e, &left, &thickness)?;
        let right_remaining = local_contact(a, &e, &right, &thickness)?;
        let end_left = a.add(&start, &left)?;
        let end = a.add(&end_left, &right)?;
        let without_left = a.sub(&end, &left_remaining)?;
        let from_end = a.sub(&without_left, &right_remaining)?;
        let mask = Tensor::ones(&[query_rows.numel()], DType::F64, c.device())?;
        let (perimeter, mask) = intersect(
            a,
            &mask,
            &from_start,
            from_end.lower(),
            from_end.upper(),
            zero,
        )?;
        let cap_upper = c.index_select(table.cap.upper(), 0, &owners)?;
        let (perimeter, mask) = intersect(a, &mask, &perimeter, zero, &cap_upper, zero)?;
        let old_present = c.index_select(&present, 0, &query_rows)?;
        let old_lower = c.index_select(&lower, 0, &query_rows)?;
        let old_upper = c.index_select(&upper, 0, &query_rows)?;
        let selected_lower = c.where_cond(
            &old_present,
            &c.minimum(&old_lower, perimeter.lower())?,
            perimeter.lower(),
        )?;
        let selected_upper = c.maximum(&old_upper, perimeter.upper())?;
        lower = c.index_put(
            &lower,
            0,
            &query_rows,
            &c.where_cond(&mask, &selected_lower, &old_lower)?,
        )?;
        upper = c.index_put(
            &upper,
            0,
            &query_rows,
            &c.where_cond(&mask, &selected_upper, &old_upper)?,
        )?;
        present = c.index_put(&present, 0, &query_rows, &c.maximum(&old_present, &mask)?)?;
    }
    reject(
        context,
        geometry.owners(),
        &c.mul(&wet, &c.eq(&present, zero)?)?,
        true,
        "no contact band encloses the requested area point. Review profile conditioning",
    )?;
    let perimeter = a.bounds(&lower, &upper)?;
    reject(
        context,
        geometry.owners(),
        &c.mul(&wet, &c.le(perimeter.lower(), zero)?)?,
        true,
        "wet perimeter has no represented positive lower bound. Review profile conditioning",
    )?;
    let denominator = masked(a, &wet, &perimeter, one)?;
    let numerator = a.expansion_interval(geometry.requested_areas_m2())?;
    let radius = a.div_positive_interval(&numerator, &denominator)?;
    let radius = nonnegative(a, radius, zero)?;
    let radius = masked(a, &wet, &radius, zero)?;
    Ok(Fields { perimeter, radius })
}

fn column<R: Runtime<DType = DType>>(
    value: &Tensor<R>,
    band: usize,
) -> Result<Tensor<R>, numr::error::Error> {
    Ok(value.narrow(1, band, 1)?.squeeze(Some(1)))
}
