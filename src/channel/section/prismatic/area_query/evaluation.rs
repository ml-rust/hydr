use super::super::{
    area_point_inverse,
    area_point_query::PrismaticAreaPointBandCoordinates,
    band_geometry::{BandRows, hull, masked},
    context::Context,
    hydraulic_geometry::{Constitutive, GatheredBands, query_error},
};
use super::{admission, chart::SupportedAreaQuery, fields, inverse};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

pub(crate) struct AreaQueryGeometry<R: Runtime<DType = DType>> {
    pub(crate) state: Constitutive<R>,
    pub(crate) coordinates: PrismaticAreaPointBandCoordinates<R>,
}

pub(in crate::channel::section::prismatic) fn recover<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Arc<Context<R>>,
    bands: &GatheredBands<R>,
    supported: &SupportedAreaQuery<R>,
) -> HydraulicResult<AreaQueryGeometry<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    admission::matching_model(Arc::ptr_eq(context, &bands.model))?;
    admission::matching_model(Arc::ptr_eq(context, &supported.query.model))?;
    admission::matching_owners(a, bands, &supported.query.owners)?;
    let c = a.client();
    let n = bands.section_owner.numel();
    let zero = &context.source.zero;
    let qualification = a.point(&Tensor::zeros(&[0], DType::F64, c.device())?)?;
    a.sqrt_nonnegative(&qualification)?;
    let zero_rows = Tensor::zeros(&[n], DType::F64, c.device())?;
    if n == 0 {
        let matrix = Tensor::zeros(&[0, bands.band_count], DType::F64, c.device())?;
        return Ok(AreaQueryGeometry {
            state: Constitutive {
                depth: a.point(&zero_rows)?,
                forward_area: a.point(&zero_rows)?,
                area: a.point(&zero_rows)?,
                top_width: a.point(&zero_rows)?,
                integral: a.point(&zero_rows)?,
                hydraulic_depth: a.point(&zero_rows)?,
                wet_possible: zero_rows.clone(),
                dry_possible: zero_rows,
            },
            coordinates: PrismaticAreaPointBandCoordinates {
                candidate: matrix.clone(),
                area_above_start: a.point(&matrix)?,
                area_below_end: a.point(&matrix)?,
                depth_above_start: a.point(&matrix)?,
                depth_below_end: a.point(&matrix)?,
            },
        });
    }
    let rows = bands.rows(a)?;
    let recovered = recover_endpoints(a, context, &rows, supported)?;
    let candidate = recovered.candidate;
    admission::reject_geometry(
        a,
        &supported.query,
        &c.eq(&c.max(&candidate, &[1], false)?, zero)?,
        "area query has no compatible inverse band. Tighten chart or geometry certificates",
    )?;
    let coordinates = PrismaticAreaPointBandCoordinates {
        area_above_start: masked(a, &candidate, &recovered.area_above_start, zero)
            .map_err(query_error)?,
        area_below_end: masked(a, &candidate, &recovered.area_below_end, zero)
            .map_err(query_error)?,
        depth_above_start: masked(a, &candidate, &recovered.depth_above_start, zero)
            .map_err(query_error)?,
        depth_below_end: masked(a, &candidate, &recovered.depth_below_end, zero)
            .map_err(query_error)?,
        candidate,
    };
    let candidate = coordinates.candidate_mask();
    let local = fields::polynomials(
        a,
        context,
        &rows,
        candidate,
        coordinates.depth_above_start_m(),
    )
    .map_err(query_error)?;
    let absolute =
        area_point_inverse::absolute_depth(a, context, &rows, &coordinates).map_err(query_error)?;
    let all = Tensor::ones(&[n], DType::F64, c.device())?;
    let depth =
        hull(a, candidate, &absolute, &all, bands.cap_depth.upper(), zero).map_err(query_error)?;
    let forward_area = hull(
        a,
        candidate,
        &local.forward_area,
        &all,
        bands.cap_area.upper(),
        zero,
    )
    .map_err(query_error)?;
    let top_width = hull(
        a,
        candidate,
        &local.top_width,
        &all,
        bands.cap_width.upper(),
        zero,
    )
    .map_err(query_error)?;
    let integral = hull(
        a,
        candidate,
        &local.integral,
        &all,
        bands.cap_integral.upper(),
        zero,
    )
    .map_err(query_error)?;
    let hydraulic = hydraulic(
        a,
        context,
        &rows,
        &supported.area,
        &coordinates,
        &local.top_width,
    )?;
    let hydraulic_depth = hull(
        a,
        candidate,
        &hydraulic,
        &all,
        bands.cap_depth.upper(),
        zero,
    )
    .map_err(query_error)?;
    let hydraulic_depth = a.bounds(
        &c.maximum(hydraulic_depth.lower(), zero)?,
        &c.minimum(hydraulic_depth.upper(), depth.upper())?,
    )?;
    Ok(AreaQueryGeometry {
        state: Constitutive {
            depth,
            forward_area,
            area: a.bounds(supported.area.lower(), supported.area.upper())?,
            top_width,
            integral,
            hydraulic_depth,
            wet_possible: c.gt(supported.area.upper(), zero)?,
            dry_possible: c.le(supported.area.lower(), zero)?,
        },
        coordinates,
    })
}

fn recover_endpoints<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    supported: &SupportedAreaQuery<R>,
) -> HydraulicResult<inverse::LocalInverse<R>>
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
    // Clip each physical area endpoint to this band before applying its inverse formula.
    let clip = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
        Ok(a.bounds(
            &c.maximum(
                &c.minimum(value.lower(), rows.area_increment.lower())?,
                zero,
            )?,
            &c.maximum(
                &c.minimum(value.upper(), rows.area_increment.upper())?,
                zero,
            )?,
        )?)
    };
    let lower_above = clip(a, &supported.lower_endpoint.above_start)?;
    let lower_below = clip(a, &supported.lower_endpoint.below_end)?;
    let upper_above = clip(a, &supported.upper_endpoint.above_start)?;
    let upper_below = clip(a, &supported.upper_endpoint.below_end)?;
    let lower = inverse::recover(
        a,
        context,
        rows,
        &supported.candidate,
        &lower_above,
        &lower_below,
    )
    .map_err(query_error)?;
    let upper = inverse::recover(
        a,
        context,
        rows,
        &supported.candidate,
        &upper_above,
        &upper_below,
    )
    .map_err(query_error)?;
    let candidate = c.mul(&lower.candidate, &upper.candidate)?;
    let bounds = |a: &mut Arithmetic<'_, R>,
                  lower: &Tensor<R>,
                  upper: &Tensor<R>|
     -> HydraulicResult<Interval<R>> {
        Ok(a.bounds(
            &c.where_cond(&candidate, lower, zero)?,
            &c.where_cond(&candidate, upper, zero)?,
        )?)
    };
    // Physical depth increases with area. Remaining distances decrease with area.
    Ok(inverse::LocalInverse {
        area_above_start: bounds(
            a,
            lower.area_above_start.lower(),
            upper.area_above_start.upper(),
        )?,
        area_below_end: bounds(
            a,
            upper.area_below_end.lower(),
            lower.area_below_end.upper(),
        )?,
        depth_above_start: bounds(
            a,
            lower.depth_above_start.lower(),
            upper.depth_above_start.upper(),
        )?,
        depth_below_end: bounds(
            a,
            upper.depth_below_end.lower(),
            lower.depth_below_end.upper(),
        )?,
        candidate,
    })
}

pub(in crate::channel::section::prismatic) fn hydraulic<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Context<R>,
    rows: &BandRows<R>,
    area: &Interval<R>,
    coordinates: &PrismaticAreaPointBandCoordinates<R>,
    width: &Interval<R>,
) -> HydraulicResult<Interval<R>>
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
    let one = &context.source.one;
    let candidate = coordinates.candidate_mask();
    let pointed = c.mul(candidate, &rows.pointed)?;
    let ordinary = c.mul(candidate, &c.eq(&pointed, zero)?)?;
    let endpoint = |a: &mut Arithmetic<'_, R>, h: &Tensor<R>| -> HydraulicResult<Interval<R>> {
        let local = a.point(h)?;
        let values =
            fields::polynomials(a, context, rows, candidate, &local).map_err(query_error)?;
        let numerator = masked(a, &ordinary, &values.forward_area, zero).map_err(query_error)?;
        let denominator = masked(a, &ordinary, &values.top_width, one).map_err(query_error)?;
        let quotient = a.div_positive_interval(&numerator, &denominator)?;
        let two = c.mul_scalar(one, 2.0)?;
        let bottom = a.div_positive(&local, &two)?;
        Ok(a.bounds(
            &c.where_cond(&pointed, bottom.lower(), quotient.lower())?,
            &c.where_cond(&pointed, bottom.upper(), quotient.upper())?,
        )?)
    };
    let lower = endpoint(a, coordinates.depth_above_start_m().lower())?;
    let upper = endpoint(a, coordinates.depth_above_start_m().upper())?;
    // Compilation proves T^2 - m*A >= 0, so endpoint values bound the complete hydraulic-depth image.
    let monotone = a.bounds(lower.lower(), upper.upper())?;
    let n = area.lower().numel();
    let area_lower = c.maximum(&area.lower().reshape(&[n, 1])?, rows.start_area.lower())?;
    let area_upper = c.minimum(&area.upper().reshape(&[n, 1])?, rows.end_area.upper())?;
    let numerator = a.bounds(
        &c.where_cond(&ordinary, &area_lower, zero)?,
        &c.where_cond(&ordinary, &area_upper, zero)?,
    )?;
    let denominator = masked(a, &ordinary, width, one).map_err(query_error)?;
    let original = a.div_positive_interval(&numerator, &denominator)?;
    let original = a.bounds(
        &c.where_cond(&pointed, monotone.lower(), original.lower())?,
        &c.where_cond(&pointed, monotone.upper(), original.upper())?,
    )?;
    let value = a.bounds(
        &c.maximum(monotone.lower(), original.lower())?,
        &c.minimum(monotone.upper(), original.upper())?,
    )?;
    masked(a, candidate, &value, zero).map_err(query_error)
}
