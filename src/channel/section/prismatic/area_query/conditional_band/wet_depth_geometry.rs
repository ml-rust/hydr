use super::super::super::context::Context;
use super::{ConditionalBandSegments, selection::capture};
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicResult, PrismaticRiemannRejection,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::sync::Arc;

pub(crate) struct WetDepthGeometry<R: Runtime<DType = DType>> {
    model: Arc<Context<R>>,
    owners: Tensor<R>,
    original_rows: Tensor<R>,
    active: Tensor<R>,
    area: Interval<R>,
    width: Interval<R>,
    slope: Interval<R>,
    depth_gap: CertifiedExpansion<R>,
    physical_depth_gap: Interval<R>,
    area_hull: Interval<R>,
    width_hull: Interval<R>,
}

impl<R: Runtime<DType = DType>> WetDepthGeometry<R> {
    pub(crate) fn owners(&self) -> &Tensor<R> {
        &self.owners
    }
    pub(crate) fn original_rows(&self) -> &Tensor<R> {
        &self.original_rows
    }
    pub(crate) fn active_mask(&self) -> &Tensor<R> {
        &self.active
    }
    pub(crate) fn start_area(&self) -> &Interval<R> {
        &self.area
    }
    pub(crate) fn start_width(&self) -> &Interval<R> {
        &self.width
    }
    pub(crate) fn slope(&self) -> &Interval<R> {
        &self.slope
    }
    pub(crate) fn retained_depth_gap(&self) -> &CertifiedExpansion<R> {
        &self.depth_gap
    }
    pub(crate) fn physical_depth_gap(&self) -> &Interval<R> {
        &self.physical_depth_gap
    }
    pub(crate) fn area_hull(&self) -> &Interval<R> {
        &self.area_hull
    }
    pub(crate) fn width_hull(&self) -> &Interval<R> {
        &self.width_hull
    }
    pub(crate) fn same_model(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.model, &other.model)
    }
}

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn wet_depth_geometry(
        &self,
        a: &mut Arithmetic<'_, R>,
        lane_rows: &Tensor<R>,
    ) -> HydraulicResult<WetDepthGeometry<R>> {
        let selected = self.select(a, lane_rows)?;
        let c = a.client();
        let n = selected.active.numel();
        let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
        let one = Tensor::ones(&[n], DType::F64, c.device())?;
        let first_parameter = a.point(&zero)?;
        let last_parameter = a.point(&one)?;
        let first = selected.evaluate_point(a, &first_parameter)?;
        let last = selected.evaluate_point(a, &last_parameter)?;
        let safe = |a: &mut Arithmetic<'_, R>,
                    value: &Interval<R>,
                    fallback: &Tensor<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.where_cond(&selected.active, value.lower(), fallback)?,
                &c.where_cond(&selected.active, value.upper(), fallback)?,
            )?)
        };
        let first_width = safe(a, &first.top_width, &one)?;
        let last_width = safe(a, &last.top_width, &one)?;
        let first_expansion = a.expansion_from_interval(&first_width)?;
        let last_expansion = a.expansion_from_interval(&last_width)?;
        let half = a.expansion_point(&c.mul_scalar(&one, 0.5)?)?;
        let half_first = a.expansion_mul(&first_expansion, &half)?;
        let half_last = a.expansion_mul(&last_expansion, &half)?;
        let mean = a.expansion_add(&half_first, &half_last)?;
        let mean = a.expansion_interval(&mean)?;
        // The physical mean lies between endpoint widths, including subnormal half-products.
        let mean = a.bounds(
            &c.maximum(
                mean.lower(),
                &c.minimum(first_width.lower(), last_width.lower())?,
            )?,
            &c.minimum(
                mean.upper(),
                &c.maximum(first_width.upper(), last_width.upper())?,
            )?,
        )?;
        let mean = a.expansion_from_interval(&mean)?;
        let depth_gap = a.expansion_div_positive(&selected.gap, &mean)?;
        let projection = a.expansion_interval(&depth_gap)?;
        let from_start = a.sub(
            last.coordinates.depth_above_start_m(),
            first.coordinates.depth_above_start_m(),
        )?;
        let from_end = a.sub(
            first.coordinates.depth_below_end_m(),
            last.coordinates.depth_below_end_m(),
        )?;
        let thickness = selected.rows.thickness.upper().reshape(&[n])?;
        let lower = c.maximum(
            &c.maximum(
                &c.maximum(projection.lower(), &from_start.lower().reshape(&[n])?)?,
                &from_end.lower().reshape(&[n])?,
            )?,
            &c.neg(&thickness)?,
        )?;
        let upper = c.minimum(
            &c.minimum(
                &c.minimum(projection.upper(), &from_start.upper().reshape(&[n])?)?,
                &from_end.upper().reshape(&[n])?,
            )?,
            &thickness,
        )?;
        let lower = c.where_cond(
            &c.ge(selected.physical_gap.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?;
        let upper = c.where_cond(
            &c.le(selected.physical_gap.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?;
        selected.reject(&c.mul(&selected.active, &c.gt(&lower, &upper)?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "wet depth gap is incompatible with local endpoints. Tighten endpoint or geometry certificates")?;
        let physical_depth_gap = a.bounds(
            &c.where_cond(&selected.active, &lower, &zero)?,
            &c.where_cond(&selected.active, &upper, &zero)?,
        )?;
        let hull = |a: &mut Arithmetic<'_, R>,
                    first: &Interval<R>,
                    last: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.minimum(first.lower(), last.lower())?,
                &c.maximum(first.upper(), last.upper())?,
            )?)
        };
        let area_hull = hull(a, &first.area, &last.area)?;
        let width_hull = hull(a, &first.top_width, &last.top_width)?;
        let slope = a.bounds(
            &selected.rows.slope.lower().reshape(&[n])?,
            &selected.rows.slope.upper().reshape(&[n])?,
        )?;
        Ok(WetDepthGeometry {
            model: Arc::clone(&selected.rows.model),
            owners: capture(a, selected.owners())?,
            original_rows: capture(a, selected.original_rows())?,
            active: capture(a, selected.active_mask())?,
            area: safe(a, &first.area, &one)?,
            width: first_width,
            slope: safe(a, &slope, &zero)?,
            depth_gap,
            physical_depth_gap,
            area_hull: safe(a, &area_hull, &one)?,
            width_hull: safe(a, &width_hull, &one)?,
        })
    }
}
