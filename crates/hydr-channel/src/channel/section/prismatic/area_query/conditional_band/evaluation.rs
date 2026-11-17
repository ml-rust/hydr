use super::super::super::{
    area_point_inverse, area_point_query::PrismaticAreaPointBandCoordinates, band_geometry::masked,
    hydraulic_geometry::query_error,
};
use super::super::{admission, evaluation, fields, inverse};
use super::{ConditionalBandPanel, ConditionalBandSegments};
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn panel(
        &self,
        a: &mut Arithmetic<'_, R>,
        lane_rows: &Tensor<R>,
        s: &Interval<R>,
    ) -> HydraulicResult<ConditionalBandPanel<R>> {
        let selected = self.select(a, lane_rows)?;
        selected.evaluate_range(a, s)
    }

    pub(super) fn reject(
        &self,
        mask: &Tensor<R>,
        reason: PrismaticRiemannRejection,
        message: &str,
    ) -> HydraulicResult<()> {
        if let Some(lane) = self.rows.model.source.rejected_lane(mask)? {
            let original = self.original_rows.narrow(0, lane, 1)?.item::<i64>()?;
            let owner = self.rows.section_owner.narrow(0, lane, 1)?.item::<i64>()?;
            return Err(HydraulicError::Rejected {
                reason,
                query: usize::try_from(original).ok(),
                section_owner: usize::try_from(owner).ok(),
                context: message.into(),
            });
        }
        Ok(())
    }

    pub(super) fn evaluate_point(
        &self,
        a: &mut Arithmetic<'_, R>,
        s: &Interval<R>,
    ) -> HydraulicResult<ConditionalBandPanel<R>> {
        let c = a.client();
        let n = self.active.numel();
        for value in [s.lower(), s.upper()] {
            admission::metadata(a, value, &[n], DType::F64)?;
        }
        if n == 0 {
            let empty = Tensor::zeros(&[0], DType::F64, c.device())?;
            let matrix = Tensor::zeros(&[0, 1], DType::F64, c.device())?;
            return Ok(ConditionalBandPanel {
                area: a.point(&empty)?,
                depth: a.point(&empty)?,
                integral: a.point(&empty)?,
                top_width: a.point(&empty)?,
                hydraulic_depth: a.point(&empty)?,
                gap: a.point(&empty)?,
                active: empty,
                coordinates: PrismaticAreaPointBandCoordinates {
                    candidate: matrix.clone(),
                    area_above_start: a.point(&matrix)?,
                    area_below_end: a.point(&matrix)?,
                    depth_above_start: a.point(&matrix)?,
                    depth_below_end: a.point(&matrix)?,
                },
            });
        }
        let zero = &self.rows.model.source.zero;
        let one = &self.rows.model.source.one;
        admission::reject(
            a,
            &c.maximum(&c.lt(s.lower(), zero)?, &c.gt(s.upper(), one)?)?,
            "conditional band parameter lies outside [0,1]. Supply a panel inside its proved segment",
        )?;
        let columns = self.active.reshape(&[n, 1])?;
        let rows = self.rows.rows(a)?;
        let delta = a.bounds(
            &self.physical_gap.lower().reshape(&[n, 1])?,
            &self.physical_gap.upper().reshape(&[n, 1])?,
        )?;
        let s = a.bounds(&s.lower().reshape(&[n, 1])?, &s.upper().reshape(&[n, 1])?)?;
        let identity = a.point(one)?;
        let reverse = a.sub(&identity, &s)?;
        let reverse = a.bounds(
            &c.maximum(reverse.lower(), zero)?,
            &c.minimum(reverse.upper(), one)?,
        )?;
        let forward_offset = a.mul(&delta, &s)?;
        let reverse_offset = a.mul(&delta, &reverse)?;
        let above_forward = a.add(&self.start.above_start, &forward_offset)?;
        let above_reverse = a.sub(&self.end.above_start, &reverse_offset)?;
        let below_forward = a.sub(&self.start.below_end, &forward_offset)?;
        let below_reverse = a.add(&self.end.below_end, &reverse_offset)?;
        let local = |a: &mut Arithmetic<'_, R>,
                     forward: &Interval<R>,
                     reverse: &Interval<R>,
                     start: &Interval<R>,
                     end: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            let hull_lower = c.minimum(start.lower(), end.lower())?;
            let hull_upper = c.maximum(start.upper(), end.upper())?;
            let lower = c.maximum(
                &c.maximum(forward.lower(), reverse.lower())?,
                &c.maximum(&hull_lower, zero)?,
            )?;
            let upper = c.minimum(
                &c.minimum(forward.upper(), reverse.upper())?,
                &c.minimum(&hull_upper, rows.area_increment.upper())?,
            )?;
            self.reject(&c.mul(&self.active, &c.gt(&lower, &upper)?.reshape(&[n])?)?, PrismaticRiemannRejection::GeometryUnresolved,
                "conditional band local coordinates are incompatible. Tighten endpoint or geometry certificates")?;
            Ok(a.bounds(
                &c.where_cond(&columns, &lower, zero)?,
                &c.where_cond(&columns, &upper, zero)?,
            )?)
        };
        let above = local(
            a,
            &above_forward,
            &above_reverse,
            &self.start.above_start,
            &self.end.above_start,
        )?;
        let below = local(
            a,
            &below_forward,
            &below_reverse,
            &self.start.below_end,
            &self.end.below_end,
        )?;
        let start_area = masked(a, &columns, &rows.start_area, zero).map_err(query_error)?;
        let end_area = masked(a, &columns, &rows.end_area, zero).map_err(query_error)?;
        let from_start = a.add(&start_area, &above)?;
        let from_end = a.sub(&end_area, &below)?;
        let lower = c.maximum(
            &c.maximum(
                &from_start.lower().reshape(&[n])?,
                &from_end.lower().reshape(&[n])?,
            )?,
            self.endpoint_hull.lower(),
        )?;
        let upper = c.minimum(
            &c.minimum(
                &from_start.upper().reshape(&[n])?,
                &from_end.upper().reshape(&[n])?,
            )?,
            self.endpoint_hull.upper(),
        )?;
        self.reject(&c.mul(&self.active, &c.gt(&lower, &upper)?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "conditional band absolute area is incompatible. Tighten endpoint or geometry certificates")?;
        let area = a.bounds(
            &c.where_cond(&self.active, &lower, zero)?,
            &c.where_cond(&self.active, &upper, zero)?,
        )?;
        let nonpositive = c.mul(&self.active, &c.le(area.lower(), zero)?)?;
        let first_band = c.cast(
            &c.eq(
                &self.band_index,
                &Tensor::zeros(&[n], DType::I64, c.device())?,
            )?,
            DType::F64,
        )?;
        self.reject(
            &c.mul(&nonpositive, &first_band)?,
            PrismaticRiemannRejection::QuadratureUnresolved,
            "conditional band panel reaches the dry first band. Use separate bottom regularization",
        )?;
        self.reject(&nonpositive, PrismaticRiemannRejection::GeometryUnresolved,
            "conditional wet panel area lacks a positive lower bound. Tighten endpoint or geometry certificates")?;
        let recovered = inverse::recover(a, &self.rows.model, &rows, &columns, &above, &below)
            .map_err(query_error)?;
        self.reject(&c.mul(&self.active, &c.eq(&recovered.candidate, zero)?.reshape(&[n])?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "conditional band inverse has no compatible state. Tighten endpoint or geometry certificates")?;
        let coordinates = PrismaticAreaPointBandCoordinates {
            candidate: recovered.candidate,
            area_above_start: recovered.area_above_start,
            area_below_end: recovered.area_below_end,
            depth_above_start: recovered.depth_above_start,
            depth_below_end: recovered.depth_below_end,
        };
        let values = fields::polynomials(
            a,
            &self.rows.model,
            &rows,
            coordinates.candidate_mask(),
            coordinates.depth_above_start_m(),
        )
        .map_err(query_error)?;
        self.reject(&c.mul(&self.active, &c.le(values.top_width.lower(), zero)?.reshape(&[n])?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "conditional wet panel width lacks a positive lower bound. Tighten endpoint or geometry certificates")?;
        let hydraulic = evaluation::hydraulic(
            a,
            &self.rows.model,
            &rows,
            &area,
            &coordinates,
            &values.top_width,
        )?;
        let depth = area_point_inverse::absolute_depth(a, &self.rows.model, &rows, &coordinates)
            .map_err(query_error)?;
        Ok(ConditionalBandPanel {
            area,
            depth: a.bounds(&depth.lower().reshape(&[n])?, &depth.upper().reshape(&[n])?)?,
            integral: a.bounds(
                &values.integral.lower().reshape(&[n])?,
                &values.integral.upper().reshape(&[n])?,
            )?,
            top_width: a.bounds(
                &values.top_width.lower().reshape(&[n])?,
                &values.top_width.upper().reshape(&[n])?,
            )?,
            hydraulic_depth: a.bounds(
                &hydraulic.lower().reshape(&[n])?,
                &hydraulic.upper().reshape(&[n])?,
            )?,
            coordinates,
            gap: a.bounds(self.physical_gap.lower(), self.physical_gap.upper())?,
            active: self.active.clone(),
        })
    }
}
