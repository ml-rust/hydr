use super::super::super::area_point_query::PrismaticAreaPointBandCoordinates;
use super::{ConditionalBandPanel, ConditionalBandSegments};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
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
    /// Area and hydraulic depth at represented parameters `s`, evaluated as one `evaluate_range` endpoint.
    pub(crate) fn point_state(
        &self,
        a: &mut Arithmetic<'_, R>,
        s: &Tensor<R>,
    ) -> HydraulicResult<[Interval<R>; 2]> {
        let parameter = a.point(s)?;
        let state = self.evaluate_point(a, &parameter)?;
        Ok([state.area, state.hydraulic_depth])
    }

    pub(super) fn evaluate_range(
        &self,
        a: &mut Arithmetic<'_, R>,
        s: &Interval<R>,
    ) -> HydraulicResult<ConditionalBandPanel<R>> {
        let lower_parameter = a.point(s.lower())?;
        let upper_parameter = a.point(s.upper())?;
        let lower = self.evaluate_point(a, &lower_parameter)?;
        let upper = self.evaluate_point(a, &upper_parameter)?;
        let c = a.client();
        let hull = |a: &mut Arithmetic<'_, R>,
                    first: &Interval<R>,
                    second: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.minimum(first.lower(), second.lower())?,
                &c.maximum(first.upper(), second.upper())?,
            )?)
        };
        // Qualified geometry is monotone in area. Either sign of the fixed gap selects endpoint extrema.
        let coordinates = PrismaticAreaPointBandCoordinates {
            candidate: lower.coordinates.candidate.clone(),
            area_above_start: hull(
                a,
                lower.coordinates.area_above_start_m2(),
                upper.coordinates.area_above_start_m2(),
            )?,
            area_below_end: hull(
                a,
                lower.coordinates.area_below_end_m2(),
                upper.coordinates.area_below_end_m2(),
            )?,
            depth_above_start: hull(
                a,
                lower.coordinates.depth_above_start_m(),
                upper.coordinates.depth_above_start_m(),
            )?,
            depth_below_end: hull(
                a,
                lower.coordinates.depth_below_end_m(),
                upper.coordinates.depth_below_end_m(),
            )?,
        };
        Ok(ConditionalBandPanel {
            area: hull(a, &lower.area, &upper.area)?,
            depth: hull(a, &lower.depth, &upper.depth)?,
            integral: hull(a, &lower.integral, &upper.integral)?,
            top_width: hull(a, &lower.top_width, &upper.top_width)?,
            hydraulic_depth: hull(a, &lower.hydraulic_depth, &upper.hydraulic_depth)?,
            coordinates,
            gap: a.bounds(self.physical_gap.lower(), self.physical_gap.upper())?,
            active: self.active.clone(),
        })
    }
}
