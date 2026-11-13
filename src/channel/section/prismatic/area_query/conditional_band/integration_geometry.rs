use super::super::super::{band_geometry::masked, config, hydraulic_geometry::query_error};
use super::super::inverse;
use super::{PrefixIntegrationGeometry, PreparedAreaPrefix, selection::capture};
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection,
};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

impl<R> PreparedAreaPrefix<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn integration_geometry(
        &self,
        a: &mut Arithmetic<'_, R>,
    ) -> HydraulicResult<PrefixIntegrationGeometry<R>> {
        let c = a.client();
        let n = self.endpoint_area.high().numel();
        let m = config::check_work(n, self.band_count)?;
        if self.segments.active.numel() != m {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "prefix lane count={} differs from rows={n} times bands={}. Rebuild the prefix",
                    self.segments.active.numel(),
                    self.band_count
                ),
            });
        }
        let rows = self.segments.rows.rows(a)?;
        let zero = &self.segments.rows.model.source.zero;
        let columns = self.segments.active.reshape(&[m, 1])?;
        let extent = if m != 0 && self.segments.rows.model.source.any(&self.segments.active)? {
            // Prefix origin is each band's start. These are the retained physical coordinates at s=1.
            let local = inverse::recover(
                a,
                &self.segments.rows.model,
                &rows,
                &columns,
                &self.segments.end.above_start,
                &self.segments.end.below_end,
            )
            .map_err(query_error)?;
            let missing = c.mul(
                &self.segments.active,
                &c.eq(&local.candidate, zero)?.reshape(&[m])?,
            )?;
            if let Some(lane) = self.segments.rows.model.source.rejected_lane(&missing)? {
                let original = self
                    .segments
                    .original_rows
                    .narrow(0, lane, 1)?
                    .item::<i64>()?;
                let owner = self
                    .segments
                    .rows
                    .section_owner
                    .narrow(0, lane, 1)?
                    .item::<i64>()?;
                return Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::GeometryUnresolved, query: usize::try_from(original).ok(), section_owner: usize::try_from(owner).ok(), context: "prefix endpoint has no compatible inverse band. Tighten endpoint or geometry certificates".into() });
            }
            masked(a, &columns, &local.depth_above_start, zero).map_err(query_error)?
        } else {
            a.point(&Tensor::zeros(&[m, 1], DType::F64, c.device())?)?
        };
        let flatten =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                Ok(a.bounds(&value.lower().reshape(&[m])?, &value.upper().reshape(&[m])?)?)
            };
        Ok(PrefixIntegrationGeometry {
            face_count: n,
            band_count: self.band_count,
            section_owner: capture(a, &self.segments.rows.section_owner)?,
            original_rows: capture(a, &self.segments.original_rows)?,
            band_index: capture(a, &self.segments.band_index)?,
            valid: capture(a, &rows.valid.reshape(&[m])?)?,
            pointed: capture(a, &rows.pointed.reshape(&[m])?)?,
            extent: flatten(a, &extent)?,
            start_width: flatten(a, &rows.start_width)?,
            slope: flatten(a, &rows.slope)?,
            start_area: flatten(a, &rows.start_area)?,
        })
    }
}
