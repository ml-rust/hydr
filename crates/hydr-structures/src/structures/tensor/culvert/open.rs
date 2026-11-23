//! The rows of a barrel context as an outlet march reads them.
//!
//! On each floored row the view describes the open section above the bed: rise `D'`, full area
//! `A'_full`, weighted full perimeter `P_n,full = P_w,full + rho^(3/2) b` with the crown of a
//! box, friction radius `A'_full / P_n,full`, and the log peak conveyance in units of `D'`. A
//! floored circle leaves the circle path, because its open section is an arc row of the march.
//! Every other field and every row without a floor keep the context's tensors.

use super::context::{Bounds, TensorBarrelContext};
use crate::structures::StructureError;
use crate::structures::numerics::culvert::BarrelClient;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{Interval, with_interval_arithmetic};

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Full area of the depth domain per row: the open full area `A'_full` above a floor, the
    /// barrel's full area elsewhere.
    pub(crate) fn open_full_area(&self) -> numr::error::Result<Bounds<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let Some(floor) = self.floor.as_deref() else {
            return Ok(self.full_area.clone());
        };
        let c = &self.client;
        let [lower, upper] = &self.full_area;
        let [open_lower, open_upper] = &floor.full_area;
        Ok([
            c.where_cond(&floor.floored, open_lower, lower)?,
            c.where_cond(&floor.floored, open_upper, upper)?,
        ])
    }

    /// The march view of these rows, or `None` when no row has a floor.
    ///
    /// The log peak of a floored circle or arc row is stored in its frame as
    /// `(5 ln A_f - 2 ln P_n,f) / 3`. In units of `D'` it is that value minus `(8/3) ln D'_f`,
    /// with `D'_f` the frame open rise.
    pub(crate) fn march_view(&self) -> Result<Option<Self>, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let Some(floor) = self.floor.as_deref() else {
            return Ok(None);
        };
        let framed = self.frame.as_ref().map_or(self, |frame| &frame.context);
        let Some(frame_rows) = framed.floor.as_deref() else {
            return Err(StructureError::numerical(
                "the barrel frame lost its floor rows: report this runtime",
            ));
        };
        let width = self.width();
        let all: Vec<i64> = (0..width)
            .map(|row| i64::try_from(row).unwrap_or(i64::MAX))
            .collect();
        let index = Tensor::<R>::from_slice(&all, &[width], &self.device)?;
        let mut view = self.select(&index)?;
        let c = &self.client;
        let floored = &floor.floored;
        let open = c.sub(floored, &floor.boxed)?;
        let ends = |x: &Interval<R>| -> Bounds<R> { [x.lower().clone(), x.upper().clone()] };
        let [perimeter, radius, log_peak] =
            with_interval_arithmetic(c, DType::F64, "culvert outlet open sections", |a| {
                let area = a.bounds(&floor.full_area[0], &floor.full_area[1])?;
                let wall = a.bounds(&floor.full_wall[0], &floor.full_wall[1])?;
                let bed = a.bounds(&floor.weighted_bed[0], &floor.weighted_bed[1])?;
                let perimeter = a.add(&wall, &bed)?;
                let radius = a.div_positive_interval(&area, &perimeter)?;
                let log = a.bounds(&self.log_peak[0], &self.log_peak[1])?;
                let top = a.point(&frame_rows.top)?;
                let ln_top = a.log_positive(&top)?;
                let eight = a.point(&self.constant(8.0)?)?;
                let scaled = a.mul(&eight, &ln_top)?;
                let third = a.div_positive(&scaled, &self.constant(3.0)?)?;
                let unit = a.sub(&log, &third)?;
                Ok([ends(&perimeter), ends(&radius), ends(&unit)])
            })?;
        let pick =
            |mask: &Tensor<R>, on: &Bounds<R>, off: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
                Ok([
                    c.where_cond(mask, &on[0], &off[0])?,
                    c.where_cond(mask, &on[1], &off[1])?,
                ])
            };
        view.rise = c.where_cond(floored, &floor.top, &self.rise)?;
        view.half_rise = c.where_cond(floored, &floor.half_top, &self.half_rise)?;
        view.circular = c.mul(&self.circular, &c.sub(&self.one, floored)?)?;
        view.full_area = pick(floored, &floor.full_area, &self.full_area)?;
        view.full_perimeter = pick(floored, &perimeter, &self.full_perimeter)?;
        view.full_radius = pick(floored, &radius, &self.full_radius)?;
        view.log_peak = pick(&open, &log_peak, &self.log_peak)?;
        Ok(Some(view))
    }
}
