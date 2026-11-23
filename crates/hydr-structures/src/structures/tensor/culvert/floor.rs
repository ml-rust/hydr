//! Per-row tensors of the open section above each declared floor.
//!
//! Every row holds an open-section layout. Rows without a floor hold a harmless one, so the
//! floor branches run on every row and `where_cond` selects.

use super::context::{Bounds, TensorBarrelContext};
use super::frame::BarrelFrame;
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::floor::{FloorInputs, FloorLayout, layout};
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

/// Open sections of one barrel context, with depths from each bed.
pub(crate) struct FloorRows<R: Runtime<DType = DType>> {
    /// 1 on rows with a floor.
    pub(crate) floored: Tensor<R>,
    /// 1 on box rows with a floor.
    pub(crate) boxed: Tensor<R>,
    /// Bed depth `h` above the invert, zero on rows without a floor. Exact.
    pub(crate) bed: Tensor<R>,
    /// Open rise `D' = D - h`, `D` on rows without a floor. Exact.
    pub(crate) top: Tensor<R>,
    /// `D' / 2`, exact. Harmless box operand on other lanes.
    pub(crate) half_top: Tensor<R>,
    pub(crate) bands: [ArcBand<R>; 3],
    pub(crate) start_widths: [Bounds<R>; 3],
    pub(crate) middles: [Tensor<R>; 3],
    pub(crate) bed_width: Bounds<R>,
    /// `rho^(3/2)`.
    pub(crate) weight: Bounds<R>,
    /// `rho^(3/2) b`.
    pub(crate) weighted_bed: Bounds<R>,
    pub(crate) widest_depth: Bounds<R>,
    pub(crate) widest_width: Tensor<R>,
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_wall: Bounds<R>,
    pub(crate) full_moment: Bounds<R>,
}

/// A row gather applied to one tensor.
type Take<'t, R> = dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>> + 't;

impl<R: Runtime<DType = DType>> FloorRows<R> {
    /// The open sections of the floored rows of `barrels`, or `None` when no row has a floor.
    ///
    /// `context` holds the rows of `barrels` without their floors. With a frame, the frame's
    /// open sections times exact powers of two give the metre rows.
    pub(crate) fn new(
        context: &TensorBarrelContext<R>,
        barrels: &[CulvertBarrel],
    ) -> Result<Option<Self>, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        if barrels.iter().all(|barrel| barrel.floor().is_none()) {
            return Ok(None);
        }
        let client = &context.client;
        if let Some(frame) = &context.frame {
            let Some(rows) = &frame.context.floor else {
                return Err(StructureError::numerical(
                    "the barrel frame holds no floor rows: report this runtime",
                ));
            };
            return Ok(Some(rows.scaled(client, frame)?));
        }
        let m = barrels.len();
        let column = |f: &dyn Fn(&CulvertBarrel) -> f64| -> numr::error::Result<Tensor<R>> {
            let values: Vec<f64> = barrels.iter().map(f).collect();
            Tensor::<R>::from_slice(&values, &[m], &context.device)
        };
        let flag = |value: bool| f64::from(u8::from(value));
        let floored = column(&|b| flag(b.floor().is_some()))?;
        let boxed = column(&|b| flag(b.floor().is_some() && b.shape() == CulvertBarrelShape::Box))?;
        let bed = column(&|b| b.floor().map_or(0.0, |f| f.sediment_depth_m))?;
        let top = column(&|b| b.open_rise_m())?;
        let half_top = column(&|b| b.open_rise_m() * 0.5)?;
        let ratio = column(&|b| b.floor().map_or(1.0, |f| f.roughness_ratio))?;
        let inputs = FloorInputs {
            floored: &floored,
            circular: &context.circular,
            boxed: &boxed,
            bed: &bed,
            top: &top,
            ratio: &ratio,
            rise: &context.rise,
            span: &context.span,
            arcs: context.arcs.as_ref(),
        };
        let layout = with_interval_arithmetic(client, DType::F64, "culvert floor layout", |a| {
            layout(a, &inputs)
        })?;
        Ok(Some(Self::from_layout(
            [floored, boxed, bed, top, half_top],
            layout,
        )))
    }

    fn from_layout(
        [floored, boxed, bed, top, half_top]: [Tensor<R>; 5],
        layout: FloorLayout<R>,
    ) -> Self {
        let FloorLayout {
            bands,
            start_widths,
            middles,
            bed_width,
            weight,
            weighted_bed,
            widest_depth,
            widest_width,
            full_area,
            full_wall,
            full_moment,
        } = layout;
        Self {
            floored,
            boxed,
            bed,
            top,
            half_top,
            bands,
            start_widths,
            middles,
            bed_width,
            weight,
            weighted_bed,
            widest_depth,
            widest_width,
            full_area,
            full_wall,
            full_moment,
        }
    }

    /// Frame rows back in metres: lengths times `2^e`, areas times `2^2e` and moments times
    /// `2^3e`, each bound exact unless it leaves the normal F64 range, then rounded outward.
    /// Angles, weights and masks stay.
    ///
    /// The upper bound stands for `widest_width`.
    fn scaled(&self, c: &R::Client, frame: &BarrelFrame<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let step = |[lower, upper]: &Bounds<R>, power: usize| frame.times([lower, upper], power);
        let exact = |value: &Tensor<R>| c.mul(value, &frame.scale);
        let band = |b: &ArcBand<R>| -> numr::error::Result<ArcBand<R>> {
            Ok(ArcBand {
                centre_x: step(&b.centre_x, 1)?,
                centre_y: step(&b.centre_y, 1)?,
                radius: step(&b.radius, 1)?,
                start_angle: b.start_angle.clone(),
                start_sin: b.start_sin.clone(),
                start_cos: b.start_cos.clone(),
                start_depth: step(&b.start_depth, 1)?,
                start_area: step(&b.start_area, 2)?,
                start_perimeter: step(&b.start_perimeter, 1)?,
                start_moment: step(&b.start_moment, 3)?,
                circle_low: step(&b.circle_low, 1)?,
                circle_high: step(&b.circle_high, 1)?,
            })
        };
        let [b0, b1, b2] = &self.bands;
        let bands = [band(b0)?, band(b1)?, band(b2)?];
        let [w0, w1, w2] = &self.start_widths;
        let start_widths = [step(w0, 1)?, step(w1, 1)?, step(w2, 1)?];
        let [m0, m1, m2] = &self.middles;
        let widest = [self.widest_width.clone(), self.widest_width.clone()];
        let [_, widest_width] = step(&widest, 1)?;
        Ok(Self {
            floored: self.floored.clone(),
            boxed: self.boxed.clone(),
            bed: exact(&self.bed)?,
            top: exact(&self.top)?,
            half_top: exact(&self.half_top)?,
            bands,
            start_widths,
            middles: [exact(m0)?, exact(m1)?, exact(m2)?],
            bed_width: step(&self.bed_width, 1)?,
            weight: self.weight.clone(),
            weighted_bed: step(&self.weighted_bed, 1)?,
            widest_depth: step(&self.widest_depth, 1)?,
            widest_width,
            full_area: step(&self.full_area, 2)?,
            full_wall: step(&self.full_wall, 1)?,
            full_moment: step(&self.full_moment, 3)?,
        })
    }

    /// Gathers rows through `take`, in its order.
    pub(crate) fn select(&self, take: &Take<'_, R>) -> numr::error::Result<Self> {
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        let band = |b: &ArcBand<R>| -> numr::error::Result<ArcBand<R>> {
            Ok(ArcBand {
                centre_x: pair(&b.centre_x)?,
                centre_y: pair(&b.centre_y)?,
                radius: pair(&b.radius)?,
                start_angle: pair(&b.start_angle)?,
                start_sin: pair(&b.start_sin)?,
                start_cos: pair(&b.start_cos)?,
                start_depth: pair(&b.start_depth)?,
                start_area: pair(&b.start_area)?,
                start_perimeter: pair(&b.start_perimeter)?,
                start_moment: pair(&b.start_moment)?,
                circle_low: pair(&b.circle_low)?,
                circle_high: pair(&b.circle_high)?,
            })
        };
        let [b0, b1, b2] = &self.bands;
        let [w0, w1, w2] = &self.start_widths;
        let [m0, m1, m2] = &self.middles;
        Ok(Self {
            floored: take(&self.floored)?,
            boxed: take(&self.boxed)?,
            bed: take(&self.bed)?,
            top: take(&self.top)?,
            half_top: take(&self.half_top)?,
            bands: [band(b0)?, band(b1)?, band(b2)?],
            start_widths: [pair(w0)?, pair(w1)?, pair(w2)?],
            middles: [take(m0)?, take(m1)?, take(m2)?],
            bed_width: pair(&self.bed_width)?,
            weight: pair(&self.weight)?,
            weighted_bed: pair(&self.weighted_bed)?,
            widest_depth: pair(&self.widest_depth)?,
            widest_width: take(&self.widest_width)?,
            full_area: pair(&self.full_area)?,
            full_wall: pair(&self.full_wall)?,
            full_moment: pair(&self.full_moment)?,
        })
    }
}
