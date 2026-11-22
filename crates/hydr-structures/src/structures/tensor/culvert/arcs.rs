//! Per-row tensors of barrels built from tangent circular arcs.
//!
//! Every row holds an arc layout. Circle and box rows hold a harmless admitted barrel's layout,
//! so the arc branches run on every row and `where_cond` selects.

use super::context::Bounds;
use crate::structures::numerics::culvert::arcs::{self, ArcBand, ArcInputs};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Halving steps of the shallow seam search, largest first.
const SEAM_STEPS: [i32; 10] = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1];

/// Arc rows of one barrel context.
pub(crate) struct ArcRows<R: Runtime<DType = DType>> {
    /// 1 on arc rows.
    pub(crate) arced: Tensor<R>,
    /// 1 on flat-floor arch rows.
    pub(crate) floored: Tensor<R>,
    /// Bottom, corner and top bands.
    pub(crate) bands: [ArcBand<R>; 3],
    /// Width `T` at the start of each band: `2 r_b sin psi_k` from the bottom arc, `2 r_t sin psi_2`
    /// from the top arc, and `B` on arch rows.
    pub(crate) start_widths: [Bounds<R>; 3],
    /// A depth inside each band: the operand of rows that another branch serves.
    pub(crate) middles: [Tensor<R>; 3],
    /// Three-radius rows below this depth take the bottom-circle series. It is the largest exact
    /// `D 2^-j` at or below `min(2 r_b / 16, y_1)`, with `y_1` the first junction depth. Zero on
    /// other rows.
    pub(crate) shallow_top: Tensor<R>,
    /// Diameter `2 r_b` of the bottom arc.
    pub(crate) bottom_diameter: Bounds<R>,
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_perimeter: Bounds<R>,
    pub(crate) full_radius: Bounds<R>,
    /// Hydrostatic moment `M(D)` at the crown.
    pub(crate) full_moment: Bounds<R>,
    /// Depth of the widest free surface.
    pub(crate) widest_depth: Bounds<R>,
    /// Upper bound of `T` at every depth: `B`, or `2 r` for an arch whose centre lies above
    /// the floor.
    pub(crate) widest_width: Tensor<R>,
    /// Floor chord `B` on arch rows, zero on other rows.
    pub(crate) floor_width: Tensor<R>,
}

impl<R: Runtime<DType = DType>> ArcRows<R> {
    /// Encloses the arc layout of every row of `inputs`.
    ///
    /// `arced` masks the arc rows. `span` and `rise` hold every row's span and rise.
    pub(crate) fn build(
        a: &mut Arithmetic<'_, R>,
        inputs: &ArcInputs<R>,
        arced: &Tensor<R>,
        span: &Tensor<R>,
        rise: &Tensor<R>,
    ) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let layout = arcs::layout(a, inputs)?;
        let floored = inputs.arch.clone();
        let three = c.sub(arced, &floored)?;
        let zero = Tensor::<R>::zeros(rise.shape(), DType::F64, rise.device())?;

        let [bottom, corner, top] = &layout.bands;
        let middle = |start: &Bounds<R>, end: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
            c.mul_scalar(&c.add(&start[1], end)?, 0.5)
        };
        let middles = [
            middle(&bottom.start_depth, &corner.start_depth[0])?,
            middle(&corner.start_depth, &top.start_depth[0])?,
            middle(&top.start_depth, rise)?,
        ];

        let radius = a.bounds(&bottom.radius[0], &bottom.radius[1])?;
        let two = a.point(&c.add_scalar(&zero, 2.0)?)?;
        let diameter = a.mul(&two, &radius)?;
        // Each start width from an arc centred on the axis, so no centre offset cancels.
        let mut width = |arc: &ArcBand<R>, start: &ArcBand<R>| -> ArithmeticResult<Bounds<R>> {
            let r = a.bounds(&arc.radius[0], &arc.radius[1])?;
            let sin = a.bounds(&start.start_sin[0], &start.start_sin[1])?;
            let twice = a.mul(&two, &r)?;
            let value = a.mul(&twice, &sin)?;
            Ok([
                c.where_cond(&floored, span, value.lower())?,
                c.where_cond(&floored, span, value.upper())?,
            ])
        };
        let start_widths = [
            width(bottom, bottom)?,
            width(bottom, corner)?,
            width(top, top)?,
        ];
        let limit = c.minimum(
            &c.mul_scalar(diameter.lower(), 0.0625)?,
            &corner.start_depth[0],
        )?;
        // Keep D 2^-j above the limit while halving, then halve once more.
        let mut seam = rise.clone();
        for step in SEAM_STEPS {
            let smaller = c.mul_scalar(&seam, 2f64.powi(-step))?;
            seam = c.where_cond(&c.gt(&smaller, &limit)?, &smaller, &seam)?;
        }
        let seam = c.mul_scalar(&seam, 0.5)?;
        let shallow_top = c.where_cond(&three, &seam, &zero)?;

        let arch_radius = a.bounds(&top.radius[0], &top.radius[1])?;
        let arch_width = a.mul(&two, &arch_radius)?;
        let raised = c.gt(&layout.widest_depth[1], &zero)?;
        let reach = c.maximum(span, arch_width.upper())?;
        let arch_widest = c.where_cond(&raised, &reach, span)?;
        let widest_width = c.where_cond(&floored, &arch_widest, span)?;
        let floor_width = c.mul(&floored, span)?;

        let arcs::ArcLayout {
            bands,
            full_area,
            full_perimeter,
            full_radius,
            full_moment,
            widest_depth,
        } = layout;
        Ok(Self {
            arced: arced.clone(),
            floored,
            bands,
            start_widths,
            middles,
            shallow_top,
            bottom_diameter: [diameter.lower().clone(), diameter.upper().clone()],
            full_area,
            full_perimeter,
            full_radius,
            full_moment,
            widest_depth,
            widest_width,
            floor_width,
        })
    }

    /// The same rows in units of the rise `rise`: lengths over `D`, areas over `D^2`, moments
    /// over `D^3`, each bound rounded outward. Angles and masks stay.
    ///
    /// Build `self` from barrels scaled by a power of two to a rise in `[1, 2)`, so no value
    /// underflows and the scaled section is the barrel's own shape.
    pub(crate) fn unit(&self, a: &mut Arithmetic<'_, R>, rise: &Tensor<R>) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let mut scale = |[lower, upper]: &Bounds<R>, power: usize| -> ArithmeticResult<Bounds<R>> {
            let mut value = a.bounds(lower, upper)?;
            for _ in 0..power {
                value = a.div_positive(&value, rise)?;
            }
            Ok([value.lower().clone(), value.upper().clone()])
        };
        let mut band = |b: &ArcBand<R>| -> ArithmeticResult<ArcBand<R>> {
            Ok(ArcBand {
                centre_x: scale(&b.centre_x, 1)?,
                centre_y: scale(&b.centre_y, 1)?,
                radius: scale(&b.radius, 1)?,
                start_angle: b.start_angle.clone(),
                start_sin: b.start_sin.clone(),
                start_cos: b.start_cos.clone(),
                start_depth: scale(&b.start_depth, 1)?,
                start_area: scale(&b.start_area, 2)?,
                start_perimeter: scale(&b.start_perimeter, 1)?,
                start_moment: scale(&b.start_moment, 3)?,
                circle_low: scale(&b.circle_low, 1)?,
                circle_high: scale(&b.circle_high, 1)?,
            })
        };
        let [bottom, corner, top] = &self.bands;
        let bands = [band(bottom)?, band(corner)?, band(top)?];
        let [w0, w1, w2] = &self.start_widths;
        let start_widths = [scale(w0, 1)?, scale(w1, 1)?, scale(w2, 1)?];
        let bottom_diameter = scale(&self.bottom_diameter, 1)?;
        let full_area = scale(&self.full_area, 2)?;
        let full_perimeter = scale(&self.full_perimeter, 1)?;
        let full_radius = scale(&self.full_radius, 1)?;
        let full_moment = scale(&self.full_moment, 3)?;
        let widest_depth = scale(&self.widest_depth, 1)?;
        let widest = [self.widest_width.clone(), self.widest_width.clone()];
        let [_, widest_width] = scale(&widest, 1)?;
        let floor = [self.floor_width.clone(), self.floor_width.clone()];
        let [_, floor_width] = scale(&floor, 1)?;
        let [m0, m1, m2] = &self.middles;
        // D 2^-j over D is the exact 2^-j, and each middle stays inside its band.
        Ok(Self {
            arced: self.arced.clone(),
            floored: self.floored.clone(),
            bands,
            start_widths,
            middles: [c.div(m0, rise)?, c.div(m1, rise)?, c.div(m2, rise)?],
            shallow_top: c.div(&self.shallow_top, rise)?,
            bottom_diameter,
            full_area,
            full_perimeter,
            full_radius,
            full_moment,
            widest_depth,
            widest_width,
            floor_width,
        })
    }

    /// Gathers rows through `take`, in its order.
    pub(crate) fn select(
        &self,
        take: &dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
    ) -> numr::error::Result<Self> {
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
        let [bottom, corner, top] = &self.bands;
        let [w0, w1, w2] = &self.start_widths;
        let [m0, m1, m2] = &self.middles;
        Ok(Self {
            arced: take(&self.arced)?,
            floored: take(&self.floored)?,
            bands: [band(bottom)?, band(corner)?, band(top)?],
            start_widths: [pair(w0)?, pair(w1)?, pair(w2)?],
            middles: [take(m0)?, take(m1)?, take(m2)?],
            shallow_top: take(&self.shallow_top)?,
            bottom_diameter: pair(&self.bottom_diameter)?,
            full_area: pair(&self.full_area)?,
            full_perimeter: pair(&self.full_perimeter)?,
            full_radius: pair(&self.full_radius)?,
            full_moment: pair(&self.full_moment)?,
            widest_depth: pair(&self.widest_depth)?,
            widest_width: take(&self.widest_width)?,
            floor_width: take(&self.floor_width)?,
        })
    }
}
