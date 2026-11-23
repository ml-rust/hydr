//! Construction-time layout of every open section: its bands from the bed, bed width, weights
//! and full state.
//!
//! | Rows | Open section |
//! | --- | --- |
//! | circle | one arc of radius `D/2` from the bed normal `psi_s` to the crown |
//! | tangent-arc barrel | the bed band from `psi_s`, then the bands above it, see `chain` |
//! | box | `B x D'`: `A_full = B D'`, `P_w,full = 2 D' + B`, `M(D') = B D'^2 / 2` |
//!
//! The circle bed point takes `r sin psi_s = sqrt(h D')` from the exact `h` and `D'`. Rows
//! without a floor hold a harmless open section: a bed at `D/4` in a circle, or at a depth
//! inside each band of the arc layout.

use super::bed::{BedPoint, Circle, End, Start, advance, bed_point, nonnegative};
use super::chain::arc_candidate;
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::arcs::cap::Shared;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{ArcRows, Bounds, SeriesTensors};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// `[m]` columns of every row. Masks hold 1 or 0, every length is metres in its frame.
pub(crate) struct FloorInputs<'i, R: Runtime<DType = DType>> {
    /// 1 on rows with a floor.
    pub(crate) floored: &'i Tensor<R>,
    pub(crate) circular: &'i Tensor<R>,
    pub(crate) boxed: &'i Tensor<R>,
    /// Bed depth `h` on floored rows, zero elsewhere.
    pub(crate) bed: &'i Tensor<R>,
    /// Open rise `D' = D - h` on floored rows, `D` elsewhere.
    pub(crate) top: &'i Tensor<R>,
    /// `rho` on floored rows, 1 elsewhere.
    pub(crate) ratio: &'i Tensor<R>,
    pub(crate) rise: &'i Tensor<R>,
    pub(crate) span: &'i Tensor<R>,
    /// Arc layouts of the barrels without their floors, if any row is an arc barrel.
    pub(crate) arcs: Option<&'i ArcRows<R>>,
}

/// Open-section layout of every row, with depths from the bed.
pub(crate) struct FloorLayout<R: Runtime<DType = DType>> {
    /// Bands from the bed. Bands below the bed band repeat it with depth range `[0, 0]`.
    pub(crate) bands: [ArcBand<R>; 3],
    /// Width at the start of each band: `b` at the bed band.
    pub(crate) start_widths: [Bounds<R>; 3],
    /// A positive depth inside each band's circle, the operand of rows another branch serves.
    pub(crate) middles: [Tensor<R>; 3],
    /// Bed width `b`.
    pub(crate) bed_width: Bounds<R>,
    /// `rho^(3/2)`.
    pub(crate) weight: Bounds<R>,
    /// `rho^(3/2) b`.
    pub(crate) weighted_bed: Bounds<R>,
    /// Depth of the widest free surface above the bed, zero when the bed lies above it.
    pub(crate) widest_depth: Bounds<R>,
    /// Upper bound of `T` at every depth.
    pub(crate) widest_width: Tensor<R>,
    pub(crate) full_area: Bounds<R>,
    /// Wall perimeter of the full open section, with the box crown.
    pub(crate) full_wall: Bounds<R>,
    /// Hydrostatic moment `M(D')` at the crown.
    pub(crate) full_moment: Bounds<R>,
}

/// One candidate open section per row, before the shape masks select.
pub(super) struct Candidate<R: Runtime<DType = DType>> {
    pub(super) bands: [ArcBand<R>; 3],
    pub(super) widths: [Bounds<R>; 3],
    pub(super) bed_width: Bounds<R>,
    pub(super) widest_depth: Bounds<R>,
    pub(super) widest_width: Tensor<R>,
    pub(super) full_area: Bounds<R>,
    pub(super) full_wall: Bounds<R>,
    pub(super) full_moment: Bounds<R>,
}

/// Stored endpoints of `x`.
pub(super) fn ends<R: Runtime<DType = DType>>(x: &Interval<R>) -> Bounds<R> {
    [x.lower().clone(), x.upper().clone()]
}

/// `on` where `mask` is 1 and `off` elsewhere.
pub(super) fn pick<R>(
    c: &R::Client,
    mask: &Tensor<R>,
    [on_lower, on_upper]: &Bounds<R>,
    [off_lower, off_upper]: &Bounds<R>,
) -> numr::error::Result<Bounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok([
        c.where_cond(mask, on_lower, off_lower)?,
        c.where_cond(mask, on_upper, off_upper)?,
    ])
}

/// Every field of `on` where `mask` is 1 and of `off` elsewhere.
pub(super) fn pick_band<R>(
    c: &R::Client,
    mask: &Tensor<R>,
    on: &ArcBand<R>,
    off: &ArcBand<R>,
) -> numr::error::Result<ArcBand<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok(ArcBand {
        centre_x: pick(c, mask, &on.centre_x, &off.centre_x)?,
        centre_y: pick(c, mask, &on.centre_y, &off.centre_y)?,
        radius: pick(c, mask, &on.radius, &off.radius)?,
        start_angle: pick(c, mask, &on.start_angle, &off.start_angle)?,
        start_sin: pick(c, mask, &on.start_sin, &off.start_sin)?,
        start_cos: pick(c, mask, &on.start_cos, &off.start_cos)?,
        start_depth: pick(c, mask, &on.start_depth, &off.start_depth)?,
        start_area: pick(c, mask, &on.start_area, &off.start_area)?,
        start_perimeter: pick(c, mask, &on.start_perimeter, &off.start_perimeter)?,
        start_moment: pick(c, mask, &on.start_moment, &off.start_moment)?,
        circle_low: pick(c, mask, &on.circle_low, &off.circle_low)?,
        circle_high: pick(c, mask, &on.circle_high, &off.circle_high)?,
    })
}

/// A copy of `band`.
pub(super) fn copy_band<R: Runtime<DType = DType>>(band: &ArcBand<R>) -> ArcBand<R> {
    ArcBand {
        centre_x: band.centre_x.clone(),
        centre_y: band.centre_y.clone(),
        radius: band.radius.clone(),
        start_angle: band.start_angle.clone(),
        start_sin: band.start_sin.clone(),
        start_cos: band.start_cos.clone(),
        start_depth: band.start_depth.clone(),
        start_area: band.start_area.clone(),
        start_perimeter: band.start_perimeter.clone(),
        start_moment: band.start_moment.clone(),
        circle_low: band.circle_low.clone(),
        circle_high: band.circle_high.clone(),
    }
}

/// Encloses the open-section layout of every row of `inputs`.
pub(crate) fn layout<R>(
    a: &mut Arithmetic<'_, R>,
    inputs: &FloorInputs<'_, R>,
) -> ArithmeticResult<FloorLayout<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = inputs.rise;
    let series = SeriesTensors::constant(like.shape()[0], like.device())?;
    let shared = Shared::new(a, like, &series)?;
    let circle = circle_candidate(a, &shared, inputs)?;
    let open = match inputs.arcs {
        Some(rows) => {
            let arcs = arc_candidate(a, &shared, rows, inputs.floored, inputs.bed)?;
            select(c, inputs.circular, &circle, &arcs)?
        }
        None => circle,
    };

    // A box is B x D' above its bed, with its own floor at h = 0.
    let boxed = inputs.boxed;
    let span = a.point(inputs.span)?;
    let top = a.point(inputs.top)?;
    let box_area = a.mul(&span, &top)?;
    let walls = a.mul(&shared.two, &top)?;
    let box_wall = a.add(&walls, &span)?;
    let lifted = a.mul(&box_area, &top)?;
    let box_moment = a.mul(&lifted, &shared.half)?;
    let span_bounds = [inputs.span.clone(), inputs.span.clone()];
    let bed_width = pick(c, boxed, &span_bounds, &open.bed_width)?;
    let full_area = pick(c, boxed, &ends(&box_area), &open.full_area)?;
    let full_wall = pick(c, boxed, &ends(&box_wall), &open.full_wall)?;
    let full_moment = pick(c, boxed, &ends(&box_moment), &open.full_moment)?;
    let widest_width = c.where_cond(boxed, inputs.span, &open.widest_width)?;

    let ratio = a.point(inputs.ratio)?;
    let root = a.sqrt_nonnegative(&ratio)?;
    let weight = a.mul(&ratio, &root)?;
    let width = a.bounds(&bed_width[0], &bed_width[1])?;
    let weighted_bed = a.mul(&weight, &width)?;

    let middles = middles(c, &open.bands, inputs.top)?;
    Ok(FloorLayout {
        bands: open.bands,
        start_widths: open.widths,
        middles,
        bed_width,
        weight: ends(&weight),
        weighted_bed: ends(&weighted_bed),
        widest_depth: open.widest_depth,
        widest_width,
        full_area,
        full_wall,
        full_moment,
    })
}

/// `circle` on circle rows and `arcs` elsewhere.
fn select<R>(
    c: &R::Client,
    circular: &Tensor<R>,
    circle: &Candidate<R>,
    arcs: &Candidate<R>,
) -> numr::error::Result<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let [c0, c1, c2] = &circle.bands;
    let [a0, a1, a2] = &arcs.bands;
    let [w0, w1, w2] = &circle.widths;
    let [v0, v1, v2] = &arcs.widths;
    Ok(Candidate {
        bands: [
            pick_band(c, circular, c0, a0)?,
            pick_band(c, circular, c1, a1)?,
            pick_band(c, circular, c2, a2)?,
        ],
        widths: [
            pick(c, circular, w0, v0)?,
            pick(c, circular, w1, v1)?,
            pick(c, circular, w2, v2)?,
        ],
        bed_width: pick(c, circular, &circle.bed_width, &arcs.bed_width)?,
        widest_depth: pick(c, circular, &circle.widest_depth, &arcs.widest_depth)?,
        widest_width: c.where_cond(circular, &circle.widest_width, &arcs.widest_width)?,
        full_area: pick(c, circular, &circle.full_area, &arcs.full_area)?,
        full_wall: pick(c, circular, &circle.full_wall, &arcs.full_wall)?,
        full_moment: pick(c, circular, &circle.full_moment, &arcs.full_moment)?,
    })
}

/// A depth inside each band: the middle of its depth range, or `D'/2` where that is not
/// positive, as on bands below the bed band.
fn middles<R>(
    c: &R::Client,
    bands: &[ArcBand<R>; 3],
    top: &Tensor<R>,
) -> numr::error::Result<[Tensor<R>; 3]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let [bottom, corner, crown] = bands;
    let half_top = c.mul_scalar(top, 0.5)?;
    let middle = |start: &Bounds<R>, end: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        let value = c.mul_scalar(&c.add(&start[1], end)?, 0.5)?;
        let zero = c.mul_scalar(&value, 0.0)?;
        c.where_cond(&c.gt(&value, &zero)?, &value, &half_top)
    };
    Ok([
        middle(&bottom.start_depth, &corner.start_depth[0])?,
        middle(&corner.start_depth, &crown.start_depth[0])?,
        middle(&crown.start_depth, top)?,
    ])
}

/// The open section of every row read as a circle of diameter `D` with its bed at `h`, or at
/// `D/4` on rows other than floored circles.
fn circle_candidate<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    inputs: &FloorInputs<'_, R>,
) -> ArithmeticResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let rows = c.mul(inputs.floored, inputs.circular)?;
    let quarter = c.mul_scalar(inputs.rise, 0.25)?;
    let three_quarters = c.mul_scalar(inputs.rise, 0.75)?;
    let depth = c.where_cond(&rows, inputs.bed, &quarter)?;
    let open = c.where_cond(&rows, inputs.top, &three_quarters)?;
    let half_rise = c.mul_scalar(inputs.rise, 0.5)?;
    let h = a.point(&depth)?;
    let d_open = a.point(&open)?;
    let radius = a.point(&half_rise)?;
    let zero = a.point(&shared.zero)?;
    let centre_y = a.sub(&radius, &h)?;
    let low = a.sub(&zero, &h)?;
    let diameter = a.point(inputs.rise)?;
    // Low and high from the invert, so the bed point reads h and D' exactly.
    let point = bed_point(
        a,
        shared,
        &Circle {
            centre_x: &zero,
            centre_y: &radius,
            radius: &radius,
            low: &zero,
            high: &diameter,
        },
        &h,
    )?;
    let BedPoint {
        sin,
        cos,
        angle,
        reach,
        width,
    } = point;
    let reached = advance(
        a,
        shared,
        &radius,
        &Start {
            sin: &sin,
            cos: &cos,
            reach: &reach,
            width: &width,
            area: &zero,
            wall: &zero,
            moment: &zero,
        },
        &End {
            sin: &zero,
            width: &zero,
            rise: &d_open,
        },
    )?;
    let band = ArcBand {
        centre_x: ends(&zero),
        centre_y: ends(&centre_y),
        radius: ends(&radius),
        start_angle: ends(&angle),
        start_sin: ends(&sin),
        start_cos: ends(&cos),
        start_depth: ends(&zero),
        start_area: ends(&zero),
        start_perimeter: ends(&zero),
        start_moment: ends(&zero),
        circle_low: ends(&low),
        circle_high: ends(&d_open),
    };
    let widest = nonnegative(a, shared, &centre_y)?;
    // A bed above the centre is the widest surface.
    let below_centre = c.le(&depth, &half_rise)?;
    let widest_width = c.where_cond(&below_centre, inputs.rise, width.upper())?;
    let bed_width = ends(&width);
    Ok(Candidate {
        bands: [copy_band(&band), copy_band(&band), band],
        widths: [bed_width.clone(), bed_width.clone(), bed_width.clone()],
        bed_width,
        widest_depth: ends(&widest),
        widest_width,
        full_area: ends(&reached.area),
        full_wall: ends(&reached.wall),
        full_moment: ends(&reached.moment),
    })
}
