//! The bed point on one arc, and one stretch of two mirrored walls up that arc.

use crate::structures::numerics::culvert::arcs::cap::{Shared, Stretch, stretch};
use crate::structures::numerics::culvert::arcs::meet;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose, clip};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// One circle of a band: centre `(x, y)`, radius `r` and its lowest and highest depth.
pub(super) struct Circle<'c, R: Runtime<DType = DType>> {
    pub(super) centre_x: &'c Interval<R>,
    pub(super) centre_y: &'c Interval<R>,
    pub(super) radius: &'c Interval<R>,
    pub(super) low: &'c Interval<R>,
    pub(super) high: &'c Interval<R>,
}

/// Normal, half width and width of a wall point at the bed.
pub(super) struct BedPoint<R: Runtime<DType = DType>> {
    pub(super) sin: Interval<R>,
    pub(super) cos: Interval<R>,
    pub(super) angle: Interval<R>,
    /// `r sin psi_s`, the reach from the centre line of the arc.
    pub(super) reach: Interval<R>,
    /// `b = 2 (x + r sin psi_s)`.
    pub(super) width: Interval<R>,
}

/// The part of `x` at or above zero, for a value known to be nonnegative.
pub(super) fn nonnegative<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    x: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.maximum(x.lower(), &shared.zero)?;
    let upper = c.maximum(x.upper(), &shared.zero)?;
    a.bounds(&lower, &upper)
}

/// Encloses the wall point of `circle` at depth `bed` above the invert.
///
/// `r sin psi_s = sqrt((h - (y - r)) ((y + r) - h))` keeps both factors as differences of the
/// bed from an end of the circle, so neither cancels at a thin bed or a bed under the crown.
pub(super) fn bed_point<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    circle: &Circle<'_, R>,
    bed: &Interval<R>,
) -> ArithmeticResult<BedPoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let above = a.sub(bed, circle.low)?;
    let above = nonnegative(a, shared, &above)?;
    let below = a.sub(circle.high, bed)?;
    let below = nonnegative(a, shared, &below)?;
    let product = a.mul(&above, &below)?;
    let reach = a.sqrt_nonnegative(&product)?;
    let sin = a.div_positive_interval(&reach, circle.radius)?;
    let sin = clip(a, &sin, 0.0, 1.0)?;
    let lift = a.sub(circle.centre_y, bed)?;
    let cos = a.div_positive_interval(&lift, circle.radius)?;
    let cos = clip(a, &cos, -1.0, 1.0)?;
    let angle = normal_angle(a, shared, &sin, &cos)?;
    let half = a.add(circle.centre_x, &reach)?;
    let width = a.mul(&shared.two, &half)?;
    Ok(BedPoint {
        sin,
        cos,
        angle,
        reach,
        width,
    })
}

/// Encloses the normal angle in `[0, pi]` from its sine and cosine.
fn normal_angle<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    sin: &Interval<R>,
    cos: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let by_cos = a.acos(cos)?;
    let complement = a.acos(sin)?;
    // psi = pi/2 - acos(sin psi) below pi/2 and pi/2 + acos(sin psi) above it.
    let acute = a.sub(&shared.half_pi, &complement)?;
    let obtuse = a.add(&shared.half_pi, &complement)?;
    let whole = a.bounds(&shared.zero, shared.pi.upper())?;
    let rising = c.ge(cos.lower(), &shared.zero)?;
    let falling = c.le(cos.upper(), &shared.zero)?;
    let by_sin = choose(a, &falling, &obtuse, &whole)?;
    let by_sin = choose(a, &rising, &acute, &by_sin)?;
    meet(a, &by_cos, &by_sin)
}

/// Start of a stretch: wall normal, width and the values there.
pub(super) struct Start<'s, R: Runtime<DType = DType>> {
    pub(super) sin: &'s Interval<R>,
    pub(super) cos: &'s Interval<R>,
    /// `r sin a`.
    pub(super) reach: &'s Interval<R>,
    pub(super) width: &'s Interval<R>,
    pub(super) area: &'s Interval<R>,
    pub(super) wall: &'s Interval<R>,
    pub(super) moment: &'s Interval<R>,
}

/// End of a stretch: wall normal sine, width and the rise from the start.
pub(super) struct End<'e, R: Runtime<DType = DType>> {
    pub(super) sin: &'e Interval<R>,
    pub(super) width: &'e Interval<R>,
    pub(super) rise: &'e Interval<R>,
}

/// Area, wall perimeter and moment about the end level, at the end of a stretch.
pub(super) struct Reached<R: Runtime<DType = DType>> {
    pub(super) area: Interval<R>,
    pub(super) wall: Interval<R>,
    pub(super) moment: Interval<R>,
}

/// Encloses two mirrored walls of `radius` from `start` up to `end`.
pub(super) fn advance<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    radius: &Interval<R>,
    start: &Start<'_, R>,
    end: &End<'_, R>,
) -> ArithmeticResult<Reached<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let end_reach = a.mul(radius, end.sin)?;
    let lever = a.add(start.reach, &end_reach)?;
    let walls = stretch(
        a,
        shared,
        &Stretch {
            radius,
            start_sin: start.sin,
            start_cos: start.cos,
            lever: &lever,
            rise: end.rise,
            start_width: start.width,
            end_width: end.width,
            start_area: start.area,
            start_moment: start.moment,
        },
    )?;
    let arcs = a.mul(&shared.two, &walls.arc)?;
    Ok(Reached {
        area: walls.area,
        wall: a.add(start.wall, &arcs)?,
        moment: walls.moment,
    })
}

/// The hull of `value` and `other` where `mask` is 1, and `value` elsewhere.
pub(super) fn hull_where<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    value: &Interval<R>,
    other: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.minimum(value.lower(), other.lower())?;
    let upper = c.maximum(value.upper(), other.upper())?;
    let wide = a.bounds(&lower, &upper)?;
    choose(a, mask, &wide, value)
}
