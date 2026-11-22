//! The corner band of a three-radius section, from the first junction to the second.
//!
//! From `psi_1` to `psi_2` with `dy = r_c (cos psi_1 - cos psi_2)`, the trigonometric chain adds:
//!
//! - `A += 2 x_c dy + r_c^2 ((psi_2 - psi_1) - sin psi_2 cos psi_2 + sin psi_1 cos psi_1)`
//! - `P += 2 r_c (psi_2 - psi_1)`
//! - `M += A_1 dy + x_c dy^2 + r_c^3 I_2`, with
//!   `I_2 = [sin t - t cos t - sin^3 t / 3] from psi_1 to psi_2 - (psi_1 - sin psi_1 cos psi_1) dy / r_c`
//!
//! A vertical ellipse has a large negative `x_c`, so `2 x_c dy` cancels against the swept term.
//! The area and moment also come from the corner walls' chords and segments, and each takes the
//! intersection of both enclosures.

use super::cap::{Cap, Shared, Stretch, stretch};
use super::meet;
use super::normals::ThreeRadius;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use numr::dtype::DType;
use numr::runtime::Runtime;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Area, wetted perimeter and moment at the second junction.
pub(super) struct Corner<R: Runtime<DType = DType>> {
    pub(super) area: Interval<R>,
    pub(super) perimeter: Interval<R>,
    pub(super) moment: Interval<R>,
}

/// Chains the corner band onto the bottom cap `bottom`.
///
/// `psi2` is the second junction angle, `dy` the junction depth gap and `cos_sum` encloses
/// `cos psi_1 - cos psi_2`.
pub(super) fn corner<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    t: &ThreeRadius<R>,
    bottom: &Cap<R>,
    psi2: &Interval<R>,
    dy: &Interval<R>,
    cos_sum: &Interval<R>,
) -> ArithmeticResult<Corner<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    // Every bracket term is nonnegative.
    let psi1 = &bottom.angle;
    let sweep = a.sub(psi2, psi1)?;
    let end2 = a.mul(&t.sin2, &t.minus_cos2)?;
    let end1 = a.mul(&t.sin1, &t.cos1)?;
    let bracket = a.add(&sweep, &end2)?;
    let bracket = a.add(&bracket, &end1)?;
    let corner_square = a.mul(&t.corner, &t.corner)?;
    let swept = a.mul(&corner_square, &bracket)?;
    let strip = a.mul(&t.offset, dy)?;
    let strip = a.mul(&shared.two, &strip)?;
    let area = a.add(&bottom.area, &strip)?;
    let area = a.add(&area, &swept)?;
    let arc = a.mul(&t.corner, &sweep)?;
    let arc = a.mul(&shared.two, &arc)?;
    let perimeter = a.add(&bottom.perimeter, &arc)?;

    let at2 = antiderivative(a, shared, psi2, &t.sin2, &t.minus_cos2, true)?;
    let at1 = antiderivative(a, shared, psi1, &t.sin1, &t.cos1, false)?;
    let segment = a.sub(psi1, &end1)?;
    let tail = a.mul(&segment, cos_sum)?;
    let i2 = a.sub(&at2, &at1)?;
    let i2 = a.sub(&i2, &tail)?;
    let corner_cube = a.mul(&corner_square, &t.corner)?;
    let swept_moment = a.mul(&corner_cube, &i2)?;
    let lifted = a.mul(&bottom.area, dy)?;
    let dy_square = a.mul(dy, dy)?;
    let strip_moment = a.mul(&t.offset, &dy_square)?;
    let moment = a.add(&bottom.moment, &lifted)?;
    let moment = a.add(&moment, &strip_moment)?;
    let moment = a.add(&moment, &swept_moment)?;

    // The junction widths come from the arcs centred on the axis.
    let sines = a.add(&t.sin1, &t.sin2)?;
    let lever = a.mul(&t.corner, &sines)?;
    let reach1 = a.mul(&t.bottom, &t.sin1)?;
    let width1 = a.mul(&shared.two, &reach1)?;
    let reach2 = a.mul(&t.top, &t.sin2)?;
    let width2 = a.mul(&shared.two, &reach2)?;
    let chord = stretch(
        a,
        shared,
        &Stretch {
            radius: &t.corner,
            start_sin: &t.sin1,
            start_cos: &t.cos1,
            lever: &lever,
            rise: dy,
            start_width: &width1,
            end_width: &width2,
            start_area: &bottom.area,
            start_moment: &bottom.moment,
        },
    )?;
    Ok(Corner {
        area: meet(a, &area, &chord.area)?,
        perimeter,
        moment: meet(a, &moment, &chord.moment)?,
    })
}

/// `sin t - t cos t - sin^3 t / 3` from the sine, the cosine (negated when `negated`) and `t`.
fn antiderivative<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    angle: &Interval<R>,
    sin: &Interval<R>,
    cos: &Interval<R>,
    negated: bool,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let angle_cos = a.mul(angle, cos)?;
    let square = a.mul(sin, sin)?;
    let cube = a.mul(&square, sin)?;
    let third = a.div_positive(&cube, &shared.three)?;
    let linear = if negated {
        a.add(sin, &angle_cos)?
    } else {
        a.sub(sin, &angle_cos)?
    };
    a.sub(&linear, &third)
}
