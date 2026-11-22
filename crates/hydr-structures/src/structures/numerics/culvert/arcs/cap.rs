//! Circle cap of radius `r` cut by a chord at cap angle `theta`, seen from its tangent point.
//!
//! The cap rises `h = r (1 - cos theta)` from the tangent point. With `s = h / (2 r)`:
//!
//! | Quantity | Trigonometric form | Series form in `s` |
//! | --- | --- | --- |
//! | `theta` | `acos(cos theta)`, and `pi/2 - acos(sin theta)` when `cos theta >= 0` | `2 asin(sqrt s)` |
//! | Area | `r^2 (theta - sin theta cos theta)` | `(2r)^2 sqrt(s) s sum c_k s^k` |
//! | Arc length | `2 r theta` | `2r theta` from the series angle |
//! | Moment about the chord | `r^3 (sin theta - theta cos theta - sin^3 theta / 3)` | `(2r)^3 sqrt(s) s^2 sum c_k s^k / (k + 5/2)` |
//!
//! The trigonometric forms cancel for a small cap and the series converges slowly for a large
//! one. Each quantity takes the intersection of both enclosures.

use super::meet;
use crate::structures::numerics::culvert::section::sums;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose, clip};
use crate::structures::tensor::culvert::SeriesTensors;
use numr::dtype::DType;
use numr::ops::CompareOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Row constants every cap reads.
pub(super) struct Shared<R: Runtime<DType = DType>> {
    pub(super) zero: Tensor<R>,
    pub(super) one: Tensor<R>,
    pub(super) half: Interval<R>,
    pub(super) two: Interval<R>,
    pub(super) three: Tensor<R>,
    pub(super) pi: Interval<R>,
    pub(super) half_pi: Interval<R>,
    pub(super) series: SeriesTensors<R>,
}

/// Cap angle, height, area, arc length and moment about the chord.
pub(super) struct Cap<R: Runtime<DType = DType>> {
    pub(super) angle: Interval<R>,
    pub(super) height: Interval<R>,
    pub(super) area: Interval<R>,
    pub(super) perimeter: Interval<R>,
    pub(super) moment: Interval<R>,
}

/// Encloses the cap of `radius` with cap-angle sine `sin` and cosine `cos`, `1 + cos > 0`.
pub(super) fn cap<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<R>,
    radius: &Interval<R>,
    sin: &Interval<R>,
    cos: &Interval<R>,
) -> ArithmeticResult<Cap<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let sin = clip(a, sin, 0.0, 1.0)?;
    let cos = clip(a, cos, -1.0, 1.0)?;

    // s = (1 - cos) / 2 = sin^2 / (2 (1 + cos)).
    let one = a.point(&shared.one)?;
    let rest = a.sub(&one, &cos)?;
    let by_cos = a.mul(&rest, &shared.half)?;
    let plus = a.add(&one, &cos)?;
    let twice_plus = a.mul(&shared.two, &plus)?;
    let square = a.mul(&sin, &sin)?;
    let by_sin = a.div_positive_interval(&square, &twice_plus)?;
    let s = meet(a, &by_cos, &by_sin)?;
    let s = clip(a, &s, 0.0, 1.0)?;

    let [asin, area_sum, moment_sum] = sums(a, &shared.series, &s, &shared.one, &shared.zero)?;
    let root = a.sqrt_nonnegative(&s)?;
    let half_angle = a.mul(&root, &asin)?;
    let series_angle = a.mul(&shared.two, &half_angle)?;

    let by_acos = a.acos(&cos)?;
    let complement = a.acos(&sin)?;
    let by_asin = a.sub(&shared.half_pi, &complement)?;
    // pi/2 - acos(sin theta) is theta only for theta <= pi/2.
    let acute = c.ge(cos.lower(), &shared.zero)?;
    let whole = a.bounds(&shared.zero, shared.pi.upper())?;
    let by_asin = choose(a, &acute, &by_asin, &whole)?;
    let angle = meet(a, &by_acos, &by_asin)?;
    let angle = meet(a, &angle, &series_angle)?;

    let diameter = a.mul(&shared.two, radius)?;
    let height = a.mul(&diameter, &s)?;
    let radius_square = a.mul(radius, radius)?;
    let radius_cube = a.mul(&radius_square, radius)?;

    let product = a.mul(&sin, &cos)?;
    let segment = a.sub(&angle, &product)?;
    let trig_area = a.mul(&radius_square, &segment)?;
    let angle_cos = a.mul(&angle, &cos)?;
    let cube = a.mul(&square, &sin)?;
    let third = a.div_positive(&cube, &shared.three)?;
    let partial = a.sub(&sin, &angle_cos)?;
    let bracket = a.sub(&partial, &third)?;
    let trig_moment = a.mul(&radius_cube, &bracket)?;

    let diameter_square = a.mul(&diameter, &diameter)?;
    let rs = a.mul(&root, &s)?;
    let area_unit = a.mul(&rs, &area_sum)?;
    let series_area = a.mul(&diameter_square, &area_unit)?;
    let diameter_cube = a.mul(&diameter_square, &diameter)?;
    let rs2 = a.mul(&rs, &s)?;
    let moment_unit = a.mul(&rs2, &moment_sum)?;
    let series_moment = a.mul(&diameter_cube, &moment_unit)?;

    let perimeter = a.mul(&diameter, &angle)?;
    let area = meet(a, &trig_area, &series_area)?;
    let moment = meet(a, &trig_moment, &series_moment)?;
    Ok(Cap {
        angle,
        height,
        area,
        perimeter,
        moment,
    })
}
