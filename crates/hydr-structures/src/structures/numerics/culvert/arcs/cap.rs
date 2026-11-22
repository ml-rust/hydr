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
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::section::sums;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose, clip};
use crate::structures::tensor::culvert::SeriesTensors;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Row constants every cap reads.
pub(crate) struct Shared<'s, R: Runtime<DType = DType>> {
    pub(crate) zero: Tensor<R>,
    pub(crate) one: Tensor<R>,
    pub(crate) half: Interval<R>,
    pub(crate) two: Interval<R>,
    pub(crate) three: Tensor<R>,
    pub(crate) pi: Interval<R>,
    pub(crate) half_pi: Interval<R>,
    pub(crate) series: &'s SeriesTensors<R>,
}

impl<'s, R: Runtime<DType = DType>> Shared<'s, R> {
    /// The constants on every row of `like`, with the series coefficients `series`.
    pub(crate) fn new(
        a: &mut Arithmetic<'_, R>,
        like: &Tensor<R>,
        series: &'s SeriesTensors<R>,
    ) -> ArithmeticResult<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let fill =
            |value: f64| Tensor::<R>::full_scalar(like.shape(), DType::F64, value, like.device());
        let pi = a.bounds(&fill(f64::from_bits(PI[0]))?, &fill(f64::from_bits(PI[1]))?)?;
        let half = a.point(&fill(0.5)?)?;
        let half_pi = a.mul(&pi, &half)?;
        Ok(Self {
            zero: fill(0.0)?,
            one: fill(1.0)?,
            two: a.point(&fill(2.0)?)?,
            three: fill(3.0)?,
            half,
            pi,
            half_pi,
            series,
        })
    }
}

/// One arc from its start normal `a` to a depth `dy` above its start, with the values there.
pub(crate) struct Stretch<'i, R: Runtime<DType = DType>> {
    pub(crate) radius: &'i Interval<R>,
    pub(crate) start_sin: &'i Interval<R>,
    pub(crate) start_cos: &'i Interval<R>,
    /// `r (sin a + sin psi)`, positive.
    pub(crate) lever: &'i Interval<R>,
    /// `dy`, nonnegative.
    pub(crate) rise: &'i Interval<R>,
    pub(crate) start_width: &'i Interval<R>,
    pub(crate) end_width: &'i Interval<R>,
    pub(crate) start_area: &'i Interval<R>,
    pub(crate) start_moment: &'i Interval<R>,
}

/// Area, moment and the arc length of one wall, at the end of a [`Stretch`].
pub(crate) struct Stretched<R: Runtime<DType = DType>> {
    pub(crate) area: Interval<R>,
    pub(crate) moment: Interval<R>,
    pub(crate) arc: Interval<R>,
}

/// Encloses a stretch of two mirrored walls from each wall's chord and circular segment.
///
/// The wall from the start to the end is the chord between its ends plus a segment of angle
/// `t = psi - a`, with `tan(t/2) = dy / (r (sin a + sin psi))`. The segment is the cap of half
/// angle `t/2`, with area `A_s` and moment `M_s` about its chord. With the end widths `T_0`, `T`:
///
/// - `A = A_0 + dy (T_0 + T) / 2 + 2 A_s`
/// - `M = M_0 + A_0 dy + dy^2 (2 T_0 + T) / 6 + A_s dy + 2 cos(a + t/2) M_s`
/// - arc length `r t`
///
/// Every term of `A` is nonnegative and no term cancels against a centre offset.
pub(crate) fn stretch<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    s: &Stretch<'_, R>,
) -> ArithmeticResult<Stretched<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let one = a.point(&shared.one)?;
    let tan = a.div_positive_interval(s.rise, s.lever)?;
    let square = a.mul(&tan, &tan)?;
    let sum = a.add(&one, &square)?;
    let secant = a.sqrt_nonnegative(&sum)?;
    let sin = a.div_positive_interval(&tan, &secant)?;
    let cos = a.div_positive_interval(&one, &secant)?;
    let segment = cap(a, shared, s.radius, &sin, &cos)?;

    let ends = a.add(s.start_width, s.end_width)?;
    let trapezoid = a.mul(s.rise, &ends)?;
    let trapezoid = a.mul(&trapezoid, &shared.half)?;
    let caps = a.mul(&shared.two, &segment.area)?;
    let area = a.add(s.start_area, &trapezoid)?;
    let area = a.add(&area, &caps)?;

    // cos(a + t/2) turns the segment's moment about its chord into one about the surface.
    let first = a.mul(s.start_cos, &cos)?;
    let second = a.mul(s.start_sin, &sin)?;
    let tilt = a.sub(&first, &second)?;
    let lifted = a.mul(s.start_area, s.rise)?;
    let doubled = a.mul(&shared.two, s.start_width)?;
    let weights = a.add(&doubled, s.end_width)?;
    let rise2 = a.mul(s.rise, s.rise)?;
    let strip = a.mul(&rise2, &weights)?;
    let six = a.client().add(&shared.three, &shared.three)?;
    let strip = a.div_positive(&strip, &six)?;
    let shifted = a.mul(&segment.area, s.rise)?;
    let turned = a.mul(&tilt, &segment.moment)?;
    let turned = a.mul(&shared.two, &turned)?;
    let moment = a.add(s.start_moment, &lifted)?;
    let moment = a.add(&moment, &strip)?;
    let moment = a.add(&moment, &shifted)?;
    let moment = a.add(&moment, &turned)?;
    Ok(Stretched {
        area,
        moment,
        arc: segment.perimeter,
    })
}

/// Cap angle, height, area, arc length and moment about the chord.
pub(crate) struct Cap<R: Runtime<DType = DType>> {
    pub(crate) angle: Interval<R>,
    pub(crate) height: Interval<R>,
    pub(crate) area: Interval<R>,
    pub(crate) perimeter: Interval<R>,
    pub(crate) moment: Interval<R>,
}

/// Encloses the cap of `radius` with cap-angle sine `sin` and cosine `cos`, `1 + cos > 0`.
pub(crate) fn cap<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
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

    let [asin, area_sum, moment_sum] = sums(a, shared.series, &s, &shared.one, &shared.zero)?;
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
