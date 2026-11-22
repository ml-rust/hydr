//! Circle series for `0 < y < D/16`, where the closed forms cancel.
//!
//! With `s = y/D` and `r = sqrt(s)`:
//!
//! - `phi = 2 asin(r) = 2 r sum a_k s^k`, all `a_k > 0`.
//! - `A / D^2 = r s sum c_k s^k`, `c_0 = 4/3`, every later `c_k < 0`.
//! - `M / D^3 = r s^2 sum c_k s^k / (k + 5/2)`.
//! - `P = D phi` and `R = A/P = D s (sum c_k s^k) / (2 sum a_k s^k)`.
//!
//! Each sum keeps `SERIES_TERMS` terms. The coefficient magnitudes decrease, so the remainder
//! lies between 0 and the first omitted term times `s^m / (1 - s)`, with the sign of that term.

use super::{Quantities, top_width};
use crate::structures::numerics::culvert::constants::SERIES_TERMS;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{Bounds, SeriesTensors, TensorBarrelContext};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the free-surface geometry at depths `0 <= depth < D` of circle rows.
pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let rise = &barrel.rise;
    let y = a.point(depth)?;
    let d = a.point(rise)?;
    let top_width = top_width(a, barrel, &y, &d)?;
    let s = a.div_positive(&y, rise)?;
    from_fraction(a, barrel, top_width, &d, &s)
}

/// Encloses the circle geometry of diameter `d` at depth fraction `s = y/d < 1`.
///
/// `top_width` is the caller's enclosure of `T`.
pub(super) fn from_fraction<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    top_width: Interval<R>,
    d: &Interval<R>,
    s: &Interval<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let r = a.sqrt_nonnegative(s)?;
    let [asin, area_sum, moment_sum] = sums(a, &barrel.series, s, &barrel.one, &barrel.zero)?;

    let two = a.point(&barrel.constant(2.0)?)?;
    let half_angle = a.mul(&r, &asin)?;
    let angle = a.mul(&two, &half_angle)?;
    let perimeter = a.mul(d, &angle)?;

    let square = a.mul(d, d)?;
    let rs = a.mul(&r, s)?;
    let area_unit = a.mul(&rs, &area_sum)?;
    let area = a.mul(&square, &area_unit)?;

    let cubed = a.mul(&square, d)?;
    let rs2 = a.mul(&rs, s)?;
    let moment_unit = a.mul(&rs2, &moment_sum)?;
    let moment = a.mul(&cubed, &moment_unit)?;

    let ds = a.mul(d, s)?;
    let numerator = a.mul(&ds, &area_sum)?;
    let denominator = a.mul(&two, &asin)?;
    let radius = a.div_positive_interval(&numerator, &denominator)?;

    Ok(Quantities {
        area,
        top_width,
        perimeter,
        radius,
        moment,
    })
}

/// Encloses `[sum a_k s^k, sum c_k s^k, sum c_k s^k / (k + 5/2)]` for `0 <= s < 1`.
///
/// `one` and `zero` are `[m]` tensors of ones and zeros.
pub(crate) fn sums<R>(
    a: &mut Arithmetic<'_, R>,
    series: &SeriesTensors<R>,
    s: &Interval<R>,
    one: &Tensor<R>,
    zero: &Tensor<R>,
) -> ArithmeticResult<[Interval<R>; 3]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let one = a.point(one)?;
    let rest = a.sub(&one, s)?;
    // weight = s^m / (1 - s) with m = 14: s^14 = s^8 s^4 s^2.
    let s2 = a.mul(s, s)?;
    let s4 = a.mul(&s2, &s2)?;
    let s8 = a.mul(&s4, &s4)?;
    let s12 = a.mul(&s8, &s4)?;
    let s14 = a.mul(&s12, &s2)?;
    let weight = a.div_positive_interval(&s14, &rest)?;
    Ok([
        sum(a, zero, &series.asin, s, &weight, Tail::Positive)?,
        sum(a, zero, &series.area, s, &weight, Tail::Negative)?,
        sum(a, zero, &series.moment, s, &weight, Tail::Negative)?,
    ])
}

/// Sign of every coefficient past the leading one.
#[derive(Clone, Copy)]
enum Tail {
    Positive,
    Negative,
}

/// Encloses `sum_k coefficient_k s^k` over all `k`, with the remainder past `SERIES_TERMS`.
fn sum<R>(
    a: &mut Arithmetic<'_, R>,
    zero: &Tensor<R>,
    coefficients: &[Bounds<R>],
    s: &Interval<R>,
    weight: &Interval<R>,
    tail: Tail,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let coefficient = |a: &mut Arithmetic<'_, R>, k: usize| {
        let [lower, upper] = &coefficients[k];
        a.bounds(lower, upper)
    };
    let mut horner = coefficient(a, SERIES_TERMS - 1)?;
    for k in (0..SERIES_TERMS - 1).rev() {
        let scaled = a.mul(&horner, s)?;
        let next = coefficient(a, k)?;
        horner = a.add(&scaled, &next)?;
    }
    let omitted = coefficient(a, SERIES_TERMS)?;
    let bound = a.mul(&omitted, weight)?;
    let remainder = match tail {
        Tail::Positive => a.bounds(zero, bound.upper())?,
        Tail::Negative => a.bounds(bound.lower(), zero)?,
    };
    a.add(&horner, &remainder)
}
