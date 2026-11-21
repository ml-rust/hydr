//! Interval sine and cosine from float series at points, with running error bounds.
//!
//! Each point `e` takes `x = fl(e - pi/2)` and plain Horner steps in `z = fl(x^2)`, so
//! `sin e = cos x` and `cos e = -sin x`. Each value carries its own error bound, with `u = 2^-53`:
//!
//! - Horner rounding: the running bound `u E`, `E_i = |z| E_{i+1} + |z| |y_{i+1}| + |y_i|`.
//! - Coefficients: `1/n!` rounds once for `n <= 18`, so they add `u` times the series of
//!   `|c_i| z^i`, at most `cosh 1.61 < 2.6`. Higher coefficients add below `1e-27`.
//! - Rounding `z`: the series slope in `z` stays below `1/2`, so `0.51 u z`.
//! - Rounding `x`: `1.01 u |x|` plus `6.2e-17` from `fl(pi/2)`, at unit slope.
//! - The series tails for `|x| <= 1.61`: below `2e-24`.
//!
//! Each float sum is scaled by `1.02` before it widens a value. That covers its own rounding.
//!
//! [`sin_cos_ends`] evaluates both ends of each interval. Sine rises to its peak at `pi/2` and
//! falls after it, and cosine falls on `[0, pi]`, so the ends and the peaks bound each range.
//! [`sin_cos_near`] evaluates the midpoint and widens by the half width, since both functions
//! have unit slope at most. It suits the narrow intervals of value queries.

use super::trig::{TERMS, fill, reciprocal_factorials};
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Unit roundoff `2^-53`.
const UNIT: f64 = 1.0 / 9_007_199_254_740_992.0;

/// Margin on every float error sum.
const MARGIN: f64 = 1.02;

/// Bound of `|fl(pi/2) - pi/2|`.
const HALF_PI_ERROR: f64 = 6.2e-17;

/// Bound of the series tails and of the coefficient errors above `n = 18`.
const TAIL: f64 = 3e-24;

/// Bound of the series of `|c_i| z^i` for both series on `|x| <= 1.61`.
const COEFFICIENT_SUM: f64 = 2.6;

/// Largest `n` whose `n!` is an exact F64.
const EXACT_FACTORIAL: usize = 18;

/// Signed series coefficient `(-1)^(n/2) / n!`, rounded once where `n!` is exact.
fn coefficient(n: usize, bounds: &[[f64; 2]]) -> f64 {
    let magnitude = if n <= EXACT_FACTORIAL {
        let factorial: u64 = (1..=n as u64).product();
        1.0 / factorial as f64
    } else {
        bounds.get(n).map_or(0.0, |pair| pair[0])
    };
    if (n / 2).is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

/// Float values with absolute error bounds: `[sin e, cos e, |error sin|, |error cos|]`.
type Values<R> = [Tensor<R>; 4];

/// Float `sin e` and `cos e` at represented points `e` in `[-0.03, 3.17]`, with error bounds.
fn float_sin_cos<R>(c: &R::Client, e: &Tensor<R>) -> numr::error::Result<Values<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let bounds = reciprocal_factorials(2 * TERMS);
    let x = c.add_scalar(e, -std::f64::consts::FRAC_PI_2)?;
    let z = c.mul(&x, &x)?;
    // Value and running bound of one Horner pass.
    let horner = |offset: usize| -> numr::error::Result<[Tensor<R>; 2]> {
        let mut value = fill(e, coefficient(offset + 2 * (TERMS - 1), &bounds))?;
        let mut running = fill(e, 0.0)?;
        for k in (0..TERMS - 1).rev() {
            let product = c.mul(&z, &value)?;
            let sum = c.add_scalar(&product, coefficient(offset + 2 * k, &bounds))?;
            let carried = c.add(&c.mul(&z, &running)?, &c.abs(&product)?)?;
            running = c.add(&carried, &c.abs(&sum)?)?;
            value = sum;
        }
        Ok([value, running])
    };
    // Error of one series value before the outer factor: Horner, coefficients and `z`.
    let series = |running: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        let inner = c.add_scalar(&c.add(running, &c.mul_scalar(&z, 0.51)?)?, COEFFICIENT_SUM)?;
        c.mul_scalar(&inner, UNIT * MARGIN)
    };
    let size = c.abs(&x)?;
    let shift = c.add_scalar(&c.mul_scalar(&size, 1.01 * UNIT)?, HALF_PI_ERROR)?;
    let [cos_x, cos_running] = horner(0)?;
    let cos_error = c.add_scalar(&c.add(&series(&cos_running)?, &shift)?, TAIL)?;
    let [odd, odd_running] = horner(1)?;
    let sin_x = c.mul(&x, &odd)?;
    let odd_error = c.mul(&size, &series(&odd_running)?)?;
    let product_error = c.mul_scalar(&c.abs(&sin_x)?, UNIT * MARGIN)?;
    let sin_error = c.add_scalar(&c.add(&c.add(&odd_error, &product_error)?, &shift)?, TAIL)?;
    let widen = |value: &Tensor<R>| c.mul_scalar(value, MARGIN);
    Ok([
        cos_x,
        c.neg(&sin_x)?,
        widen(&cos_error)?,
        widen(&sin_error)?,
    ])
}

/// Encloses `[sin phi, cos phi]` over every interval of `phi` from float series at its ends.
///
/// Each interval must lie inside `[-0.03, 3.17]`, as for [`super::trig::sin_cos`].
pub(crate) fn sin_cos_ends<R>(
    a: &mut Arithmetic<'_, R>,
    phi: &Interval<R>,
) -> ArithmeticResult<[Interval<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let (lo, hi) = (phi.lower(), phi.upper());
    let n = lo.numel();
    let ends = c.cat(&[lo, hi], 0)?;
    let [sine, cosine, sine_error, cosine_error] = float_sin_cos(c, &ends)?;
    let half = |value: &Tensor<R>, index: usize| value.narrow(0, index * n, n);
    // Lowest and highest certified values over both ends.
    let range = |value: &Tensor<R>, error: &Tensor<R>| -> numr::error::Result<[Tensor<R>; 2]> {
        let (at_lo, at_hi) = (half(value, 0)?, half(value, 1)?);
        let (err_lo, err_hi) = (half(error, 0)?, half(error, 1)?);
        let lower = c.minimum(&c.sub(&at_lo, &err_lo)?, &c.sub(&at_hi, &err_hi)?)?;
        let upper = c.maximum(&c.add(&at_lo, &err_lo)?, &c.add(&at_hi, &err_hi)?)?;
        Ok([c.next_down(&lower)?, c.next_up(&upper)?])
    };
    let [sin_lower, sin_upper] = range(&sine, &sine_error)?;
    let [cos_lower, cos_upper] = range(&cosine, &cosine_error)?;
    let one = fill(lo, 1.0)?;
    let minus_one = fill(lo, -1.0)?;
    let pi_low = f64::from_bits(PI[0]);
    let pi_high = f64::from_bits(PI[1]);
    // The sine peak lies inside unless the interval ends certainly before or after pi/2.
    let peak = c.mul(
        &c.le(lo, &fill(lo, pi_high * 0.5)?)?,
        &c.ge(hi, &fill(lo, pi_low * 0.5)?)?,
    )?;
    let sin_upper = c.where_cond(&peak, &one, &sin_upper)?;
    // Cosine peaks at 0 and bottoms out at pi.
    let zero = fill(lo, 0.0)?;
    let top = c.mul(&c.le(lo, &zero)?, &c.ge(hi, &zero)?)?;
    let bottom = c.mul(
        &c.le(lo, &fill(lo, pi_high)?)?,
        &c.ge(hi, &fill(lo, pi_low)?)?,
    )?;
    let cos_lower = c.where_cond(&bottom, &minus_one, &cos_lower)?;
    let cos_upper = c.where_cond(&top, &one, &cos_upper)?;
    Ok([
        clip(a, &sin_lower, &sin_upper)?,
        clip(a, &cos_lower, &cos_upper)?,
    ])
}

/// Encloses `[sin phi, cos phi]` over every interval of `phi` from float series at its midpoint.
///
/// Each interval must lie inside `[-0.03, 3.17]`. The enclosure widens by the half width, so it
/// is tight only on narrow intervals.
pub(crate) fn sin_cos_near<R>(
    a: &mut Arithmetic<'_, R>,
    phi: &Interval<R>,
) -> ArithmeticResult<[Interval<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let (lo, hi) = (phi.lower(), phi.upper());
    let middle = c.add(lo, &c.mul_scalar(&c.sub(hi, lo)?, 0.5)?)?;
    let middle = c.minimum(&c.maximum(&middle, lo)?, hi)?;
    // Every point of the interval lies within `reach` of the midpoint.
    let reach = c.next_up(&c.maximum(&c.sub(&middle, lo)?, &c.sub(hi, &middle)?)?)?;
    let [sine, cosine, sine_error, cosine_error] = float_sin_cos(c, &middle)?;
    let around = |value: &Tensor<R>, error: &Tensor<R>| -> numr::error::Result<[Tensor<R>; 2]> {
        let spread = c.next_up(&c.add(error, &reach)?)?;
        Ok([
            c.next_down(&c.sub(value, &spread)?)?,
            c.next_up(&c.add(value, &spread)?)?,
        ])
    };
    let [sin_lower, sin_upper] = around(&sine, &sine_error)?;
    let [cos_lower, cos_upper] = around(&cosine, &cosine_error)?;
    Ok([
        clip(a, &sin_lower, &sin_upper)?,
        clip(a, &cos_lower, &cos_upper)?,
    ])
}

/// Interval `[lower, upper]` clipped into `[-1, 1]`.
fn clip<R>(
    a: &mut Arithmetic<'_, R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let one = fill(lower, 1.0)?;
    let minus_one = fill(lower, -1.0)?;
    let lower = c.minimum(&c.maximum(lower, &minus_one)?, &one)?;
    let upper = c.maximum(&c.minimum(upper, &one)?, &minus_one)?;
    a.bounds(&lower, &upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    /// `[lower, upper]` per row of sine and of cosine.
    type Bounds = [Vec<[f64; 2]>; 2];

    /// Interval sine and cosine of one form.
    type Form = fn(
        &mut Arithmetic<'_, CpuRuntime>,
        &Interval<CpuRuntime>,
    ) -> ArithmeticResult<[Interval<CpuRuntime>; 2]>;

    fn evaluate(form: Form, lower: &[f64], upper: &[f64]) -> Result<Bounds, StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let tensor =
            |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
        let (lower, upper) = (tensor(lower)?, tensor(upper)?);
        let read = |value: &Interval<CpuRuntime>| -> numr::error::Result<Vec<[f64; 2]>> {
            let low = value.lower().try_to_vec::<f64>()?;
            let high = value.upper().try_to_vec::<f64>()?;
            Ok(low.into_iter().zip(high).map(|(l, h)| [l, h]).collect())
        };
        Ok(with_interval_arithmetic(
            &client,
            DType::F64,
            "series",
            |a| {
                let phi = a.bounds(&lower, &upper)?;
                let [s, k] = form(a, &phi)?;
                Ok([read(&s)?, read(&k)?])
            },
        )?)
    }

    #[test]
    fn point_enclosures_hold_library_values_within_a_few_ulps() -> Result<(), StructureError> {
        let angles = [
            -0.02,
            0.0,
            1e-3,
            0.5,
            1.0,
            std::f64::consts::FRAC_PI_2,
            2.5,
            3.1,
            std::f64::consts::PI,
            3.16,
        ];
        for form in [sin_cos_ends as Form, sin_cos_near as Form] {
            let [sine, cosine] = evaluate(form, &angles, &angles)?;
            for ((angle, s), k) in angles.iter().zip(&sine).zip(&cosine) {
                for ([low, high], value) in [(*s, angle.sin()), (*k, angle.cos())] {
                    assert!(
                        low <= value.next_up() && value.next_down() <= high,
                        "angle={angle}"
                    );
                    let ulps = (high - low) / f64::EPSILON;
                    assert!(ulps <= 16.0, "angle={angle} width={ulps} eps");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn wide_intervals_hold_every_sample_and_both_peaks() -> Result<(), StructureError> {
        let lower = [0.0, 1.0, 2.0, -0.02];
        let upper = [1.0, 2.0, 3.17, 3.17];
        for form in [sin_cos_ends as Form, sin_cos_near as Form] {
            let [sine, cosine] = evaluate(form, &lower, &upper)?;
            let rows = lower.iter().zip(&upper).zip(sine.iter().zip(&cosine));
            for ((low, high), (s, k)) in rows {
                for step in 0..=64 {
                    let t = low + (high - low) * f64::from(step) / 64.0;
                    assert!(s[0] <= t.sin() && t.sin() <= s[1], "t={t}");
                    assert!(k[0] <= t.cos() && t.cos() <= k[1], "t={t}");
                }
            }
        }
        // The ends form finds the sine peak at pi/2 and the cosine floor at pi exactly.
        let [sine, cosine] = evaluate(sin_cos_ends, &lower, &upper)?;
        assert_eq!(sine.get(1).map(|s| s[1]), Some(1.0));
        assert_eq!(cosine.get(3).map(|k| k[0]), Some(-1.0));
        Ok(())
    }
}
