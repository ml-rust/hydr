//! Interval sine and cosine on the circle angle domain `[0, pi]`.
//!
//! With `x = phi - pi/2`, `sin phi = cos x` and `cos phi = -sin x`. Both series run in `z = x^2`
//! by Horner's rule with outward coefficient bounds. The Lagrange remainder bounds the
//! truncation for `|x| <= 1.6`, which covers `phi` in `[-0.029, 3.17]`.

use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use numr::dtype::DType;
use numr::ops::{BinaryOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Terms of each series. The cosine series ends at `x^26` and the sine series at `x^27`.
const TERMS: usize = 14;

/// Upper bound of `|x|^28 / 28!` for `|x| <= 1.6`.
const COSINE_REMAINDER: f64 = 2e-24;

/// Upper bound of `|x|^29 / 29!` for `|x| <= 1.6`.
const SINE_REMAINDER: f64 = 1e-25;

/// Outward bounds of `1 / n!` for `n` in `0..count`.
fn reciprocal_factorials(count: usize) -> Vec<[f64; 2]> {
    let mut bounds = Vec::with_capacity(count);
    let (mut lower, mut upper) = (1.0_f64, 1.0_f64);
    bounds.push([lower, upper]);
    for n in 1..count {
        let divisor = n as f64;
        lower = (lower / divisor).next_down();
        upper = (upper / divisor).next_up();
        bounds.push([lower, upper]);
    }
    bounds
}

/// `F64` tensor shaped like `like`, holding `value` everywhere.
pub(crate) fn fill<R>(like: &Tensor<R>, value: f64) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
{
    Tensor::<R>::full_scalar(like.shape(), DType::F64, value, like.device())
}

/// Encloses `-x`.
pub(crate) fn negate<R>(a: &mut Arithmetic<'_, R>, x: &Interval<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.neg(x.upper())?;
    let upper = c.neg(x.lower())?;
    a.bounds(&lower, &upper)
}

/// Encloses `sum_k sign^k z^k bounds[offset + 2k]` by Horner's rule.
fn series<R>(
    a: &mut Arithmetic<'_, R>,
    z: &Interval<R>,
    factorials: &[[f64; 2]],
    offset: usize,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let like = z.lower();
    let mut sum: Option<Interval<R>> = None;
    for k in (0..TERMS).rev() {
        let [lower, upper] = factorials
            .get(offset + 2 * k)
            .copied()
            .unwrap_or([0.0, 0.0]);
        let coefficient = if k % 2 == 0 {
            a.bounds(&fill(like, lower)?, &fill(like, upper)?)?
        } else {
            a.bounds(&fill(like, -upper)?, &fill(like, -lower)?)?
        };
        sum = Some(match sum {
            None => coefficient,
            Some(inner) => {
                let product = a.mul(z, &inner)?;
                a.add(&coefficient, &product)?
            }
        });
    }
    match sum {
        Some(value) => Ok(value),
        None => a.point(&fill(like, 0.0)?),
    }
}

/// Widens `x` by `[-radius, radius]` and clips it to `[-1, 1]`.
fn settle<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    radius: f64,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = x.lower();
    let error = a.bounds(&fill(like, -radius)?, &fill(like, radius)?)?;
    let wide = a.add(x, &error)?;
    let lower = c.maximum(wide.lower(), &fill(like, -1.0)?)?;
    let upper = c.minimum(wide.upper(), &fill(like, 1.0)?)?;
    a.bounds(&lower, &upper)
}

/// Encloses `[sin phi, cos phi]` over every interval of `phi`.
///
/// Each interval must lie inside `[-0.029, 3.17]`. The circle angle domain `[0, pi]` and its
/// one-ulp outward bounds lie inside it.
pub(crate) fn sin_cos<R>(
    a: &mut Arithmetic<'_, R>,
    phi: &Interval<R>,
) -> ArithmeticResult<[Interval<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = phi.lower();
    // pi / 2 halves the bounds of pi exactly.
    let half_pi = a.bounds(
        &fill(like, f64::from_bits(PI[0]) * 0.5)?,
        &fill(like, f64::from_bits(PI[1]) * 0.5)?,
    )?;
    let x = a.sub(phi, &half_pi)?;
    let square = a.mul(&x, &x)?;
    let zero = fill(like, 0.0)?;
    let z = a.bounds(&c.maximum(square.lower(), &zero)?, square.upper())?;
    let factorials = reciprocal_factorials(2 * TERMS + 1);
    let cos_x = series(a, &z, &factorials, 0)?;
    let odd = series(a, &z, &factorials, 1)?;
    let sin_x = a.mul(&x, &odd)?;
    let sine = settle(a, &cos_x, COSINE_REMAINDER)?;
    let negative = negate(a, &sin_x)?;
    let cosine = settle(a, &negative, SINE_REMAINDER)?;
    Ok([sine, cosine])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    #[test]
    fn remainder_bounds_cover_the_series_tails() {
        let factorials = reciprocal_factorials(30);
        let tail = |power: i32| {
            factorials
                .get(usize::try_from(power).unwrap_or(0))
                .map_or(f64::INFINITY, |pair| 1.6_f64.powi(power) * pair[1])
        };
        assert!(tail(28) * 1.01 < COSINE_REMAINDER);
        assert!(tail(29) * 1.01 < SINE_REMAINDER);
    }

    #[test]
    fn enclosures_hold_library_values_and_stay_narrow() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let angles = [
            0.0,
            1e-3,
            0.5,
            1.0,
            std::f64::consts::FRAC_PI_2,
            2.5,
            std::f64::consts::PI,
        ];
        let phi = Tensor::<CpuRuntime>::from_slice(&angles, &[angles.len()], &device)?;
        let [sine, cosine] = with_interval_arithmetic(&client, DType::F64, "trig", |a| {
            let x = a.point(&phi)?;
            let [s, k] = sin_cos(a, &x)?;
            Ok([
                [
                    s.lower().try_to_vec::<f64>()?,
                    s.upper().try_to_vec::<f64>()?,
                ],
                [
                    k.lower().try_to_vec::<f64>()?,
                    k.upper().try_to_vec::<f64>()?,
                ],
            ])
        })?;
        for (row, angle) in angles.iter().enumerate() {
            for ([lower, upper], value) in [(&sine, angle.sin()), (&cosine, angle.cos())] {
                let (low, high) = (lower[row], upper[row]);
                assert!(
                    low <= value.next_up() && value.next_down() <= high,
                    "row={row}"
                );
                // Fourteen outward Horner steps on terms near one.
                assert!(high - low <= 64.0 * f64::EPSILON, "row={row}");
            }
        }
        Ok(())
    }
}
