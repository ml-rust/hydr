//! Interval powers `x^p` for `x >= 0` and `p > 0` as `exp(p ln x)`.
//!
//! `cr_log` and `cr_exp` round correctly, so stepping each result one F64 outward encloses the
//! exact value. `x^p` rises in `x`, so each endpoint of `x` gives its own bound.

use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses `x^p` for intervals `x` with a nonnegative lower end and points `exponent > 0`.
///
/// An endpoint at zero gives an exact zero bound.
pub(super) fn power<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    x: &Interval<R>,
    exponent: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = &barrel.zero;
    let lower_wet = c.gt(x.lower(), zero)?;
    let upper_wet = c.gt(x.upper(), zero)?;
    let p = a.point(exponent)?;
    let mut ends = Vec::with_capacity(2);
    for (endpoint, wet) in [(x.lower(), &lower_wet), (x.upper(), &upper_wet)] {
        let operand = a.point(&c.where_cond(wet, endpoint, &barrel.one)?)?;
        let logarithm = a.log_positive(&operand)?;
        ends.push(a.mul(&p, &logarithm)?);
    }
    let lower = c.maximum(&c.next_down(&c.cr_exp(ends[0].lower())?)?, zero)?;
    let upper = c.next_up(&c.cr_exp(ends[1].upper())?)?;
    let lower = c.where_cond(&lower_wet, &lower, zero)?;
    let upper = c.where_cond(&upper_wet, &upper, zero)?;
    a.bounds(&lower, &upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    #[test]
    fn powers_enclose_exact_values_and_keep_zero_exact() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = vec![CulvertBarrel::circular(1.0)?; 4];
        let barrel = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        let base = Tensor::<CpuRuntime>::from_slice(&[4.0, 3.5, 0.0, 0.25], &[4], &device)?;
        let exponent = Tensor::<CpuRuntime>::from_slice(&[2.0, 1.0, 0.667, 0.5], &[4], &device)?;
        let [lower, upper] = with_interval_arithmetic(&client, DType::F64, "power", |a| {
            let x = a.point(&base)?;
            let value = power(a, &barrel, &x, &exponent)?;
            Ok([value.lower().clone(), value.upper().clone()])
        })?;
        let lower = lower.try_to_vec::<f64>()?;
        let upper = upper.try_to_vec::<f64>()?;
        // 4^2 = 16, 3.5^1 = 3.5 and 0.25^0.5 = 0.5 are exact, so each bound holds them.
        // The width is a few ulps of `p ln x`, which `exp` turns into relative width.
        for (row, exact) in [(0, 16.0), (1, 3.5), (3, 0.5)] {
            assert!(lower[row] <= exact && exact <= upper[row], "row={row}");
            assert!(
                upper[row] - lower[row] <= 32.0 * f64::EPSILON * exact,
                "row={row}"
            );
        }
        assert_eq!((lower[2], upper[2]), (0.0, 0.0));
        Ok(())
    }
}
