//! Two-input `ln(e^a + e^b)` from correctly rounded `exp` and `log1p`.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::CorrectlyRoundedOps;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

use crate::structures::StructureError;

/// Returns `ln(e^a + e^b)` elementwise as `m + log1p(exp(n - m))`.
///
/// `m = max(a, b)` and `n = min(a, b)`. Where `m` is `-inf` the result is `-inf`.
/// That guard removes the NaN of `-inf - -inf`. `negative_infinity` holds `-inf` in every lane.
///
/// Error bound for finite `m`, with `d = n - m <= 0` and `t = e^d <= 1`:
/// - The subtraction rounds `d` with relative error at most `2^-53`. The slope of
///   `ln(1 + e^d)` is at most `1/2`, and `|d| e^d / (1 + e^d) < 0.28`, so it adds below `2^-54`.
/// - `cr_exp` adds relative error at most `2^-53` to `t`. Through `log1p`, whose slope is
///   `1/(1 + t)`, that is at most `2^-54` absolute.
/// - `cr_log1p` rounds `s <= ln 2` once, which is at most `2^-54`.
/// - The final add rounds once, at most `ulp(result)/2`.
/// - Total absolute error is at most `3 * 2^-54 + ulp(result)/2`. The first term carries
///   enough slack to cover the second-order products and a subnormal `t`.
///
/// Every step is correctly rounded or IEEE basic, so the bits are identical on every runtime
/// that implements `CorrectlyRoundedOps`.
pub(super) fn log_add_exp<R: Runtime<DType = DType>>(
    c: &R::Client,
    a: &Tensor<R>,
    b: &Tensor<R>,
    negative_infinity: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::CorrectlyRoundedOps<R>,
{
    let m = c.maximum(a, b)?;
    let n = c.minimum(a, b)?;
    let sum = c.add(&m, &c.cr_log1p(&c.cr_exp(&c.sub(&n, &m)?)?)?)?;
    let absent = c.eq(&m, negative_infinity)?;
    Ok(c.where_cond(&absent, negative_infinity, &sum)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn values(a: &[f64], b: &[f64]) -> Result<Vec<f64>, StructureError> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let tensor = |v: &[f64]| Tensor::<CpuRuntime>::from_slice(v, &[v.len()], &device);
        let result = log_add_exp(
            &client,
            &tensor(a)?,
            &tensor(b)?,
            &tensor(&vec![f64::NEG_INFINITY; a.len()])?,
        )?;
        Ok(result.try_to_vec::<f64>()?)
    }

    #[test]
    fn absent_strips_give_exact_operands() -> Result<(), StructureError> {
        let result = values(
            &[f64::NEG_INFINITY, 0.75, f64::NEG_INFINITY],
            &[f64::NEG_INFINITY, f64::NEG_INFINITY, -3.5],
        )?;
        assert_eq!(result[0], f64::NEG_INFINITY);
        assert_eq!(result[1].to_bits(), 0.75_f64.to_bits());
        assert_eq!(result[2].to_bits(), (-3.5_f64).to_bits());
        Ok(())
    }

    #[test]
    fn equal_operands_add_ln_two() -> Result<(), StructureError> {
        let result = values(&[1.25], &[1.25])?;
        let ln_two = numr::algorithm::correctly_rounded::log(2.0);
        assert_eq!(result[0].to_bits(), (1.25 + ln_two).to_bits());
        Ok(())
    }

    #[test]
    fn result_is_symmetric_and_within_bound() -> Result<(), StructureError> {
        let a = [0.1, -2.0, 7.5, -0.3];
        let b = [0.2, -2.5, -1.0, 4.0];
        let forward = values(&a, &b)?;
        let reverse = values(&b, &a)?;
        for (index, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(forward[index].to_bits(), reverse[index].to_bits());
            let reference = x.max(y) + (-(x - y).abs()).exp().ln_1p();
            let bound = 3.0 * 2.0_f64.powi(-54) + 2.0_f64.powi(-53) * reference.abs() * 2.0;
            assert!((forward[index] - reference).abs() <= bound);
        }
        Ok(())
    }
}
