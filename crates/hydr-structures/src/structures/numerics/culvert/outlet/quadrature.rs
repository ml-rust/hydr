//! Certified Simpson integrals of the direct-step integrand over coordinate lanes.
//!
//! Each lane carries its own integrand parameters. Value queries enclose `G` over each node
//! interval and fourth-derivative queries bound `24 |c_4|` of the order-4 jet over each panel.
//! Zero-width lanes return exact zeros without callbacks. Lanes run in chunks, so the work of
//! one call stays bounded.

use super::integrand::{ProfileParams, integrand};
use super::jet;
use super::trig::fill;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, IntegrateError};
use std::cell::Cell;

/// Simpson panels per lane, a power of two.
pub(crate) const PANELS: usize = 64;

/// Most lanes per Simpson call.
const CHUNK: usize = 2048;

/// Lower and upper bounds that reject the enclosing call on rows outside `valid`.
fn admitted<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    valid: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.where_cond(valid, value.lower(), &fill(valid, 1.0)?)?;
    let upper = c.where_cond(valid, value.upper(), &fill(valid, 0.0)?)?;
    a.bounds(&lower, &upper)
}

/// Integral bounds of `G` over `[left, right]` per lane, with lane parameters `params`.
///
/// Every lane needs `left <= right` inside the profile domain, where the integrand is smooth.
pub(crate) fn integrate<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let lanes = left.shape().first().copied().unwrap_or(0);
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut start = 0;
    while start < lanes {
        let len = CHUNK.min(lanes - start);
        let chunk = params.narrow(start, len)?;
        let [low, high] = chunk_integral(
            c,
            &chunk,
            &left.narrow(0, start, len)?,
            &right.narrow(0, start, len)?,
        )?;
        lower.push(low);
        upper.push(high);
        start += len;
    }
    if lower.len() == 1 {
        let (Some(low), Some(high)) = (lower.pop(), upper.pop()) else {
            return Err(StructureError::numerical(
                "culvert outlet quadrature lost its only chunk: report this runtime",
            ));
        };
        return Ok([low, high]);
    }
    if lower.is_empty() {
        let empty = left.clone();
        return Ok([empty.clone(), empty]);
    }
    let lower: Vec<&Tensor<R>> = lower.iter().collect();
    let upper: Vec<&Tensor<R>> = upper.iter().collect();
    Ok([c.cat(&lower, 0)?, c.cat(&upper, 0)?])
}

fn chunk_integral<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let lanes = left.numel();
    let options = CertifiedSimpsonOptions {
        panels: PANELS,
        max_points: lanes.saturating_mul(3 * PANELS + 1),
    };
    let cause = Cell::new(None);
    let fail = |error: StructureError| {
        let reason = error.to_string();
        cause.set(Some(error));
        IntegrateError::NumericalError { message: reason }
    };
    let result = c.certified_simpson(
        left,
        right,
        options,
        |query| {
            let outcome =
                with_interval_arithmetic(c, DType::F64, "culvert outlet profile values", |a| {
                    let lane = params.select(c, &query.owner)?;
                    let v = a.bounds(&query.lower, &query.upper)?;
                    let over = integrand(a, &lane, &v, 1)?;
                    let value = jet::coefficient(a, &over.value, 0, &lane.rise)?;
                    let value = admitted(a, &value, &over.valid)?;
                    let point = a.point(&query.point)?;
                    let at = integrand(a, &lane, &point, 1)?;
                    let nominal = jet::coefficient(a, &at.value, 0, &lane.rise)?;
                    let ac = a.client();
                    let half = ac.mul_scalar(&ac.sub(nominal.upper(), nominal.lower())?, 0.5)?;
                    let middle = ac.add(nominal.lower(), &half)?;
                    let middle = ac.maximum(value.lower(), &ac.minimum(&middle, value.upper())?)?;
                    Ok([middle, value.lower().clone(), value.upper().clone()])
                });
            match outcome {
                Ok([value, lower, upper]) => Ok(solvr::integrate::CertifiedSimpsonValues {
                    value,
                    lower,
                    upper,
                }),
                Err(error) => Err(fail(error.into())),
            }
        },
        |panel| {
            let outcome = with_interval_arithmetic(
                c,
                DType::F64,
                "culvert outlet profile derivatives",
                |a| {
                    let lane = params.select(c, &panel.owner)?;
                    let v = a.bounds(&panel.lower, &panel.upper)?;
                    let over = integrand(a, &lane, &v, 5)?;
                    let fourth = jet::coefficient(a, &over.value, 4, &lane.rise)?;
                    let fourth = admitted(a, &fourth, &over.valid)?;
                    let factor = a.point(&fill(&lane.rise, 24.0)?)?;
                    let scaled = a.mul(&factor, &fourth)?;
                    let ac = a.client();
                    Ok(ac.maximum(&ac.abs(scaled.lower())?, &ac.abs(scaled.upper())?)?)
                },
            );
            outcome.map_err(|error| fail(error.into()))
        },
    );
    match (result, cause.take()) {
        (_, Some(error)) => Err(error),
        (Ok(integral), None) => Ok([integral.lower().clone(), integral.upper().clone()]),
        (Err(error), None) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn box_profile_integral_matches_its_closed_form_without_friction() -> Result<(), StructureError>
    {
        // No friction and F = 0: G = D / S per unit s, so the integral is D (right - left) / S.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let tensor =
            |values: &[f64]| Tensor::<CpuRuntime>::from_slice(values, &[values.len()], &device);
        let params = ProfileParams {
            circular: tensor(&[0.0, 0.0])?,
            beta: [tensor(&[2.0, 2.0])?, tensor(&[2.0, 2.0])?],
            rise: tensor(&[1.5, 1.5])?,
            froude: [tensor(&[0.0, 0.0])?, tensor(&[0.0, 0.0])?],
            log_friction: [tensor(&[-800.0, -800.0])?, tensor(&[-800.0, -800.0])?],
            slope: tensor(&[0.01, 0.01])?,
            numerator: tensor(&[1.0, 1.0])?,
            denominator: tensor(&[1.0, 1.0])?,
        };
        let [lower, upper] = integrate(
            &client,
            &params,
            &tensor(&[0.25, 0.5])?,
            &tensor(&[0.75, 0.5])?,
        )?;
        let lower = lower.try_to_vec::<f64>()?;
        let upper = upper.try_to_vec::<f64>()?;
        assert!(lower[0] <= 75.0 && 75.0 <= upper[0]);
        assert!(upper[0] - lower[0] < 1e-9);
        assert_eq!((lower[1], upper[1]), (0.0, 0.0));
        Ok(())
    }
}
