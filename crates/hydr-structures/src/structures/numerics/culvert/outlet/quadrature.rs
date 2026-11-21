//! Certified Simpson integrals of the direct-step integrand over coordinate lanes.
//!
//! Each lane carries its own integrand parameters. Value queries enclose `G` over each node
//! interval and fourth-derivative queries bound `24 |c_4|` of the order-4 jet over groups of
//! panels. Zero-width lanes return exact zeros without callbacks. Lanes run in chunks, so the
//! work of one call stays bounded.

use super::integrand::{ProfileParams, integrand};
use super::jet;
use super::trig::fill;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, IntegrateError};
use std::cell::Cell;

/// Simpson panels per lane at each accuracy level, coarsest first.
///
/// Each level has four times the panels of the one before, so its remainder bound is about
/// `4^4` times smaller.
pub(crate) const PANELS: [usize; 3] = [4, 16, 64];

/// Ratio of remainder bounds between neighbouring levels.
pub(crate) const LEVEL_RATIO: f64 = 256.0;

/// Most fourth-derivative jets per lane, a power of two. Lanes with more panels bound groups
/// of neighbouring panels, each a quarter of the lane: the width of a survey panel. Wider
/// groups let interval jets near a pole or a shallow depth lose positivity or overflow.
const BOUND_GROUPS: usize = 4;

/// Most lanes per Simpson call.
const CHUNK: usize = 2048;

/// Lane integral bounds with the Simpson remainder bound of each lane.
pub(crate) struct Quadrature<R: Runtime<DType = DType>> {
    pub(crate) bounds: Bounds<R>,
    pub(crate) remainder: Tensor<R>,
}

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
/// `level` (F64) holds each lane's index into [`PANELS`]. Every lane needs `left <= right`
/// inside the profile domain, where the integrand is smooth.
pub(crate) fn integrate<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
    level: &Tensor<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let zero = fill(left, 0.0)?;
    refine(c, params, left, right, level, 0, [zero.clone(), zero])
}

/// [`integrate`] on the lanes whose level is `first` or above. Other lanes keep `base`.
pub(crate) fn refine<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
    level: &Tensor<R>,
    first: usize,
    base: Bounds<R>,
) -> Result<Bounds<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let [mut lower, mut upper] = base;
    for (index, &panels) in PANELS.iter().enumerate().skip(first) {
        // Lanes at other levels collapse to zero width, which Simpson skips.
        let here = c.eq(level, &fill(level, index as f64)?)?;
        let right = c.where_cond(&here, right, left)?;
        let found = simpson(c, params, left, &right, panels)?;
        lower = c.where_cond(&here, &found.bounds[0], &lower)?;
        upper = c.where_cond(&here, &found.bounds[1], &upper)?;
    }
    Ok([lower, upper])
}

/// Integral bounds and Simpson remainder bounds per lane with `panels` panels in every lane.
pub(crate) fn simpson<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
    panels: usize,
) -> Result<Quadrature<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let lanes = left.shape().first().copied().unwrap_or(0);
    let mut parts = Vec::new();
    let mut start = 0;
    while start < lanes {
        let len = CHUNK.min(lanes - start);
        let chunk = params.narrow(start, len)?;
        parts.push(chunk_integral(
            c,
            &chunk,
            &left.narrow(0, start, len)?,
            &right.narrow(0, start, len)?,
            panels,
        )?);
        start += len;
    }
    if parts.len() == 1 {
        return parts.pop().ok_or_else(|| {
            StructureError::numerical(
                "culvert outlet quadrature lost its only chunk: report this runtime",
            )
        });
    }
    if parts.is_empty() {
        let empty = left.clone();
        return Ok(Quadrature {
            bounds: [empty.clone(), empty.clone()],
            remainder: empty,
        });
    }
    let join = |pick: fn(&Quadrature<R>) -> &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        let pieces: Vec<&Tensor<R>> = parts.iter().map(pick).collect();
        c.cat(&pieces, 0)
    };
    Ok(Quadrature {
        bounds: [join(|q| &q.bounds[0])?, join(|q| &q.bounds[1])?],
        remainder: join(|q| &q.remainder)?,
    })
}

fn chunk_integral<R>(
    c: &R::Client,
    params: &ProfileParams<R>,
    left: &Tensor<R>,
    right: &Tensor<R>,
    panels: usize,
) -> Result<Quadrature<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let lanes = left.numel();
    let options = CertifiedSimpsonOptions {
        panels,
        max_points: lanes.saturating_mul(3 * panels + 1),
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
                    // The bounds alone enter the enclosure. The nominal value is the clamped
                    // midpoint of the node enclosure, which holds the query point.
                    let ac = a.client();
                    let half = ac.mul_scalar(&ac.sub(value.upper(), value.lower())?, 0.5)?;
                    let middle = ac.add(value.lower(), &half)?;
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
                    // One jet bounds each group of neighbouring panels: a bound over their
                    // union covers every panel in it.
                    let groups = panels.min(BOUND_GROUPS);
                    let per = panels / groups;
                    let active = panel.owner.numel() / panels;
                    let shape = [active, groups, per];
                    let first = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
                        value
                            .reshape(&shape)?
                            .narrow(2, 0, 1)?
                            .contiguous()?
                            .reshape(&[active * groups])
                    };
                    let owner = first(&panel.owner)?;
                    let low = first(&panel.lower)?;
                    let high = panel
                        .upper
                        .reshape(&shape)?
                        .narrow(2, per - 1, 1)?
                        .contiguous()?
                        .reshape(&[active * groups])?;
                    let lane = params.select(c, &owner)?;
                    let v = a.bounds(&low, &high)?;
                    let over = integrand(a, &lane, &v, 5)?;
                    let fourth = jet::coefficient(a, &over.value, 4, &lane.rise)?;
                    let fourth = admitted(a, &fourth, &over.valid)?;
                    let factor = a.point(&fill(&lane.rise, 24.0)?)?;
                    let scaled = a.mul(&factor, &fourth)?;
                    let ac = a.client();
                    let bound = ac.maximum(&ac.abs(scaled.lower())?, &ac.abs(scaled.upper())?)?;
                    Ok(bound
                        .reshape(&[active, groups, 1])?
                        .broadcast_to(&shape)?
                        .contiguous()?
                        .reshape(&[active * panels])?)
                },
            );
            outcome.map_err(|error| fail(error.into()))
        },
    );
    match (result, cause.take()) {
        (_, Some(error)) => Err(error),
        (Ok(integral), None) => Ok(Quadrature {
            bounds: [integral.lower().clone(), integral.upper().clone()],
            remainder: integral.quadrature_error_bound().clone(),
        }),
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
            &tensor(&[0.0, 2.0])?,
        )?;
        let lower = lower.try_to_vec::<f64>()?;
        let upper = upper.try_to_vec::<f64>()?;
        assert!(lower[0] <= 75.0 && 75.0 <= upper[0]);
        assert!(upper[0] - lower[0] < 1e-9);
        assert_eq!((lower[1], upper[1]), (0.0, 0.0));
        Ok(())
    }
}
