//! Graded node partition of each profile and the certified distance at every node.
//!
//! A profile runs in `v` from its start toward its limit. Nodes sit at the fractions `0`,
//! `2^-35 .. 2^-5`, `k/16` for `k = 1 .. 15`, `1 - 2^-5 .. 1 - 2^-35` and `1` of that span. No
//! lane is longer than its distance to the nearer end, or than `1/16`. Lanes near an end stay
//! short where `G` vanishes at a critical start and where it grows near a normal-depth pole.
//!
//! An asymptotic profile ends at the bound of a normal depth that the profile never reaches. Its
//! nodes keep `2^-36 |v_limit|` from that bound, its last lane is empty and its last node stands
//! for an infinite distance.

use super::integrand::ProfileParams;
use super::quadrature::integrate;
use crate::structures::StructureError;
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;
use solvr::integrate::CertifiedSimpsonAlgorithms;

/// Finest graded level: the node nearest an end sits `2^-DEEPEST` of the span from it.
const DEEPEST: i32 = 35;

/// Coarsest graded level, where the uniform middle lanes of width `2^-UNIFORM` start.
const UNIFORM: i32 = 4;

/// Graded nodes per end.
const GRADED: usize = (DEEPEST - UNIFORM) as usize;

/// Uniform middle nodes `k / 2^UNIFORM`.
const MIDDLE: usize = (1 << UNIFORM) - 1;

/// Lanes per profile.
pub(crate) const LANES: usize = 2 * GRADED + MIDDLE + 1;

/// Fraction of `|v_limit|` that every node keeps from an asymptotic limit.
const GUARD: f64 = 1.0 / 68_719_476_736.0;

/// Node fractions of the span, from the start (`0`) to the limit (`1`).
fn fractions() -> Vec<f64> {
    let step = 2.0_f64.powi(-UNIFORM);
    let mut values = vec![0.0];
    values.extend((UNIFORM + 1..=DEEPEST).rev().map(|k| 2.0_f64.powi(-k)));
    values.extend((1..=MIDDLE).map(|k| k as f64 * step));
    values.extend((UNIFORM + 1..=DEEPEST).map(|k| 1.0 - 2.0_f64.powi(-k)));
    values.push(1.0);
    values
}

/// Per-row profile inputs.
#[derive(Clone)]
pub(crate) struct ProfileSpec<R: Runtime<DType = DType>> {
    pub(crate) params: ProfileParams<R>,
    /// 1 on rows with a profile.
    pub(crate) active: Tensor<R>,
    /// 1 on rows whose limit is a normal-depth pole.
    pub(crate) asymptotic: Tensor<R>,
    /// +1 where `v` rises along the profile, -1 where it falls.
    pub(crate) sign: Tensor<R>,
    /// Represented start `v_s`.
    pub(crate) start: Tensor<R>,
    /// End of the start enclosure farther from the limit.
    pub(crate) start_far: Tensor<R>,
    /// Distance from the exact start to `v_s`.
    pub(crate) start_error: Bounds<R>,
    /// Represented limit `v_l`.
    pub(crate) limit: Tensor<R>,
}

/// A built profile: nodes `[m, LANES + 1]` and the distance enclosure at each node.
pub(crate) struct Profile<R: Runtime<DType = DType>> {
    pub(crate) spec: ProfileSpec<R>,
    pub(crate) nodes: Tensor<R>,
    pub(crate) reach: Bounds<R>,
}

/// 1 where an asymptotic limit lies ahead of the start along `sign`, with room for the guard.
pub(crate) fn guarded<R>(
    c: &R::Client,
    sign: &Tensor<R>,
    start: &Tensor<R>,
    limit: &Tensor<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let span = c.mul(sign, &c.sub(limit, start)?)?;
    let guard = c.mul_scalar(&c.abs(limit)?, 4.0 * GUARD)?;
    c.gt(&span, &guard)
}

/// Places the nodes, integrates every lane and accumulates the node distances.
pub(crate) fn build<R>(c: &R::Client, spec: ProfileSpec<R>) -> Result<Profile<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedSimpsonAlgorithms<R>,
{
    let m = spec.start.numel();
    let device = spec.start.device().clone();
    let fractions = fractions();
    let count = fractions.len();
    let lanes = count - 1;
    let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value.broadcast_to(&[m, count])?.contiguous()
    };
    let column = |index: usize| -> numr::error::Result<Tensor<R>> {
        let values: Vec<f64> = (0..count)
            .map(|k| f64::from(u8::from(k == index)))
            .collect();
        wide(&Tensor::<R>::from_slice(&values, &[1, count], &device)?)
    };
    let fraction = wide(&Tensor::<R>::from_slice(&fractions, &[1, count], &device)?)?;
    let start = wide(&spec.start.reshape(&[m, 1])?)?;
    let limit = wide(&spec.limit.reshape(&[m, 1])?)?;
    let span = c.sub(&limit, &start)?;
    let mut nodes = c.add(&start, &c.mul(&span, &fraction)?)?;

    // Asymptotic rows keep every interior node a guard away from the pole bound.
    let sign = wide(&spec.sign.reshape(&[m, 1])?)?;
    let guard = c.mul_scalar(&c.abs(&limit)?, GUARD)?;
    let gap = c.mul(&sign, &c.sub(&limit, &nodes)?)?;
    let kept = c.sub(&limit, &c.mul(&sign, &c.maximum(&gap, &guard)?)?)?;
    let first = column(0)?;
    let last = column(lanes)?;
    let asymptotic = wide(&spec.asymptotic.reshape(&[m, 1])?)?;
    let ends = c.maximum(&first, &last)?;
    let interior = c.mul(
        &asymptotic,
        &c.sub(
            &wide(&Tensor::<R>::ones(&[1, 1], DType::F64, &device)?)?,
            &ends,
        )?,
    )?;
    nodes = c.where_cond(&interior, &kept, &nodes)?;
    // The span ends exactly at the start and the limit.
    nodes = c.where_cond(&last, &limit, &nodes)?;
    nodes = c.where_cond(&first, &start, &nodes)?;
    let active = wide(&spec.active.reshape(&[m, 1])?)?;
    nodes = c.where_cond(&active, &nodes, &start)?;

    let low = nodes.narrow(1, 0, lanes)?.contiguous()?;
    let high = nodes.narrow(1, 1, lanes)?.contiguous()?;
    let mut left = c.minimum(&low, &high)?;
    let mut right = c.maximum(&low, &high)?;
    // The last lane of an asymptotic row reaches the pole and stays empty.
    let narrow = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value.narrow(1, 0, lanes)?.contiguous()
    };
    let final_lane = narrow(&column(lanes - 1)?)?;
    let pole = c.mul(&narrow(&asymptotic)?, &final_lane)?;
    let idle = c.sub(
        &narrow(&wide(&Tensor::<R>::ones(&[1, 1], DType::F64, &device)?)?)?,
        &narrow(&active)?,
    )?;
    let empty = c.maximum(&pole, &idle)?;
    right = c.where_cond(&empty, &left, &right)?;
    left = left.reshape(&[m * lanes])?;
    right = right.reshape(&[m * lanes])?;

    let owner: Vec<i64> = (0..m * lanes)
        .map(|lane| i64::try_from(lane / lanes).unwrap_or(i64::MAX))
        .collect();
    let owner = Tensor::<R>::from_slice(&owner, &[m * lanes], &device)?;
    let params = spec.params.select(c, &owner)?;
    let [lower, upper] = integrate(c, &params, &left, &right)?;
    let lower = lower.reshape(&[m, lanes])?;
    let upper = upper.reshape(&[m, lanes])?;

    let reach = with_interval_arithmetic(c, DType::F64, "culvert outlet profile distances", |a| {
        let mut total = a.bounds(&spec.start_error[0], &spec.start_error[1])?;
        let mut lows = vec![total.lower().clone()];
        let mut highs = vec![total.upper().clone()];
        for k in 0..lanes {
            let piece = a.bounds(
                &lower.narrow(1, k, 1)?.contiguous()?.reshape(&[m])?,
                &upper.narrow(1, k, 1)?.contiguous()?.reshape(&[m])?,
            )?;
            total = a.add(&total, &piece)?;
            lows.push(total.lower().clone());
            highs.push(total.upper().clone());
        }
        let lows: Vec<&Tensor<R>> = lows.iter().collect();
        let highs: Vec<&Tensor<R>> = highs.iter().collect();
        Ok([a.client().stack(&lows, 1)?, a.client().stack(&highs, 1)?])
    })?;
    Ok(Profile { spec, nodes, reach })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fractions_rise_from_zero_to_one_and_grade_both_ends() {
        let values = fractions();
        assert_eq!(values.len(), LANES + 1);
        assert_eq!(values.first().copied(), Some(0.0));
        assert_eq!(values.last().copied(), Some(1.0));
        for pair in values.windows(2) {
            if let [low, high] = pair {
                assert!(low < high);
            }
        }
        // Each interior lane is no longer than its distance to the nearer end.
        for pair in values.windows(2).skip(1).take(LANES - 2) {
            if let [low, high] = pair {
                let width = high - low;
                assert!(width <= low.min(1.0 - high) * 2.0 + 1e-300);
            }
        }
    }
}
