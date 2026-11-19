//! Two-sided certified contraction of an increasing trace root bracket.
//!
//! Midpoint bisection with the correction `root ± R / κ` tightens only the side opposite each
//! sample. When every sample lands on one side of the root, the other side halves per round.
//! A bracket `[a, b]` with certified signs `R(a) <= 0 <= R(b)` and a slope floor `κ` over it
//! gives both sides at once:
//!
//! - `E(b) - E(r) >= κ (b - r)`, so `r >= b - R(b).upper / κ`.
//! - `E(r) - E(a) >= κ (r - a)`, so `r <= a - R(a).lower / κ`.
//!
//! The certified bisection then finishes on the contracted bracket.

use super::{client::TraceClient, error::trace_error, point::Evaluation, roots::Residual};
use crate::channel::flow::{ChannelError, context::Context, surveyed::work::QueryWork};
use numr::{dtype::DType, runtime::Runtime};
use std::cmp::Ordering;

/// A slope floor of the residual over the normalized-depth bracket `[a, b]`.
pub(super) type SlopeFloor<'a> = dyn Fn(f64, f64) -> Result<f64, ChannelError> + 'a;

/// Contraction rounds per bracket. Each round evaluates both new ends and one slope floor.
const ROUNDS: usize = 8;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// Contracts `[lower, upper]` around the root of the increasing `residual`.
    ///
    /// Each end of the returned bracket holds a certified residual sign, as the certified
    /// bisection requires. A contracted end whose residual straddles zero keeps the previous
    /// end on its side. A round runs while the slope floor is finite and positive. Rounds stop
    /// once a round fails to halve the width. The returned bracket contains every root the
    /// input bracket contains. An input end without a certified sign returns the input.
    pub(super) fn surveyed_trace_contract(
        &self,
        evaluation: &Evaluation<'_>,
        lower: f64,
        upper: f64,
        residual: &Residual<'_, R>,
        slope: &SlopeFloor<'_>,
    ) -> Result<(f64, f64), ChannelError> {
        let value = |t: f64| -> Result<[f64; 2], ChannelError> {
            evaluation
                .work
                .charge(QueryWork::Residual, 1)
                .map_err(trace_error)?;
            let value = residual(&self.tensor(&[t])?)?;
            Ok([value.lower().item::<f64>()?, value.upper().item::<f64>()?])
        };
        let below = |[low, high]: [f64; 2]| high <= 0.0 && low.is_finite();
        let above = |[low, high]: [f64; 2]| low >= 0.0 && high.is_finite();
        let open = |a: f64, b: f64| a.partial_cmp(&b) == Some(Ordering::Less);
        let (mut a, mut b) = (lower, upper);
        if !open(a, b) {
            return Ok((a, b));
        }
        let (mut at_a, mut at_b) = (value(a)?, value(b)?);
        if !(below(at_a) && above(at_b)) {
            return Ok((a, b));
        }
        for _ in 0..ROUNDS {
            if !open(a, b) {
                break;
            }
            let kappa = slope(a, b)?;
            if !(kappa > 0.0 && kappa.is_finite()) {
                break;
            }
            // Each nearest quotient and difference moves one step outward.
            let rise = (at_b[1] / kappa).next_up();
            let fall = (-at_a[0] / kappa).next_up();
            let next_a = a.max((b - rise).next_down());
            let next_b = b.min((a + fall).next_up());
            if next_a.partial_cmp(&next_b).is_none_or(Ordering::is_gt) {
                return Err(ChannelError::numerical(format!(
                    "scalar channel boundary: surveyed trace contraction crossed at [{next_a:e}, {next_b:e}]: inspect the slope floor"
                )));
            }
            let width = b - a;
            let mut moved = false;
            if next_a > a && next_a < b {
                let at = value(next_a)?;
                if below(at) {
                    (a, at_a, moved) = (next_a, at, true);
                }
            }
            if next_b < b && next_b > a {
                let at = value(next_b)?;
                if above(at) {
                    (b, at_b, moved) = (next_b, at, true);
                }
            }
            if !moved || b - a > 0.5 * width {
                break;
            }
        }
        Ok((a, b))
    }
}
