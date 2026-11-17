//! Fixed-point iteration over end stages and signed link volumes, with separate budgets.

use super::context::{Context, Ops};
use super::evaluate::{Attempt, Evaluation};
use super::start::Start;
use super::{SurfaceChannelConfig, SurfaceChannelError};
use crate::channel::ChannelReach;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::AndersonAlgorithms;
use solvr::optimize::roots::{AndersonConvergence, AndersonOptions, RootOptions};
use std::cell::{Cell, RefCell};

/// Separate head and transfer residuals between an iterate and its map value.
pub(super) struct Residuals {
    /// Largest absolute end-stage change, in metres.
    pub head_m: f64,
    /// Largest absolute signed-volume change, in cubic metres.
    pub transfer_m3: f64,
    /// Largest change over its allowance, across heads and transfers.
    pub norm: f64,
    /// Both budgets pass and support and regime flags repeat exactly.
    pub converged: bool,
}

/// Converged final evaluation of one attempted interval.
pub(super) struct Solved<R: Runtime<DType = DType>> {
    pub evaluation: Evaluation<R>,
    pub iterations: usize,
    pub residuals: Residuals,
    /// Contraction per update over the last two updates, `sqrt(r_k / r_{k-2})`. Zero below two
    /// updates or after a zero norm.
    pub contraction: f64,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Head and transfer residuals of map value `gx` at iterate `x`.
    pub fn residuals(
        &self,
        config: &SurfaceChannelConfig,
        x: &Tensor<R>,
        gx: &Tensor<R>,
    ) -> Result<Residuals, SurfaceChannelError> {
        Ok(self.residuals_against(config, x, gx, None)?.0)
    }

    /// Residuals of `gx` at `x`, and whether `gx` repeats iterate `earlier` bit for bit.
    ///
    /// Every maximum, and the repeat test, comes back in one device-to-host read.
    fn residuals_against(
        &self,
        config: &SurfaceChannelConfig,
        x: &Tensor<R>,
        gx: &Tensor<R>,
        earlier: Option<&Tensor<R>>,
    ) -> Result<(Residuals, bool), SurfaceChannelError> {
        let c = &self.client;
        let n = self.surface_count + self.slot_count;
        let l = self.link_count;
        let change = c.abs(&c.sub(gx, x)?)?;
        let largest = |t: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.max(t, &[0], false)?.reshape(&[1])?)
        };
        let budget = |start: usize, len: usize, absolute: f64, relative: f64| {
            let allowance = c.add_scalar(
                &c.mul_scalar(&c.abs(&gx.narrow(0, start, len)?)?, relative)?,
                absolute,
            )?;
            let part = change.narrow(0, start, len)?;
            Ok::<_, SurfaceChannelError>([largest(&part)?, largest(&c.div(&part, &allowance)?)?])
        };
        let [head, head_ratio] = budget(0, n, config.head_absolute_m, config.head_relative)?;
        let [transfer, transfer_ratio] =
            budget(n, l, config.transfer_absolute_m3, config.transfer_relative)?;
        let mut parts = vec![
            head,
            head_ratio,
            transfer,
            transfer_ratio,
            largest(&change.narrow(0, n + l, n + l)?)?,
        ];
        if let Some(earlier) = earlier {
            parts.push(largest(&c.abs(&c.sub(gx, earlier)?)?)?);
        }
        let values = c
            .cat(&parts.iter().collect::<Vec<_>>(), 0)?
            .try_to_vec::<f64>()?;
        let &[head_m, head_ratio, transfer_m3, transfer_ratio, flags, ..] = values.as_slice()
        else {
            return Err(SurfaceChannelError::Numerical(
                "iteration residual read lacks a maximum".into(),
            ));
        };
        // Equal values can still differ in signed zeros, so a repeat is confirmed bit for bit.
        let repeats = match (earlier, values.get(5)) {
            (Some(earlier), Some(&0.0)) => same_bits(gx, earlier)?,
            _ => false,
        };
        if !head_m.is_finite() || !transfer_m3.is_finite() || !flags.is_finite() {
            return Err(SurfaceChannelError::Numerical(
                "iteration residual is nonfinite: rescale heads or link volumes".into(),
            ));
        }
        let norm = head_ratio.max(transfer_ratio);
        let residuals = Residuals {
            head_m,
            transfer_m3,
            norm,
            converged: head_ratio <= 1.0 && transfer_ratio <= 1.0 && flags == 0.0,
        };
        Ok((residuals, repeats))
    }

    /// Solves the attempt by Anderson iteration with no history.
    ///
    /// The levelling operands come from one zero-exchange completion before the first update.
    /// The root's own evaluation is the candidate. An attempt that cannot converge rejects at
    /// once and records its [`Stall`] in `stalled`:
    /// - An iterate that repeats the one two updates earlier: the update-limit rejection it
    ///   would reach, with contraction one.
    /// - A residual forecast past the update limit: see [`stall`].
    #[allow(clippy::too_many_arguments)]
    pub fn solve(
        &self,
        config: &SurfaceChannelConfig,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
        stalled: &Cell<Option<Stall>>,
    ) -> Result<Solved<R>, SurfaceChannelError> {
        let levelling = self.levelling(surface, reaches, reach_ids, start, attempt)?;
        let error = RefCell::new(None);
        let record = |e: SurfaceChannelError| -> numr::error::Error {
            if let Ok(mut slot) = error.try_borrow_mut()
                && slot.is_none()
            {
                *slot = Some(e);
            }
            numr::error::Error::InvalidArgument {
                arg: "surface/channel fixed-point map",
                reason: "a child or link law rejected the candidate".into(),
            }
        };
        // The map keeps its latest evaluation with its iterate. With no history and unit
        // mixing, the root is the last iterate, so its evaluation is the accepted candidate.
        let latest: RefCell<Option<(Tensor<R>, Evaluation<R>)>> = RefCell::new(None);
        let checked: RefCell<Option<Residuals>> = RefCell::new(None);
        let map = |x: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
            let evaluation = self
                .evaluate(surface, reaches, reach_ids, start, attempt, &levelling, x)
                .map_err(record)?;
            let packet = evaluation.packet.clone();
            *latest.borrow_mut() = Some((x.clone(), evaluation));
            Ok(packet)
        };
        let options = AndersonOptions {
            m: 0,
            alpha: 1.0,
            base: RootOptions {
                max_iter: config.iteration_limit,
                ..RootOptions::default()
            },
        };
        // The iterate before `x`. The map is deterministic, so a next iterate equal to it in
        // every bit starts a cycle of unconverged iterates that lasts to the update limit.
        let before: RefCell<Option<Tensor<R>>> = RefCell::new(None);
        let norms: RefCell<Vec<f64>> = RefCell::new(Vec::new());
        let root =
            self.client
                .anderson_with_convergence(map, &start.initial, &options, |x, gx, _| {
                    let earlier = before.replace(Some(x.clone()));
                    let (residual, repeats) = self
                        .residuals_against(config, x, gx, earlier.as_ref())
                        .map_err(record)?;
                    let mut norms = norms.borrow_mut();
                    norms.push(residual.norm);
                    if !residual.converged {
                        // A cycle never contracts.
                        let stop = if repeats {
                            let cycle = Stall {
                                updates: norms.len() - 1,
                                contraction: 1.0,
                            };
                            Some((cycle, exhausted(config)))
                        } else {
                            stall(&norms, config.iteration_limit)
                                .map(|forecast| (forecast, forecast.rejection(config)))
                        };
                        if let Some((forecast, rejection)) = stop {
                            stalled.set(Some(forecast));
                            return Err(record(rejection));
                        }
                    }
                    let decision = AndersonConvergence {
                        residual_norm: residual.norm,
                        converged: residual.converged,
                    };
                    *checked.borrow_mut() = Some(residual);
                    Ok(decision)
                });
        if let Some(error) = error.into_inner() {
            return Err(error);
        }
        let root = root.map_err(|e| SurfaceChannelError::Numerical(e.to_string()))?;
        if !root.converged {
            return Err(exhausted(config));
        }
        // Reevaluation repeats the same calculation on the same operands, so it runs only
        // when the root is not the latest evaluated iterate.
        let (evaluation, residuals) = match (latest.into_inner(), checked.into_inner()) {
            (Some((x, evaluation)), Some(residuals)) if x.shares_storage_with(&root.x) => {
                (evaluation, residuals)
            }
            _ => {
                let evaluation = self.evaluate(
                    surface, reaches, reach_ids, start, attempt, &levelling, &root.x,
                )?;
                let residuals = self.residuals(config, &root.x, &evaluation.packet)?;
                (evaluation, residuals)
            }
        };
        if !residuals.converged {
            return Err(SurfaceChannelError::Numerical(
                "final reevaluation exceeds head or transfer budgets, or changes support or regime: shorten the common interval".into(),
            ));
        }
        self.check_crossing(config, &evaluation)?;
        let norms = norms.into_inner();
        let contraction = match norms.as_slice() {
            [.., earlier, _, latest] if *earlier > 0.0 => (latest / earlier).sqrt(),
            _ => 0.0,
        };
        Ok(Solved {
            evaluation,
            iterations: root.iterations,
            residuals,
            contraction,
        })
    }

    /// Rejects a link whose receiver ends above its donor beyond the head budget.
    ///
    /// Heads below the crest count as the crest, so a stopped overflow never crosses.
    fn check_crossing(
        &self,
        config: &SurfaceChannelConfig,
        evaluation: &Evaluation<R>,
    ) -> Result<(), SurfaceChannelError> {
        let c = &self.client;
        let n = self.surface_count + self.slot_count;
        let end = evaluation.packet.narrow(0, 0, n)?;
        let head = |index: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.maximum(&c.index_select(&end, 0, index)?, &self.crest)?)
        };
        let surface = head(&self.link_a)?;
        let channel = head(&self.link_b)?;
        let signed = &evaluation.exchange.signed;
        let zero = self.scalar(0.0)?;
        let allowance = |receiver: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.add_scalar(
                &c.mul_scalar(&c.abs(receiver)?, config.head_relative)?,
                config.head_absolute_m,
            )?)
        };
        let crossed =
            |donor: &Tensor<R>, receiver: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
                Ok(c.lt(&c.add(donor, &allowance(receiver)?)?, receiver)?)
            };
        let forward = c.mul(&c.gt(signed, &zero)?, &crossed(&surface, &channel)?)?;
        let backward = c.mul(&c.lt(signed, &zero)?, &crossed(&channel, &surface)?)?;
        self.reject(
            &c.maximum(&forward, &backward)?,
            "link heads cross beyond the head budget over the interval: shorten the common interval",
        )
    }
}

/// Measured contraction of an attempt that cannot converge within its update limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Stall {
    /// Updates completed when the forecast fails.
    pub updates: usize,
    /// Contraction per update: `sqrt(r_k / r_{k-2})`.
    pub contraction: f64,
}

impl Stall {
    /// The retryable rejection that ends the attempt.
    fn rejection(&self, config: &SurfaceChannelConfig) -> SurfaceChannelError {
        SurfaceChannelError::Numerical(format!(
            "fixed-point iteration contracts by {} per update after {} updates and cannot converge within {} updates: shorten the common interval",
            self.contraction, self.updates, config.iteration_limit
        ))
    }
}

/// Forecast of the residual norms `norms` of one attempt, one per completed update and the start.
///
/// `r_k` is the latest norm and `k` the updates so far. The contraction `ρ = sqrt(r_k / r_{k-2})`
/// spans two updates, so a 2-cycle shows its true rate. Convergence needs `r <= 1`, so it takes
/// `ln(r_k) / -ln(ρ)` more updates. A forecast past `limit` updates stalls.
fn stall(norms: &[f64], limit: usize) -> Option<Stall> {
    let [.., earlier, _, latest] = norms else {
        return None;
    };
    let updates = norms.len() - 1;
    if !(*earlier > 0.0 && *latest > 1.0) {
        return None;
    }
    let contraction = (latest / earlier).sqrt();
    let needed = latest.ln() / -contraction.ln();
    let stalls = contraction >= 1.0 || updates as f64 + needed > limit as f64;
    stalls.then_some(Stall {
        updates,
        contraction,
    })
}

/// The rejection of an iteration that reaches its update limit unconverged.
fn exhausted(config: &SurfaceChannelConfig) -> SurfaceChannelError {
    SurfaceChannelError::Numerical(format!(
        "fixed-point iteration exhausted {} updates: shorten the common interval",
        config.iteration_limit
    ))
}

/// True when `a` and `b` hold the same bits, signed zeros included.
fn same_bits<R: Runtime<DType = DType>>(
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<bool, SurfaceChannelError> {
    let bits = |t: &Tensor<R>| -> Result<Vec<u64>, SurfaceChannelError> {
        Ok(t.contiguous()?
            .try_to_vec::<f64>()?
            .iter()
            .map(|v| v.to_bits())
            .collect())
    };
    Ok(bits(a)? == bits(b)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stall_needs_three_norms_above_the_budget() {
        assert_eq!(stall(&[1e9, 1e9], 64), None);
        assert_eq!(stall(&[1e9, 1e8, 0.5], 64), None);
        assert_eq!(stall(&[0.0, 1e8, 1e8], 64), None);
    }

    #[test]
    fn a_two_cycle_stalls_at_its_two_update_rate() {
        // One-step ratios alternate around one; the two-step ratio stays at one.
        let found = stall(&[3.0e9, 3.1e9, 3.0e9], 64);
        assert_eq!(
            found,
            Some(Stall {
                updates: 2,
                contraction: 1.0
            })
        );
    }

    #[test]
    fn forecast_compares_needed_updates_with_the_limit() {
        // ρ = 0.5 from 1e6 needs about 20 more updates.
        let norms = [4e6, 2e6, 1e6];
        assert_eq!(stall(&norms, 64), None);
        let found = stall(&norms, 21);
        assert_eq!(found.map(|s| s.contraction), Some(0.5));
        assert_eq!(stall(&norms, 22), None);
    }
}
