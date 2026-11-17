//! Fixed-point iteration over end stages and signed link volumes, with separate budgets.

use super::context::{Context, Ops};
use super::evaluate::{Attempt, Evaluation};
use super::start::Start;
use super::{SurfaceChannelConfig, SurfaceChannelError};
use crate::channel::ChannelReach;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, IndexingOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::AndersonAlgorithms;
use solvr::optimize::roots::{AndersonConvergence, AndersonOptions, RootOptions};
use std::cell::RefCell;

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
        let c = &self.client;
        let n = self.surface_count + self.slot_count;
        let l = self.link_count;
        let change = c.abs(&c.sub(gx, x)?)?;
        let largest = |t: &Tensor<R>| -> Result<f64, SurfaceChannelError> {
            Ok(c.max(t, &[0], false)?.item::<f64>()?)
        };
        let budget = |start: usize, len: usize, absolute: f64, relative: f64| {
            let allowance = c.add_scalar(
                &c.mul_scalar(&c.abs(&gx.narrow(0, start, len)?)?, relative)?,
                absolute,
            )?;
            let part = change.narrow(0, start, len)?;
            Ok::<_, SurfaceChannelError>((largest(&part)?, largest(&c.div(&part, &allowance)?)?))
        };
        let (head_m, head_ratio) = budget(0, n, config.head_absolute_m, config.head_relative)?;
        let (transfer_m3, transfer_ratio) =
            budget(n, l, config.transfer_absolute_m3, config.transfer_relative)?;
        let flags = largest(&change.narrow(0, n + l, n + l)?)?;
        if !head_m.is_finite() || !transfer_m3.is_finite() || !flags.is_finite() {
            return Err(SurfaceChannelError::Numerical(
                "iteration residual is nonfinite: rescale heads or link volumes".into(),
            ));
        }
        let norm = head_ratio.max(transfer_ratio);
        Ok(Residuals {
            head_m,
            transfer_m3,
            norm,
            converged: head_ratio <= 1.0 && transfer_ratio <= 1.0 && flags == 0.0,
        })
    }

    /// Solves the attempt by Anderson iteration with no history, then reevaluates the root.
    pub fn solve(
        &self,
        config: &SurfaceChannelConfig,
        surface: &Solver<R>,
        reaches: &[ChannelReach<R>],
        reach_ids: &[&str],
        start: &Start<R>,
        attempt: &Attempt<R>,
    ) -> Result<Solved<R>, SurfaceChannelError> {
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
        let map = |x: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
            self.evaluate(surface, reaches, reach_ids, start, attempt, x)
                .map(|evaluation| evaluation.packet)
                .map_err(record)
        };
        let options = AndersonOptions {
            m: 0,
            alpha: 1.0,
            base: RootOptions {
                max_iter: config.iteration_limit,
                ..RootOptions::default()
            },
        };
        let root =
            self.client
                .anderson_with_convergence(map, &start.initial, &options, |x, gx, _| {
                    self.residuals(config, x, gx)
                        .map(|residual| AndersonConvergence {
                            residual_norm: residual.norm,
                            converged: residual.converged,
                        })
                        .map_err(record)
                });
        if let Some(error) = error.into_inner() {
            return Err(error);
        }
        let root = root.map_err(|e| SurfaceChannelError::Numerical(e.to_string()))?;
        if !root.converged {
            return Err(SurfaceChannelError::Numerical(format!(
                "fixed-point iteration exhausted {} updates: shorten the common interval",
                config.iteration_limit
            )));
        }
        let evaluation = self.evaluate(surface, reaches, reach_ids, start, attempt, &root.x)?;
        let residuals = self.residuals(config, &root.x, &evaluation.packet)?;
        if !residuals.converged {
            return Err(SurfaceChannelError::Numerical(
                "final reevaluation exceeds head or transfer budgets, or changes support or regime: shorten the common interval".into(),
            ));
        }
        self.check_crossing(config, &evaluation)?;
        Ok(Solved {
            evaluation,
            iterations: root.iterations,
            residuals,
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
