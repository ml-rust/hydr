//! Certified increasing roots of surveyed trace equations in normalized depth `t = d / H`.

use super::{
    client::TraceClient,
    error::trace_error,
    interior::{Interior, bounds},
    point::{Evaluation, TraceDepth, TracePoint},
};
use crate::channel::flow::{
    ChannelError,
    boundary::{Bounds, Reservoir, ReservoirHeads},
    context::Context,
    surveyed::{error::checked, work::QueryWork},
};
use numr::{dtype::DType, ops::BinaryOps, runtime::Runtime, tensor::Tensor};
use solvr::optimize::roots::CertifiedBisectionAlgorithms;
use solvr::{
    common::arithmetic::Interval,
    optimize::roots::{CertifiedRootStatus, CertifiedRootValues},
};
use std::cell::Cell;

/// Certified normalized-depth root enclosure and the midpoint rounds that produced it.
#[derive(Clone, Copy, Debug)]
pub(super) struct RootEnclosure {
    pub lower: f64,
    pub upper: f64,
    pub iterations: usize,
}

/// Residual enclosures of one increasing trace equation at exact normalized depths.
pub(super) type Residual<'a, R> = dyn Fn(&Tensor<R>) -> Result<Interval<R>, ChannelError> + 'a;

/// Outcome of one certified bisection over a closed normalized-depth bracket.
pub(super) enum RootOutcome {
    /// Certified endpoint signs prove one root inside the enclosure.
    Root(RootEnclosure),
    /// Certified endpoint signs prove no root in the bracket.
    NoRoot,
    /// An endpoint residual straddles zero, so no root is established.
    Unresolved,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TraceClient<R>,
{
    /// The trace state at exact normalized depths `t`.
    pub(super) fn surveyed_trace_at(
        &self,
        interior: &Interior<R>,
        t: &Tensor<R>,
        evaluation: &Evaluation<'_>,
    ) -> Result<TracePoint<R>, ChannelError> {
        let c = interior.sources.geometry.client();
        let t = checked(c, "surveyed trace normalized depth", |a| Ok(a.point(t)?))
            .map_err(trace_error)?;
        self.surveyed_trace_point(interior, TraceDepth::Normalized(&t), evaluation)
    }

    /// `S = c_b - u_b`, increasing in depth.
    pub(super) fn surveyed_sonic_residual(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
    ) -> Result<Interval<R>, ChannelError> {
        let c = interior.sources.geometry.client();
        checked(c, "surveyed trace sonic residual", |a| {
            Ok(a.sub(&point.sound, &point.velocity)?)
        })
        .map_err(trace_error)
    }

    /// `-G = q - A u_b`, increasing in depth above the outflow sonic depth.
    pub(super) fn surveyed_discharge_residual(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
        target: &Tensor<R>,
    ) -> Result<Interval<R>, ChannelError> {
        let c = interior.sources.geometry.client();
        checked(c, "surveyed trace discharge residual", |a| {
            let water = a.mul(&point.area, &point.velocity)?;
            let target = a.point(target)?;
            Ok(a.sub(&target, &water)?)
        })
        .map_err(trace_error)
    }

    /// `E = d + (1 + k) max(-u_b, 0)² / (2 g) - (η - z_min)`, with `dE/dd >= 1`.
    ///
    /// Leaving flow carries no velocity head: it mixes completely in the reservoir.
    pub(super) fn surveyed_reservoir_residual(
        &self,
        interior: &Interior<R>,
        point: &TracePoint<R>,
        reservoir: &Reservoir<R>,
    ) -> Result<Interval<R>, ChannelError> {
        let depth = bounds(&point.depth);
        let ReservoirHeads {
            velocity_head,
            entrance_loss_head,
            ..
        } = self.reservoir_heads(&bounds(&point.velocity), &reservoir.coefficient)?;
        let total = self.reservoir_add_head(
            &self.reservoir_add_head(&depth, &velocity_head)?,
            &entrance_loss_head,
        )?;
        let residual = self.bsub(&total, &reservoir.bounds)?;
        // Nonnegative heads give E >= d - head.
        let floor = self.bsub(&depth, &reservoir.bounds)?;
        let residual = Bounds {
            lo: self.client.maximum(&residual.lo, &floor.lo)?,
            hi: residual.hi,
        };
        let c = interior.sources.geometry.client();
        checked(c, "surveyed trace reservoir residual", |a| {
            Ok(a.bounds(&residual.lo, &residual.hi)?)
        })
        .map_err(trace_error)
    }

    /// Slope floor of `E` in normalized depth over `[a, b]`, at least `floor = κ_E = H`.
    ///
    /// Entering flow has `u_b < 0` and `dE/dd = 1 + (1 + k) |u_b| |u_b'| / g`. With
    /// `u_b' = -T f_i'(A)` and `A f_i' >= c`, `|u_b'| >= T c / A = g / c`. So
    /// `dE/dd >= 1 + (1 + k) |u_b| / c`. Over the bracket, `|u_b|` is at least the negated
    /// upper velocity bound and `c` at most the upper sound bound. A bracket that admits
    /// `u_b >= 0` keeps `floor`.
    pub(super) fn surveyed_reservoir_slope(
        &self,
        interior: &Interior<R>,
        reservoir: &Reservoir<R>,
        evaluation: &Evaluation<'_>,
        a: f64,
        b: f64,
        floor: f64,
    ) -> Result<f64, ChannelError> {
        let c = interior.sources.geometry.client();
        let (lower, upper) = (self.tensor(&[a])?, self.tensor(&[b])?);
        let t = checked(c, "surveyed reservoir slope bracket", |a| {
            Ok(a.bounds(&lower, &upper)?)
        })
        .map_err(trace_error)?;
        let point = self.surveyed_trace_point(interior, TraceDepth::Normalized(&t), evaluation)?;
        let velocity = point.velocity.upper().item::<f64>()?;
        let sound = point.sound.upper().item::<f64>()?;
        let loss = reservoir.coefficient.item::<f64>()?;
        if !(velocity < 0.0 && sound > 0.0 && sound.is_finite() && loss >= 0.0) {
            return Ok(floor);
        }
        // Each nearest operation moves one step down, so every factor stays a lower bound.
        let froude = (-velocity / sound).next_down();
        let gain = ((1.0 + loss).next_down() * froude).next_down();
        let kappa = (floor * (1.0 + gain).next_down()).next_down();
        Ok(if kappa.is_finite() && kappa > floor {
            kappa
        } else {
            floor
        })
    }

    /// Certified root of the increasing `residual` over `[lower, upper]` in normalized depth.
    ///
    /// `kappa` bounds the residual slope from below over the whole bracket. Solvr narrows
    /// each enclosure to `root ± max(R, 0) / kappa`. A straddling endpoint residual proves no
    /// root and returns `Unresolved`.
    pub(super) fn surveyed_trace_root(
        &self,
        evaluation: &Evaluation<'_>,
        lower: f64,
        upper: f64,
        kappa: f64,
        residual: &Residual<'_, R>,
    ) -> Result<RootOutcome, ChannelError> {
        let c = &self.client;
        let (lo, hi, slope) = (
            self.tensor(&[lower])?,
            self.tensor(&[upper])?,
            self.tensor(&[kappa])?,
        );
        evaluation.work.begin_root().map_err(trace_error)?;
        let cause = Cell::new(None);
        let result = c.bisect_certified(
            |query| {
                let value = evaluation
                    .work
                    .charge(QueryWork::Residual, query.point.numel())
                    .map_err(trace_error)
                    .and_then(|()| residual(&query.point));
                match value {
                    Ok(value) => Ok(CertifiedRootValues {
                        lower: value.lower().clone(),
                        upper: value.upper().clone(),
                    }),
                    Err(error) => {
                        let first = cause.take().unwrap_or(error);
                        cause.set(Some(first));
                        Err(numr::error::Error::InvalidArgument {
                            arg: "surveyed trace residual",
                            reason: "trace evaluation rejected: read the retained channel error"
                                .into(),
                        })
                    }
                }
            },
            &lo,
            &hi,
            &slope,
            &evaluation.config.root,
        );
        if let Some(error) = cause.take() {
            return Err(error);
        }
        let result = result.map_err(|error| {
            ChannelError::numerical(format!(
                "scalar channel boundary: surveyed trace certified root: {error}. Increase normalized_root limits or rescale inputs"
            ))
        })?;
        evaluation
            .work
            .finish_root(result.iterations())
            .map_err(trace_error)?;
        let code = result.status().item::<i64>()?;
        Ok(match CertifiedRootStatus::from_code(code) {
            Some(
                CertifiedRootStatus::Converged
                | CertifiedRootStatus::AccuracyFloor
                | CertifiedRootStatus::WorkLimit,
            ) => RootOutcome::Root(RootEnclosure {
                lower: result.root_lower().item::<f64>()?,
                upper: result.root_upper().item::<f64>()?,
                iterations: result.iterations(),
            }),
            Some(CertifiedRootStatus::NoRoot) => RootOutcome::NoRoot,
            Some(CertifiedRootStatus::UnresolvedBracket) => RootOutcome::Unresolved,
            None => {
                return Err(ChannelError::numerical(format!(
                    "scalar channel boundary: surveyed trace root status={code} is unknown: inspect the Solvr root contract"
                )));
            }
        })
    }
}
