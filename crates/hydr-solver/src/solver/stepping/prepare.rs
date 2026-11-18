//! Source phase of one accepted-interval trial: every step before the transport update.

use super::super::Solver;
use super::super::tensor::land_sources::{
    SourceInputs, SourceTrial, calculate as calculate_sources,
};
use super::super::tensor::transport::{self, PortGhostFaces, TensorTransport};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// One trial interval from the accepted time.
#[derive(Clone, Copy, Debug)]
pub(in crate::solver) struct IntervalTrial {
    /// Index of the forcing interval that covers the accepted time.
    pub(in crate::solver) index: usize,
    /// Trial end time.
    pub(in crate::solver) end: f64,
    /// Trial step, `end - time`.
    pub(in crate::solver) dt: f64,
    /// Retry count of this trial within its accepted step.
    pub(in crate::solver) retry: usize,
}

/// Trial state after sources and face fluxes, before the transport update.
#[derive(Clone)]
pub(in crate::solver) struct PreparedInterval<R: Runtime<DType = DType>> {
    pub(in crate::solver) trial: IntervalTrial,
    /// Soil and land sources with the post-source state.
    pub(in crate::solver) sources: SourceTrial<R>,
    /// HLL face fluxes, boundary rates, and the transport derivative.
    pub(in crate::solver) flux: TensorTransport<R>,
    /// Largest step the CFL condition admits.
    pub(in crate::solver) cfl_limit: f64,
    /// Largest step that no donor cell exports more than its available volume.
    pub(in crate::solver) depletion_limit: f64,
}

impl<R: Runtime<DType = DType>> PreparedInterval<R> {
    /// Largest step the CFL and depletion limits admit together.
    pub(in crate::solver) fn limit(&self) -> f64 {
        self.cfl_limit.min(self.depletion_limit)
    }
}

/// Source-phase outcome. A retry variant asks the caller to halve the step.
pub(in crate::solver) enum Preparation<R: Runtime<DType = DType>> {
    /// Sources and fluxes are ready for the step limits and completion.
    Ready(Box<PreparedInterval<R>>),
    /// A land source or a retryable soil error rejects this step.
    SourceRetry(EngineError),
    /// A boundary error in the face fluxes rejects this step.
    BoundaryRetry(EngineError),
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Runs sources, post-source checks, face fluxes, and the CFL and depletion limits.
    ///
    /// Reads only the accepted state, the forcing interval at `trial.index`, `trial.dt`, and
    /// the coordinator port ghosts. Port face speeds and outflow enter both limits.
    ///
    /// # Errors
    /// Returns non-retryable source, transport, CFL, and depletion errors.
    pub(in crate::solver) fn prepare_interval(
        &self,
        trial: IntervalTrial,
        ports: Option<PortGhostFaces<'_, R>>,
    ) -> Result<Preparation<R>, EngineError> {
        let config = self.config();
        let client = &self.context.client;
        let time = self.time();
        let source = self.context.sources.get(trial.index).ok_or_else(|| {
            EngineError::rejected(
                "forcing",
                format!(
                    "time={time}, index={}: no device source covers this interval; rebuild the solver",
                    trial.index
                ),
            )
        })?;
        let zero = &self.context.zero;
        let one = client.add_scalar(zero, 1.0)?;
        let area = self.domain().grid().cell_area();
        let sources = match calculate_sources(
            &self.context,
            SourceInputs {
                original: &self.accepted.flow,
                land: &self.accepted.land,
                forcing: source,
            },
            area,
            trial.dt,
            self.ledger().water_adjustment_abs_bound,
            config,
        ) {
            Ok(sources) => sources,
            Err(error @ EngineError::LandSource { .. })
            | Err(
                error @ EngineError::Soil {
                    retryable: true, ..
                },
            ) => return Ok(Preparation::SourceRetry(error)),
            Err(error) => return Err(error),
        };
        let state = &sources.state;
        self.finite(&state.qx, "post-source qx")?;
        self.finite(&state.qy, "post-source qy")?;
        let flux = match transport::calculate(&self.context, state, source, config, area, ports) {
            Ok(flux) => flux,
            Err(error @ EngineError::Boundary { .. }) => {
                return Ok(Preparation::BoundaryRetry(error));
            }
            Err(error) => return Err(error),
        };
        let (dx, dy) = self.domain().grid().spacing();
        let frequency = flux.speeds[0] / dx + flux.speeds[1] / dy;
        if !frequency.is_finite() {
            return Err(EngineError::rejected(
                "CFL",
                format!("frequency={frequency}: rescale geometry and velocity"),
            ));
        }
        let cfl_limit = if frequency > 0.0 {
            config.cfl / frequency
        } else {
            f64::INFINITY
        };
        self.finite(&flux.outgoing, "outgoing rate")?;
        let exporting = client.gt(&flux.outgoing, zero)?;
        if self.any(&client.mul(&client.eq(&state.h, zero)?, &exporting)?)? {
            return Err(EngineError::rejected(
                "depletion",
                format!(
                    "time={time}: a dry donor exports water, correct the source-complete port or numerical state"
                ),
            ));
        }
        let denominator = client.where_cond(&exporting, &flux.outgoing, &one)?;
        let bounds = client.div(&client.mul_scalar(&state.h, area)?, &denominator)?;
        let unlimited = client.add_scalar(zero, f64::INFINITY)?;
        let bounds = client.where_cond(&exporting, &bounds, &unlimited)?;
        let depletion_limit = client.min(&bounds, &[0], false)?.item::<f64>()?;
        Ok(Preparation::Ready(Box::new(PreparedInterval {
            trial,
            sources,
            flux,
            cfl_limit,
            depletion_limit,
        })))
    }
}
