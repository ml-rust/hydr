//! Coordinator stage API: prepare one surface interval, then complete it with an exchange.

use super::super::Solver;
use super::super::definition::Definition;
use super::super::stepping::{Completion, IntervalTrial, Preparation, PreparedInterval};
use super::{ExchangeSource, PortFluxes, PortGhosts, SurfaceLinks, SurfacePorts};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

/// One prepared surface interval: post-source state, step limits, and linked donor state.
///
/// Belongs to the engine that prepared it, at that engine's accepted time and step count.
#[doc(hidden)]
pub struct SurfaceCouplingStage<R: Runtime<DType = DType>> {
    definition: Arc<Definition>,
    start: f64,
    steps: usize,
    prepared: PreparedInterval<R>,
    links: SurfaceLinks<R>,
    available_volume_m3: Tensor<R>,
    velocity_xy: [Tensor<R>; 2],
}

impl<R: Runtime<DType = DType>> SurfaceCouplingStage<R> {
    /// Returns the accepted start time in seconds.
    pub fn start(&self) -> f64 {
        self.start
    }

    /// Returns the trial end time in seconds.
    pub fn end(&self) -> f64 {
        self.prepared.trial.end
    }

    /// Returns the trial step `end - start` in seconds.
    pub fn dt(&self) -> f64 {
        self.prepared.trial.dt
    }

    /// Returns the retry count the accepted step records.
    pub fn retry(&self) -> usize {
        self.prepared.trial.retry
    }

    /// Returns the CFL step limit in seconds. Infinity denotes no wave restriction.
    pub fn cfl_limit(&self) -> f64 {
        self.prepared.cfl_limit
    }

    /// Returns the donor depletion step limit in seconds. Infinity denotes no outflow.
    pub fn depletion_limit(&self) -> f64 {
        self.prepared.depletion_limit
    }

    /// Returns the smaller of the CFL and depletion limits. Completion requires `dt() <= limit()`.
    pub fn limit(&self) -> f64 {
        self.prepared.limit()
    }

    /// Returns the full-grid post-source depth in metres.
    pub fn post_source_depth(&self) -> &Tensor<R> {
        &self.prepared.sources.state.h
    }

    /// Returns the full-grid post-source x/y discharge per width in m²/s.
    pub fn post_source_discharge(&self) -> [&Tensor<R>; 2] {
        let state = &self.prepared.sources.state;
        [&state.qx, &state.qy]
    }

    /// Returns the linked cells the per-link tensors cover.
    pub fn links(&self) -> &SurfaceLinks<R> {
        &self.links
    }

    /// Returns each linked cell's post-transport volume in m³, the donor limit for withdrawal.
    ///
    /// A cell that transport drives negative reports zero. Completion then asks for a retry.
    pub fn available_volume_m3(&self) -> &Tensor<R> {
        &self.available_volume_m3
    }

    /// Returns each linked cell's post-transport x/y velocity in m/s. Dry cells report zero.
    pub fn velocity_xy(&self) -> [&Tensor<R>; 2] {
        let [x, y] = &self.velocity_xy;
        [x, y]
    }

    /// Returns the port face and per-port flux receipts. A stage without ports returns `None`.
    ///
    /// The fluxes come from the post-source surface state and the trial's ghost states.
    /// Their speeds are in [`Self::cfl_limit`] and their outflow in [`Self::depletion_limit`].
    pub fn port_fluxes(&self) -> Option<&PortFluxes<R>> {
        self.prepared.flux.ports.as_ref()
    }
}

/// Port faces with the ghost states of one trial.
pub type CouplingPorts<'a, R> = (&'a SurfacePorts<R>, &'a PortGhosts<R>);

/// Outcome of [`Solver::coupling_prepare`]. A retry variant asks the coordinator to shorten the step.
#[doc(hidden)]
pub enum CouplingPreparation<R: Runtime<DType = DType>> {
    /// The stage is ready for the coordinator's exchange.
    Ready(Box<SurfaceCouplingStage<R>>),
    /// A land source or a retryable soil error rejects this step.
    SourceRetry(EngineError),
    /// A boundary error in the face fluxes rejects this step.
    BoundaryRetry(EngineError),
}

/// Outcome of [`Solver::coupling_complete`].
#[doc(hidden)]
pub enum CouplingCompletion<R: Runtime<DType = DType>> {
    /// A new engine holds the accepted step. The calling engine stays unchanged.
    Accepted(Box<Solver<R>>),
    /// Transport creates negative depth or dry momentum. The coordinator shortens the step.
    Retry(String),
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
    /// Prepares the interval from the accepted time to `end` for a coordinator exchange.
    ///
    /// Runs the forcing lookup, sources, face fluxes, and step limits of a standalone trial.
    /// `retry` is the retry count the accepted step records.
    ///
    /// `ports` replaces the missing side of each port face with its ghost state before the
    /// HLL flux. Port face speeds enter the CFL limit and port outflow the depletion limit.
    /// Completion counts port water in the exchange totals.
    ///
    /// # Errors
    /// Rejects links or ports for another grid, ghosts for other ports, an `end` outside the
    /// covering forcing interval or past `max_step`, a `retry` above `max_retries`, a retried
    /// step below `min_step`, and every non-retryable source, transport, CFL, and depletion error.
    #[doc(hidden)]
    pub fn coupling_prepare(
        &self,
        end: f64,
        retry: usize,
        links: &SurfaceLinks<R>,
        ports: Option<CouplingPorts<'_, R>>,
    ) -> Result<CouplingPreparation<R>, EngineError> {
        let config = self.config();
        let time = self.time();
        links.check_grid(self.domain().grid().cell_count())?;
        if let Some((ports, ghosts)) = ports {
            ports.check(self)?;
            ghosts.check(ports)?;
        }
        let (index, forcing) = self.forcing_interval()?;
        let dt = end - time;
        if !end.is_finite()
            || !dt.is_finite()
            || dt <= 0.0
            || end > forcing.end
            || end > time + config.max_step
            || retry > config.max_retries
            || (retry > 0 && dt < config.min_step)
        {
            return Err(EngineError::rejected(
                "coupling interval",
                format!(
                    "time={time}, end={end}, retry={retry}, forcing_end={}, max_step={}, min_step={}, max_retries={}: choose an end inside the covering forcing interval and the step controls",
                    forcing.end, config.max_step, config.min_step, config.max_retries
                ),
            ));
        }
        let trial = IntervalTrial {
            index,
            end,
            dt,
            retry,
        };
        let ports = ports.map(|(ports, ghosts)| (ports.faces.as_ref(), &ghosts.state));
        let prepared = match self.prepare_interval(trial, ports)? {
            Preparation::Ready(prepared) => *prepared,
            Preparation::SourceRetry(error) => return Ok(CouplingPreparation::SourceRetry(error)),
            Preparation::BoundaryRetry(error) => {
                return Ok(CouplingPreparation::BoundaryRetry(error));
            }
        };
        let client = &self.context.client;
        let zero = &self.context.zero;
        let transported = self.transported(&prepared.sources.state, &prepared.flux, dt)?;
        let wet = client.gt(&transported.h, zero)?;
        let depth = client.where_cond(&wet, &transported.h, zero)?;
        let donor = client.where_cond(&wet, &transported.h, &client.add_scalar(zero, 1.0)?)?;
        let gather = |values: &Tensor<R>| client.index_select(values, 0, &links.ids);
        let velocity = |q: &Tensor<R>| -> Result<Tensor<R>, EngineError> {
            let velocity = client.where_cond(&wet, &client.div(q, &donor)?, zero)?;
            Ok(gather(&velocity)?)
        };
        let available_volume_m3 =
            gather(&client.mul_scalar(&depth, self.domain().grid().cell_area())?)?;
        let velocity_xy = [velocity(&transported.qx)?, velocity(&transported.qy)?];
        Ok(CouplingPreparation::Ready(Box::new(SurfaceCouplingStage {
            definition: Arc::clone(&self.definition),
            start: time,
            steps: self.accepted.steps,
            prepared,
            links: links.clone(),
            available_volume_m3,
            velocity_xy,
        })))
    }

    /// Completes `stage` with an optional exchange and returns a new engine at the stage end.
    ///
    /// `None` runs the standalone completion. The calling engine stays unchanged.
    /// The new engine shares this engine's definition and device context.
    ///
    /// # Errors
    /// Rejects a stage from another engine state, a step above the stage limit, exchange
    /// operands for another grid, and every non-retryable completion and exchange error.
    #[doc(hidden)]
    pub fn coupling_complete(
        &self,
        stage: &SurfaceCouplingStage<R>,
        exchange: Option<&ExchangeSource<R>>,
    ) -> Result<CouplingCompletion<R>, EngineError> {
        if !Arc::ptr_eq(&stage.definition, &self.definition)
            || stage.start.to_bits() != self.time().to_bits()
            || stage.steps != self.accepted.steps
        {
            return Err(EngineError::rejected(
                "coupling stage",
                format!(
                    "stage start={}, steps={}; engine time={}, steps={}: prepare the stage from this engine",
                    stage.start,
                    stage.steps,
                    self.time(),
                    self.accepted.steps
                ),
            ));
        }
        if stage.dt() > stage.limit() {
            return Err(EngineError::rejected(
                "coupling step limit",
                format!(
                    "dt={}, CFL limit={}, depletion limit={}: prepare a step at or below the stage limit",
                    stage.dt(),
                    stage.cfl_limit(),
                    stage.depletion_limit()
                ),
            ));
        }
        match self.complete_interval(stage.prepared.clone(), exchange)? {
            Completion::Accepted(accepted) => Ok(CouplingCompletion::Accepted(Box::new(Self {
                definition: Arc::clone(&self.definition),
                accepted: *accepted,
                context: Arc::clone(&self.context),
            }))),
            Completion::Retry(reason) => Ok(CouplingCompletion::Retry(reason)),
        }
    }

    /// Returns accepted water-surface elevation in metres and a wet mask at each linked cell.
    ///
    /// The mask holds `1.0` for positive depth and `0.0` for a dry cell.
    ///
    /// # Errors
    /// Rejects links for another grid and returns runtime errors.
    #[doc(hidden)]
    pub fn coupling_cell_stages(
        &self,
        links: &SurfaceLinks<R>,
    ) -> Result<(Tensor<R>, Tensor<R>), EngineError> {
        links.check_grid(self.domain().grid().cell_count())?;
        let client = &self.context.client;
        let h = &self.accepted.flow.h;
        let gather = |values: &Tensor<R>| client.index_select(values, 0, &links.ids);
        let stage = gather(&client.add(&self.context.bed, h)?)?;
        let wet = client.cast(&gather(&client.gt(h, &self.context.zero)?)?, DType::F64)?;
        Ok((stage, wet))
    }
}
