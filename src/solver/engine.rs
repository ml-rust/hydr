use super::{
    BoundaryConditions, CONTRACT_VERSION, CellResult, EngineError, ForcingSchedule, InitialState,
    Snapshot, SolverConfig, SolverSetup, StepDiagnostics, WaterLedger,
};
use crate::model::{CellState, Domain, FlowState};
use crate::soil::{LandModel, SoilState};
use crate::tensor::land::TensorLand;
use crate::tensor::land_sources::checked_volume;
use crate::tensor::{TensorDomain, TensorState};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Definition {
    pub domain: Domain,
    pub land: LandModel,
    pub boundaries: BoundaryConditions,
    pub forcing: ForcingSchedule,
    pub config: SolverConfig,
}

#[derive(Clone)]
pub(super) struct Accepted<R: Runtime<DType = DType>> {
    pub time: f64,
    pub land: TensorLand<R>,
    pub ledger: WaterLedger,
    pub steps: usize,
    pub last: Option<super::receipts::StepReceipt<R>>,
    pub flow: TensorState<R>,
}

/// Why an advancement call returns at its accepted-state boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdvanceStatus {
    /// The requested elapsed time is reached exactly.
    ReachedTarget,
    /// The caller requests cancellation before the next numerical step.
    Cancelled,
    /// The caller's accepted-step budget is exhausted.
    StepLimit,
}

/// Bounded advancement outcome. An incomplete result retains a complete accepted checkpoint.
#[derive(Clone, Debug, PartialEq)]
pub struct AdvanceReport {
    /// Completion or safe interruption status.
    pub status: AdvanceStatus,
    /// Last accepted elapsed time in seconds.
    pub accepted_time: f64,
    /// Steps accepted during this call.
    pub accepted_steps: usize,
}

/// Runtime-selected shallow-water engine with atomic accepted steps.
pub struct Solver<R: Runtime<DType = DType>> {
    pub(super) definition: Arc<Definition>,
    pub(super) accepted: Accepted<R>,
    pub(super) context: Arc<TensorDomain<R>>,
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
    /// Constructs an engine from explicit physical state and complete bounded forcing.
    ///
    /// # Errors
    /// Rejects incompatible spatial support, initial time, state, boundaries, or numerical controls.
    pub fn new(
        client: R::Client,
        setup: SolverSetup,
        initial: InitialState,
    ) -> Result<Self, EngineError> {
        let SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config,
        } = setup;
        let InitialState {
            time: initial_time,
            flow: state,
            soil,
        } = initial;
        config.check()?;
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(EngineError::rejected(
                "arithmetic environment",
                error.message("surface"),
            ));
        }
        let state = FlowState::new(&domain, state.cells().to_vec())?;
        let boundaries =
            BoundaryConditions::with_open_faces(&domain, boundaries.open_faces().to_vec())?;
        let forcing = ForcingSchedule::new(&domain, &boundaries, forcing.intervals().to_vec())?;
        if !initial_time.is_finite() || forcing.at(initial_time).is_none() {
            return Err(EngineError::rejected(
                "initial time",
                format!("time={initial_time}: supply a finite time inside forcing coverage"),
            ));
        }
        let land = LandModel::new(&domain, land.cells().to_vec())
            .map_err(crate::tensor::land::static_error)?;
        let soil = SoilState::new(&land, soil.cells().to_vec())
            .map_err(crate::tensor::land::static_error)?;
        let land_state = TensorLand::new(client.clone(), &land, &soil)?;
        let context = TensorDomain::<R>::new(client, &domain, &forcing)?;
        let flow_state = TensorState::from_flow(context.client.device(), &state)?;
        let volume = checked_volume(&context.client, &flow_state.h, domain.grid().cell_area())?;
        Ok(Self {
            definition: Arc::new(Definition {
                domain,
                land,
                boundaries,
                forcing,
                config,
            }),
            accepted: Accepted {
                time: initial_time,
                land: land_state,
                ledger: WaterLedger::new(volume),
                steps: 0,
                last: None,
                flow: flow_state,
            },
            context: Arc::new(context),
        })
    }

    /// Returns the immutable checked domain and reference metadata.
    pub fn domain(&self) -> &Domain {
        &self.definition.domain
    }

    /// Returns the explicit physical land support.
    pub fn land(&self) -> &LandModel {
        &self.definition.land
    }

    /// Projects complete accepted native soil memory.
    ///
    /// # Errors
    /// Returns runtime transfer or physical state errors.
    pub fn soil_state(&self) -> Result<SoilState, EngineError> {
        self.accepted.land.soil_state()
    }

    /// Returns the immutable numerical controls.
    pub fn config(&self) -> &SolverConfig {
        &self.definition.config
    }

    /// Returns accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.time
    }

    /// Projects complete accepted depth and discharge fields.
    ///
    /// # Errors
    /// Returns runtime transfer or physical state errors.
    pub fn state(&self) -> Result<FlowState, EngineError> {
        self.accepted.flow.to_flow(self.domain())
    }

    /// Returns the component's cumulative water ledger.
    pub fn ledger(&self) -> &WaterLedger {
        &self.accepted.ledger
    }

    /// Projects the last accepted interval and detailed tensor receipts.
    ///
    /// # Errors
    /// Returns runtime transfer or receipt shape errors.
    pub fn last_step(&self) -> Result<Option<StepDiagnostics>, EngineError> {
        self.accepted
            .last
            .as_ref()
            .map(|receipt| receipt.project(&self.context, &self.accepted.land))
            .transpose()
    }

    /// Captures every continuation-relevant value without exposing mutable aliases.
    pub fn snapshot(&self) -> Snapshot<R> {
        Snapshot {
            context: Arc::clone(&self.context),
            contract: CONTRACT_VERSION,
            definition: Arc::clone(&self.definition),
            accepted: self.accepted.clone(),
        }
    }

    /// Restores complete accepted state after checking the immutable numerical definition.
    ///
    /// # Errors
    /// Rejects different devices, geometry, parameters, boundaries, forcing, or contracts before mutation.
    pub fn restore(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.context.client.device())
            || snapshot.contract != CONTRACT_VERSION
            || !self.definition.matches(
                &snapshot.definition.domain,
                &snapshot.definition.land,
                &snapshot.definition.boundaries,
                &snapshot.definition.forcing,
                &snapshot.definition.config,
            )
        {
            return Err(EngineError::rejected(
                "snapshot",
                "definition or contract mismatch: restore into an identically constructed engine",
            ));
        }
        let mut land = self.accepted.land.clone();
        land.restore_trusted(&snapshot.accepted.land)?;
        let mut accepted = snapshot.accepted.clone();
        accepted.land = land;
        if let Some(super::receipts::StepReceipt::Device(receipt)) = &mut accepted.last {
            receipt.sources.soil.next = accepted.land.clone();
        }
        self.accepted = accepted;
        Ok(())
    }

    /// Advances toward a supported time under a bounded accepted-step budget.
    ///
    /// # Errors
    /// Invalid targets reject before advancement. Numerical errors retain the last accepted step.
    pub fn advance_to(
        &mut self,
        target: f64,
        max_steps: usize,
    ) -> Result<AdvanceReport, EngineError> {
        self.advance_inner(
            target,
            max_steps,
            || false,
            None::<fn(&StepDiagnostics, &[CellState])>,
        )
    }

    /// Advances with cancellation and observations only at safe accepted-state boundaries.
    ///
    /// Callbacks own no numerical state. The observer receives each accepted field without interpolation.
    ///
    /// # Errors
    /// Invalid targets or unresolved trials return actionable errors with the last accepted state intact.
    pub fn advance_controlled(
        &mut self,
        target: f64,
        max_steps: usize,
        mut cancelled: impl FnMut() -> bool,
        mut on_accepted: impl FnMut(&StepDiagnostics, &[CellState]),
    ) -> Result<AdvanceReport, EngineError> {
        self.advance_inner(target, max_steps, &mut cancelled, Some(&mut on_accepted))
    }

    fn advance_inner(
        &mut self,
        target: f64,
        max_steps: usize,
        mut cancelled: impl FnMut() -> bool,
        mut observer: Option<impl FnMut(&StepDiagnostics, &[CellState])>,
    ) -> Result<AdvanceReport, EngineError> {
        let end = self
            .definition
            .forcing
            .intervals()
            .last()
            .map(|interval| interval.end);
        if !target.is_finite() || target < self.time() || end.is_none_or(|end| target > end) {
            return Err(EngineError::rejected(
                "target",
                format!(
                    "target={target}, accepted_time={}: supply a finite nondecreasing target inside forcing coverage",
                    self.time()
                ),
            ));
        }
        let mut count = 0;
        let status = loop {
            if self.time() == target {
                break AdvanceStatus::ReachedTarget;
            }
            if cancelled() {
                break AdvanceStatus::Cancelled;
            }
            if count == max_steps {
                break AdvanceStatus::StepLimit;
            }
            let next = self.next_accepted(target)?;
            let projected = if observer.is_some() {
                let step = next
                    .last
                    .as_ref()
                    .map(|receipt| receipt.project(&self.context, &next.land))
                    .transpose()?;
                Some((step, next.flow.to_flow(self.domain())?))
            } else {
                None
            };
            self.accepted = next;
            count += 1;
            if let (Some(observer), Some((Some(step), state))) = (&mut observer, projected) {
                observer(&step, state.cells());
            }
        };
        Ok(AdvanceReport {
            status,
            accepted_time: self.time(),
            accepted_steps: count,
        })
    }

    /// Projects one cell's accepted physical output without crossing masks or barriers.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn cell(&self, index: usize) -> Result<CellResult, EngineError> {
        if index >= self.domain().grid().cell_count() || !self.domain().active()[index] {
            return Ok(CellResult::Unsupported);
        }
        let cell = CellState {
            h: self.accepted.flow.h.narrow(0, index, 1)?.item::<f64>()?,
            qx: self.accepted.flow.qx.narrow(0, index, 1)?.item::<f64>()?,
            qy: self.accepted.flow.qy.narrow(0, index, 1)?.item::<f64>()?,
        };
        if cell.h == 0.0 {
            return Ok(CellResult::Dry);
        }
        Ok(CellResult::Wet {
            depth: cell.h,
            elevation: self.domain().bed()[index] + cell.h,
            velocity: [cell.qx / cell.h, cell.qy / cell.h],
            discharge_per_width: [cell.qx, cell.qy],
        })
    }

    /// Queries projected metre coordinates using their containing cell, with no interpolation.
    ///
    /// Internal ties select the positive-coordinate cell. Maximum exterior edges are unsupported.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn cell_at(&self, x: f64, y: f64) -> Result<CellResult, EngineError> {
        let [ox, oy] = self.domain().reference().origin();
        let (dx, dy) = self.domain().grid().spacing();
        let (nx, ny) = self.domain().grid().dimensions();
        let column = (x - ox) / dx;
        let row = (y - oy) / dy;
        if !column.is_finite()
            || !row.is_finite()
            || column < 0.0
            || row < 0.0
            || column >= nx as f64
            || row >= ny as f64
        {
            return Ok(CellResult::Unsupported);
        }
        self.cell(row.floor() as usize * nx + column.floor() as usize)
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for Solver<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Solver")
            .field("runtime", &R::name())
            .field("time", &self.accepted.time)
            .field("steps", &self.accepted.steps)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "restore_tests.rs"]
mod tests;
