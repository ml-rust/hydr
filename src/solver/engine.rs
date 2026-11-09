use super::{
    BoundaryConditions, CONTRACT_VERSION, CellResult, EngineError, ForcingSchedule, Snapshot,
    SolverConfig, StepDiagnostics, WaterLedger,
};
use crate::cpu::{CpuDomain, CpuState};
use crate::model::{CellState, Domain, FlowState};
use numr::ops::{ReduceOps, ScalarOps};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq)]
pub(super) struct Definition {
    pub domain: Domain,
    pub boundaries: BoundaryConditions,
    pub forcing: ForcingSchedule,
    pub config: SolverConfig,
}

#[derive(Clone, Debug)]
pub(super) struct Accepted {
    pub time: f64,
    pub state: FlowState,
    pub ledger: WaterLedger,
    pub steps: usize,
    pub last: Option<StepDiagnostics>,
    pub cpu: CpuState,
}

impl PartialEq for Accepted {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
            && self.state == other.state
            && self.ledger == other.ledger
            && self.steps == other.steps
            && self.last == other.last
    }
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

/// Sequential numr CPU reference with atomic accepted steps.
#[derive(Debug)]
pub struct Solver {
    pub(super) definition: Arc<Definition>,
    pub(super) accepted: Accepted,
    pub(super) cpu: CpuDomain,
}

impl Solver {
    /// Constructs an engine from explicit physical state and complete bounded forcing.
    ///
    /// # Errors
    /// Rejects incompatible spatial support, initial time, state, boundaries, or numerical controls.
    pub fn new(
        domain: Domain,
        state: FlowState,
        boundaries: BoundaryConditions,
        forcing: ForcingSchedule,
        config: SolverConfig,
        initial_time: f64,
    ) -> Result<Self, EngineError> {
        config.check()?;
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
        let cpu = CpuDomain::new(&domain, &forcing)?;
        let cpu_state = CpuState::from_flow(&state)?;
        let volume = cpu
            .client
            .sum(
                &cpu.client
                    .mul_scalar(&cpu_state.h, domain.grid().cell_area())?,
                &[0],
                false,
            )?
            .item::<f64>()?;
        Ok(Self {
            definition: Arc::new(Definition {
                domain,
                boundaries,
                forcing,
                config,
            }),
            accepted: Accepted {
                time: initial_time,
                state,
                ledger: WaterLedger::new(volume),
                steps: 0,
                last: None,
                cpu: cpu_state,
            },
            cpu,
        })
    }

    /// Returns the immutable checked domain and reference metadata.
    pub fn domain(&self) -> &Domain {
        &self.definition.domain
    }

    /// Returns the immutable numerical controls.
    pub fn config(&self) -> &SolverConfig {
        &self.definition.config
    }

    /// Returns accepted elapsed time in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.time
    }

    /// Returns the complete accepted depth and discharge fields.
    pub fn state(&self) -> &FlowState {
        &self.accepted.state
    }

    /// Returns the component's cumulative water ledger.
    pub fn ledger(&self) -> &WaterLedger {
        &self.accepted.ledger
    }

    /// Returns the last accepted numerical interval and supported face fluxes.
    pub fn last_step(&self) -> Option<&StepDiagnostics> {
        self.accepted.last.as_ref()
    }

    /// Captures every continuation-relevant value without exposing mutable aliases.
    pub fn snapshot(&self) -> Snapshot {
        Snapshot {
            contract: CONTRACT_VERSION,
            definition: Arc::clone(&self.definition),
            accepted: self.accepted.clone(),
        }
    }

    /// Restores complete accepted state after checking the immutable numerical definition.
    ///
    /// # Errors
    /// Rejects different geometry, parameters, boundaries, forcing, or numerical contract before mutation.
    pub fn restore(&mut self, snapshot: &Snapshot) -> Result<(), EngineError> {
        if snapshot.contract != CONTRACT_VERSION
            || !self.definition.matches(
                &snapshot.definition.domain,
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
        self.accepted = snapshot.accepted.clone();
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
        self.advance_controlled(target, max_steps, || false, |_, _| {})
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
            self.accepted = next;
            count += 1;
            if let Some(step) = self.last_step() {
                on_accepted(step, self.state().cells());
            }
        };
        Ok(AdvanceReport {
            status,
            accepted_time: self.time(),
            accepted_steps: count,
        })
    }

    /// Returns a cell's accepted physical output without crossing masks or barriers.
    pub fn cell(&self, index: usize) -> CellResult {
        let Some(cell) = self.state().cells().get(index) else {
            return CellResult::Unsupported;
        };
        if !self.domain().active()[index] {
            return CellResult::Unsupported;
        }
        if cell.h == 0.0 {
            return CellResult::Dry;
        }
        CellResult::Wet {
            depth: cell.h,
            elevation: self.domain().bed()[index] + cell.h,
            velocity: [cell.qx / cell.h, cell.qy / cell.h],
            discharge_per_width: [cell.qx, cell.qy],
        }
    }

    /// Queries projected metre coordinates using their containing cell, with no interpolation.
    ///
    /// Internal ties select the positive-coordinate cell. Maximum exterior edges are unsupported.
    pub fn cell_at(&self, x: f64, y: f64) -> CellResult {
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
            return CellResult::Unsupported;
        }
        self.cell(row.floor() as usize * nx + column.floor() as usize)
    }
}
