use super::definition::Definition;
use super::forcing_history::ForcingPrefix;
use super::tensor::land::TensorLand;
use super::tensor::land_sources::checked_volume;
use super::tensor::{TensorDomain, TensorState};
use super::{
    BoundaryConditions, CONTRACT_VERSION, CellResult, EngineError, ForcingSchedule, InitialState,
    Snapshot, SolverConfig, SolverSetup, StepDiagnostics, WaterLedger,
};
use crate::model::{CellState, Domain, FlowState};
use crate::soil::{LandModel, SoilState};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct Accepted<R: Runtime<DType = DType>> {
    pub time: f64,
    pub land: TensorLand<R>,
    pub ledger: WaterLedger,
    pub steps: usize,
    pub last: Option<super::receipts::StepReceipt<R>>,
    pub flow: TensorState<R>,
}

/// Runtime-selected shallow-water engine with atomic accepted steps.
pub struct Solver<R: Runtime<DType = DType>> {
    pub(super) definition: Arc<Definition>,
    pub(super) accepted: Accepted<R>,
    pub(super) context: Arc<TensorDomain<R>>,
}

/// A clone shares the immutable definition, domain and accepted tensors.
impl<R: Runtime<DType = DType>> Clone for Solver<R> {
    fn clone(&self) -> Self {
        Self {
            definition: Arc::clone(&self.definition),
            accepted: self.accepted.clone(),
            context: Arc::clone(&self.context),
        }
    }
}

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
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
        Self::with_prefix(client, setup, None, initial)
    }

    /// Constructs an engine whose forcing window follows `prefix`.
    ///
    /// Device sources cover only the retained window. `None` starts an empty prefix at the first interval.
    /// With a prefix, `initial.time` can equal the window end, as a checkpoint at the forcing end does.
    pub(super) fn with_prefix(
        client: R::Client,
        setup: SolverSetup,
        prefix: Option<ForcingPrefix>,
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
        if let Some(error) = crate::tensor::probe_correctly_rounded::<R>() {
            return Err(EngineError::rejected(
                "arithmetic environment",
                error.message("surface"),
            ));
        }
        let state = FlowState::new(&domain, state.cells().to_vec())?;
        let boundaries =
            BoundaryConditions::with_open_faces(&domain, boundaries.open_faces().to_vec())?;
        let forcing = ForcingSchedule::new(&domain, &boundaries, forcing.intervals().to_vec())?;
        let covered = match &prefix {
            None => forcing.at(initial_time).is_some(),
            Some(_) => {
                let intervals = forcing.intervals();
                intervals
                    .first()
                    .is_some_and(|first| first.start <= initial_time)
                    && intervals
                        .last()
                        .is_some_and(|last| initial_time <= last.end)
            }
        };
        if !initial_time.is_finite() || !covered {
            return Err(EngineError::rejected(
                "initial time",
                format!("time={initial_time}: supply a finite time inside forcing coverage"),
            ));
        }
        let land = LandModel::new(&domain, land.cells().to_vec())
            .map_err(super::tensor::land::static_error)?;
        let soil = SoilState::new(&land, soil.cells().to_vec())
            .map_err(super::tensor::land::static_error)?;
        let land_state = TensorLand::new(client.clone(), &land, &soil)?;
        let prefix = match prefix {
            Some(prefix) => prefix,
            None => ForcingPrefix::empty(
                forcing
                    .intervals()
                    .first()
                    .map(|interval| interval.start)
                    .ok_or_else(|| {
                        EngineError::rejected("forcing", "no intervals: supply explicit coverage")
                    })?,
            ),
        };
        let definition = Definition::new(domain, land, boundaries, forcing, config, prefix)?;
        let context = TensorDomain::<R>::new(client, &definition.domain, &definition.forcing)?;
        let flow_state = TensorState::from_flow(context.client.device(), &state)?;
        let volume = checked_volume(
            &context.client,
            &flow_state.h,
            definition.domain.grid().cell_area(),
        )?;
        Ok(Self {
            definition: Arc::new(definition),
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
