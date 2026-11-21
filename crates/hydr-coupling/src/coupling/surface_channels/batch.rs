use super::context::Context;
use super::operating::{OperatingChildren, admit};
use super::pools::{PoolChild, PoolParts, inflow_error};
use super::runoff::build_child;
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelInitialState, SurfaceChannelSetup,
};
use crate::channel::{ChannelInitialState, ChannelReach};
use crate::runoff::SubcatchmentBatch;
use crate::solver::Solver;
use crate::storage::LevelPoolBatch;
use crate::structures::{ControlledGateBatch, ControlledPumpBatch, CulvertClient};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TensorOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::{Interp1dAlgorithms, LinearInterp1dDiagnostics};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

/// Accepted surface, channel, pool, and operating states sharing one runtime and one accepted
/// time.
///
/// Construction checks the definition and builds every child. Linked reaches use their
/// coordinator definitions with synthesized surface-exchange and structure-exchange bindings.
pub struct SurfaceChannelBatch<R: Runtime<DType = DType>> {
    pub(super) definition: Arc<SurfaceChannelDefinition>,
    pub(super) context: Arc<Context<R>>,
    /// Original common epoch and child states. Checkpoints carry it unchanged.
    pub(super) initial: Arc<SurfaceChannelInitialState>,
    pub(super) time_s: f64,
    pub(super) surface: Solver<R>,
    /// Reaches in setup order.
    pub(super) reaches: Vec<ChannelReach<R>>,
    pub(super) accepted_steps: usize,
    pub(super) counters: LinkCounters<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
    /// Accepted pools, or `None` without pools.
    pub(super) pools: Option<PoolChild<R>>,
    /// Accepted operating children, or `None` when every structure has a manual owner.
    pub(super) operating: Option<OperatingChildren<R>>,
    /// Accepted runoff child, or `None` without active runoff.
    pub(super) runoff: Option<SubcatchmentBatch<R>>,
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for SurfaceChannelBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceChannelBatch")
            .field("setup", self.definition.setup())
            .field("time_s", &self.time_s)
            .field("accepted_steps", &self.accepted_steps)
            .field(
                "surface_cells",
                &self
                    .context
                    .links
                    .as_ref()
                    .map_or(&[][..], |links| links.cells()),
            )
            .finish()
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R> {
    /// Checked definition with synthesized coordinator reach definitions.
    pub fn definition(&self) -> &SurfaceChannelDefinition {
        &self.definition
    }

    /// Caller setup. `surface.forcing` holds the forcing window the surface engine steps with.
    ///
    /// Construction keeps the supplied schedule. Continuation and resume hold the engine window.
    pub fn setup(&self) -> &SurfaceChannelSetup {
        self.definition.setup()
    }

    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted common intervals.
    pub fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }

    /// Read-only surface engine.
    pub fn surface(&self) -> &Solver<R> {
        &self.surface
    }

    /// Read-only reach selected by its declared identity.
    pub fn reach(&self, id: &str) -> Option<&ChannelReach<R>> {
        self.definition
            .setup()
            .reaches
            .iter()
            .position(|reach| reach.id == id)
            .and_then(|index| self.reaches.get(index))
    }

    /// Read-only pool child, every pool in ascending identity, or `None` without pools.
    pub fn pools(&self) -> Option<&LevelPoolBatch<R>> {
        self.pools.as_ref().map(|pools| &pools.batch)
    }

    /// Read-only controlled-gate child, or `None` without controlled gates.
    pub fn controlled_gates(&self) -> Option<&ControlledGateBatch<R>> {
        self.operating.as_ref().and_then(OperatingChildren::gate)
    }

    /// Read-only controlled-pump child, or `None` without controlled pumps.
    pub fn controlled_pumps(&self) -> Option<&ControlledPumpBatch<R>> {
        self.operating.as_ref().and_then(OperatingChildren::pump)
    }

    /// Read-only runoff child, or `None` without active runoff.
    #[doc(hidden)]
    pub fn runoff(&self) -> Option<&SubcatchmentBatch<R>> {
        self.runoff.as_ref()
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + SortingOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + LinearInterp1dDiagnostics<R>
        + CulvertClient<R>,
{
    /// Checks `setup` and `initial`, then builds the surface engine, every reach, the pools, and
    /// the operating children on `client`.
    ///
    /// # Errors
    /// Returns every [`SurfaceChannelDefinition::new`] rejection.
    /// Returns [`SurfaceChannelError::Input`] for:
    /// - a nonfinite common time
    /// - a surface or reach time that differs from `time_s` bit for bit
    /// - a reach state that is missing, repeated, or names no setup reach
    /// - a pool state count other than the setup pool count, or a pool state the pool child
    ///   rejects
    /// - a link law the structures crate rejects on `client`
    /// - an operating child the structures crate rejects, naming the child
    /// - every `operation-ownership` rejection of the operating children: a declaration without a
    ///   child, a structure with zero owners or two, a child off the common time, and a signal
    ///   naming no controller signal, bound twice, bound to no coordinator node, or missing
    ///
    /// Returns [`SurfaceChannelError::Surface`], [`SurfaceChannelError::Channel`], or
    /// [`SurfaceChannelError::Runoff`] when a child rejects its definition or state.
    pub fn new(
        client: R::Client,
        setup: SurfaceChannelSetup,
        initial: SurfaceChannelInitialState,
    ) -> Result<Self, SurfaceChannelError> {
        Self::build(client, SurfaceChannelDefinition::new(setup)?, initial)
    }

    /// Builds every child of a checked `definition` at `initial` on `client`.
    pub(super) fn build(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
    ) -> Result<Self, SurfaceChannelError> {
        let pools = pool_parts(&client, &definition, &initial)?;
        let runoff = build_child(&client, &definition, &initial)?;
        let original = Arc::new(initial.clone());
        let SurfaceChannelInitialState {
            time_s,
            surface,
            reaches,
            pools: _,
            runoff: _,
        } = initial;
        if !time_s.is_finite() {
            return Err(SurfaceChannelError::input(format!(
                "initial time_s={time_s}: supply a finite common time"
            )));
        }
        if surface.time.to_bits() != time_s.to_bits() {
            return Err(SurfaceChannelError::input(format!(
                "initial surface time={}, time_s={time_s}: supply the surface state at the common time",
                surface.time
            )));
        }
        let states = order_reach_states(definition.setup(), reaches, time_s)?;
        let surface =
            Solver::<R>::new(client.clone(), definition.setup().surface.clone(), surface)?;
        let reaches = definition
            .setup()
            .reaches
            .iter()
            .zip(&definition.coordinated)
            .zip(states)
            .map(|((reach, coordinated), state)| {
                ChannelReach::<R>::new_coordinated(client.clone(), coordinated.clone(), state)
                    .map_err(|source| SurfaceChannelError::Channel {
                        reach_id: reach.id.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let context = Context::new(client, &definition, &surface, pools.as_ref())?;
        let counters = context.zero_counters()?;
        let pools = match pools {
            Some(parts) => Some(PoolChild {
                external: context
                    .pool_topology()?
                    .inflows
                    .zero_state(&context.client)
                    .map_err(inflow_error)?,
                batch: parts.batch,
            }),
            None => None,
        };
        let operating = match &definition.setup().operating {
            Some(declared) => Some(admit(&definition, &context, time_s, declared)?),
            None => None,
        };
        Ok(Self {
            definition: Arc::new(definition),
            context: Arc::new(context),
            initial: original,
            time_s,
            surface,
            reaches,
            accepted_steps: 0,
            counters,
            last: None,
            pools,
            operating,
            runoff,
        })
    }
}

/// The pool child of `definition` at the pool states of `initial` on `client`, or `None`
/// without pools.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] for a state count other than the pool count, and
/// every pool child rejection.
pub(super) fn pool_parts<R: Runtime<DType = DType>>(
    client: &R::Client,
    definition: &SurfaceChannelDefinition,
    initial: &SurfaceChannelInitialState,
) -> Result<Option<PoolParts<R>>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    match &definition.pools {
        Some(pools) => Ok(Some(PoolParts::new(
            client.clone(),
            pools,
            &initial.pools,
            initial.time_s,
        )?)),
        None if initial.pools.is_empty() => Ok(None),
        None => Err(SurfaceChannelError::input(format!(
            "initial pool state count={}, pools=0: supply no pool states without pools",
            initial.pools.len()
        ))),
    }
}

/// Returns one reach state per setup reach, in setup order.
pub(super) fn order_reach_states(
    setup: &SurfaceChannelSetup,
    states: Vec<(String, ChannelInitialState)>,
    time_s: f64,
) -> Result<Vec<ChannelInitialState>, SurfaceChannelError> {
    let mut ordered: Vec<Option<ChannelInitialState>> = vec![None; setup.reaches.len()];
    for (id, state) in states {
        let Some(index) = setup.reaches.iter().position(|reach| reach.id == id) else {
            return Err(SurfaceChannelError::input(format!(
                "initial reach id={id:?} names no setup reach: supply states for setup reaches only"
            )));
        };
        if state.time_s.to_bits() != time_s.to_bits() {
            return Err(SurfaceChannelError::input(format!(
                "initial reach={id:?} time_s={}, common time_s={time_s}: supply the reach state at the common time",
                state.time_s
            )));
        }
        match ordered.get_mut(index) {
            Some(slot @ None) => *slot = Some(state),
            _ => {
                return Err(SurfaceChannelError::input(format!(
                    "initial reach id={id:?} repeats: supply one state per reach"
                )));
            }
        }
    }
    setup
        .reaches
        .iter()
        .zip(ordered)
        .map(|(reach, state)| {
            state.ok_or_else(|| {
                SurfaceChannelError::input(format!(
                    "initial reach id={:?} has no state: supply one state per setup reach",
                    reach.id
                ))
            })
        })
        .collect()
}
