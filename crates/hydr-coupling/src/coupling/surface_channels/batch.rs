use super::context::Context;
use super::operating::{OperatingChildren, SurfaceChannelOperating, admit, check_manual};
use super::pools::{PoolChild, PoolParts, inflow_error};
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelInitialState, SurfaceChannelSetup,
};
use crate::channel::{ChannelInitialState, ChannelReach};
use crate::solver::Solver;
use crate::storage::LevelPoolState;
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

/// Accepted surface and channel states sharing one runtime and one accepted time.
///
/// Construction checks the definition and builds every child. Linked reaches use their
/// coordinator definitions with synthesized surface-exchange bindings.
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
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Checks `setup` and `initial`, then builds the surface engine and every reach on `client`.
    ///
    /// # Errors
    /// Returns every [`SurfaceChannelDefinition::new`] rejection.
    /// Returns [`SurfaceChannelError::Input`] for:
    /// - a nonfinite common time
    /// - a surface or reach time that differs from `time_s` bit for bit
    /// - a reach state that is missing, repeated, or names no setup reach
    /// - a link weir law the structures crate rejects on `client`
    ///
    /// Returns [`SurfaceChannelError::Surface`] or [`SurfaceChannelError::Channel`] when a
    /// child rejects its definition or state.
    pub fn new(
        client: R::Client,
        setup: SurfaceChannelSetup,
        initial: SurfaceChannelInitialState,
    ) -> Result<Self, SurfaceChannelError> {
        Self::from_definition(client, SurfaceChannelDefinition::new(setup)?, initial)
    }

    /// Checks `initial` against a checked `definition` without pools, then builds every child on
    /// `client`.
    ///
    /// # Errors
    /// Returns every [`Self::new`] rejection after the definition checks, and
    /// [`SurfaceChannelError::Input`] for a definition with pools or a structure with a dynamic
    /// owner (`operation-ownership`).
    pub(super) fn from_definition(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
    ) -> Result<Self, SurfaceChannelError> {
        if definition.pools.is_some() {
            return Err(SurfaceChannelError::input(
                "pool states: build a coordinator with pools through its pool constructor",
            ));
        }
        check_manual(&definition)?;
        Self::build(client, definition, initial, None)
    }

    /// Builds every child of `definition` at `initial` on `client`, with the pool child `pools`.
    fn build(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
        pools: Option<PoolParts<R>>,
    ) -> Result<Self, SurfaceChannelError> {
        let original = Arc::new(initial.clone());
        let SurfaceChannelInitialState {
            time_s,
            surface,
            reaches,
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
            operating: None,
        })
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
        + SortingOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>,
{
    /// Checks `initial` and the pool `states` against a checked `definition`, then builds every
    /// child on `client`.
    ///
    /// `states` holds one accepted volume per declared pool, in declaration order. A definition
    /// without pools takes none.
    ///
    /// # Errors
    /// Returns every [`Self::from_definition`] rejection, every pool child rejection, and
    /// [`SurfaceChannelError::Input`] for a state count that differs from the pool count.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "in-crate callers construct pool coordinators")
    )]
    pub(crate) fn with_pools(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
        states: &[LevelPoolState],
    ) -> Result<Self, SurfaceChannelError> {
        check_manual(&definition)?;
        Self::pooled(client, definition, initial, states)
    }

    /// Builds every child of `definition` at `initial` and the pool `states` on `client`, then
    /// binds the fresh operating children `operating`.
    ///
    /// # Errors
    /// Returns every [`Self::with_pools`] rejection except the manual-owner check, and every
    /// `operation-ownership` rejection of the operating children. See
    /// [`super::operating::admit`].
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "in-crate callers construct operating coordinators"
        )
    )]
    pub(crate) fn with_operating(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
        states: &[LevelPoolState],
        operating: SurfaceChannelOperating<R>,
    ) -> Result<Self, SurfaceChannelError>
    where
        R::Client: LinearInterp1dDiagnostics<R>,
    {
        let time_s = initial.time_s;
        let mut batch = Self::pooled(client, definition, initial, states)?;
        batch.operating = Some(admit(&batch.definition, &batch.context, time_s, operating)?);
        Ok(batch)
    }

    /// Builds every child of `definition` at `initial` and the pool `states` on `client`.
    fn pooled(
        client: R::Client,
        definition: SurfaceChannelDefinition,
        initial: SurfaceChannelInitialState,
        states: &[LevelPoolState],
    ) -> Result<Self, SurfaceChannelError> {
        let parts = match &definition.pools {
            Some(pools) => Some(PoolParts::new(
                client.clone(),
                pools,
                states,
                initial.time_s,
            )?),
            None if states.is_empty() => None,
            None => {
                return Err(SurfaceChannelError::input(format!(
                    "initial pool state count={}, pools=0: supply no pool states without pools",
                    states.len()
                )));
            }
        };
        Self::build(client, definition, initial, parts)
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
