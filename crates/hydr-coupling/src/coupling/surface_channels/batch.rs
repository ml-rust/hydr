use super::context::Context;
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelInitialState, SurfaceChannelSetup,
};
use crate::channel::{ChannelInitialState, ChannelReach};
use crate::solver::Solver;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

/// Accepted surface and channel states sharing one runtime and one accepted time.
///
/// Construction checks the definition and builds every child. Linked reaches use their
/// coordinator definitions with synthesized surface-exchange bindings.
pub struct SurfaceChannelBatch<R: Runtime<DType = DType>> {
    pub(super) definition: Arc<SurfaceChannelDefinition>,
    pub(super) context: Arc<Context<R>>,
    pub(super) time_s: f64,
    pub(super) surface: Solver<R>,
    /// Reaches in setup order.
    pub(super) reaches: Vec<ChannelReach<R>>,
    pub(super) accepted_steps: usize,
    pub(super) counters: LinkCounters<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for SurfaceChannelBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceChannelBatch")
            .field("setup", self.definition.setup())
            .field("time_s", &self.time_s)
            .field("accepted_steps", &self.accepted_steps)
            .field("surface_cells", &self.context.links.cells())
            .finish()
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R> {
    /// Checked definition with synthesized coordinator reach definitions.
    pub fn definition(&self) -> &SurfaceChannelDefinition {
        &self.definition
    }

    /// Immutable caller setup.
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
        let definition = SurfaceChannelDefinition::new(setup)?;
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
        let context = Context::new(client, &definition, &surface)?;
        let counters = context.zero_counters()?;
        Ok(Self {
            definition: Arc::new(definition),
            context: Arc::new(context),
            time_s,
            surface,
            reaches,
            accepted_steps: 0,
            counters,
            last: None,
        })
    }
}

/// Returns one reach state per setup reach, in setup order.
fn order_reach_states(
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
