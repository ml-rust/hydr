use super::context_types::Context;
use super::state::{Accepted, ParentCounters};
use super::{ChannelPoolError, ChannelPoolInitialState, ChannelPoolSetup};
use crate::channel::ChannelReach;
use crate::storage::LevelPoolBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use std::sync::Arc;

/// Accepted channel and pool states sharing one runtime and common interval history.
#[derive(Clone)]
pub struct ChannelPoolBatch<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) accepted: Accepted<R>,
}
impl<R: Runtime<DType = DType>> std::fmt::Debug for ChannelPoolBatch<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelPoolBatch")
            .field("setup", &self.context.setup)
            .field("time", &self.accepted.pools.time())
            .field("accepted_steps", &self.accepted.pools.accepted_steps())
            .finish()
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolBatch<R> {
    /// Immutable definitions, connection evidence, and numerical controls.
    pub fn setup(&self) -> &ChannelPoolSetup {
        &self.context.setup
    }
    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.accepted.pools.time()
    }
    /// Number of complete accepted common intervals.
    pub fn accepted_steps(&self) -> usize {
        self.accepted.pools.accepted_steps()
    }
    /// Read-only native reach selected by its declared identity.
    pub fn reach(&self, id: &str) -> Option<&ChannelReach<R>> {
        self.context
            .topology
            .reach_by_id
            .get(id)
            .and_then(|&i| self.accepted.reaches.get(i))
    }
    /// Read-only finite receiving and donating storage.
    pub fn pools(&self) -> &LevelPoolBatch<R> {
        &self.accepted.pools
    }
}
impl<R: Runtime<DType = DType>> ChannelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    /// Construct all children on the supplied client with explicit initial storage and momentum.
    pub fn new(
        client: R::Client,
        setup: ChannelPoolSetup,
        initial: ChannelPoolInitialState,
    ) -> Result<Self, ChannelPoolError>
    where
        R::Client:
            numr::ops::HypotOps<R> + numr::ops::TensorOps<R> + numr::ops::NextRepresentableOps<R>,
    {
        setup.config.check()?;
        if setup.reaches.iter().any(|reach| {
            matches!(
                &reach.definition.geometry,
                crate::channel::ChannelGeometry::PrismaticSurveyed { .. }
            )
        }) {
            return Err(ChannelPoolError::input(
                "surveyed coordinator attachment is unsupported: use standalone reach advancement",
            ));
        }
        if !initial.time_s.is_finite()
            || initial.time_s < 0.0
            || initial.reaches.len() != setup.reaches.len()
            || initial.pools.len() != setup.pools.len()
            || initial
                .reaches
                .iter()
                .any(|r| r.time_s.to_bits() != initial.time_s.to_bits())
        {
            return Err(ChannelPoolError::input(
                "initial child counts or epoch differ: supply complete states at one finite nonnegative exact time",
            ));
        }
        let reaches = setup
            .reaches
            .iter()
            .zip(&initial.reaches)
            .map(|(definition, state)| {
                ChannelReach::<R>::new(client.clone(), definition.definition.clone(), state.clone())
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: definition.id.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pools = LevelPoolBatch::<R>::new(
            client.clone(),
            setup.pools.clone(),
            initial.pools.clone(),
            initial.time_s,
            setup.config.storage.clone(),
        )?;
        let context = Arc::new(Context::new(client, setup, initial, &reaches, &pools)?);
        let counters = ParentCounters::zeros(
            &context.client,
            context.setup.connections.len(),
            context.topology.external_endpoints.len(),
            context.setup.pools.len(),
        )?;
        let result = Self {
            context,
            accepted: Accepted {
                reaches,
                pools,
                counters,
                last: None,
            },
        };
        result
            .context
            .check_epochs(&result.accepted.reaches, &result.accepted.pools)?;
        result.context.water_balance(&result.accepted)?;
        Ok(result)
    }
}
