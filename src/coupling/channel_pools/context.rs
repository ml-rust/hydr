use super::context_types::{Context, ReductionPlans, Topology};
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

impl<R: Runtime<DType = DType>> Context<R>
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
    pub(super) fn new(
        client: R::Client,
        setup: ChannelPoolSetup,
        initial: ChannelPoolInitialState,
        reaches: &[ChannelReach<R>],
        pools: &LevelPoolBatch<R>,
    ) -> Result<Self, ChannelPoolError> {
        setup.config.check()?;
        let (minimum, maximum) = duration_limits(&setup);
        if minimum > maximum {
            return Err(ChannelPoolError::input(format!(
                "common duration intersection=[{minimum},{maximum}] is empty: supply overlapping parent and channel duration limits"
            )));
        }
        let topology = Topology::new(&client, &setup)?;
        let cut_plan = pools.bind_coupling_cuts(&topology.cuts)?;
        let reductions = ReductionPlans::new(&client, &setup, topology.external_endpoints.len())?;
        let (initial_storage_m3, initial_storage_bound_m3) =
            reductions.inventory(&client, reaches, pools)?;
        Ok(Self {
            client,
            setup,
            initial,
            topology,
            reductions,
            cut_plan,
            initial_storage_m3,
            initial_storage_bound_m3,
        })
    }
    pub(super) fn duration_limits(&self) -> (f64, f64) {
        duration_limits(&self.setup)
    }
    pub(super) fn check_epochs(
        &self,
        reaches: &[ChannelReach<R>],
        pools: &LevelPoolBatch<R>,
    ) -> Result<(), ChannelPoolError> {
        if reaches.len() != self.setup.reaches.len()
            || reaches.iter().any(|r| {
                r.time().to_bits() != pools.time().to_bits()
                    || r.accepted_steps() != pools.accepted_steps()
            })
        {
            return Err(ChannelPoolError::numerical(
                "child epochs or accepted-step counts differ. Restore a complete common state",
            ));
        }
        Ok(())
    }
}

fn duration_limits(setup: &ChannelPoolSetup) -> (f64, f64) {
    setup.reaches.iter().fold(
        (setup.config.minimum_step_s, setup.config.maximum_step_s),
        |(minimum, maximum), reach| {
            (
                minimum.max(reach.definition.config.min_step),
                maximum.min(reach.definition.config.max_step),
            )
        },
    )
}
