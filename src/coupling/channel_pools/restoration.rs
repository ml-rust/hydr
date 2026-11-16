use super::identity::{same_data, same_initial, same_setup};
use super::snapshot::{ChannelPoolCounterData, ChannelPoolSnapshot, ChannelPoolSnapshotData};
use super::snapshot_values::import_counters;
use super::state::{Accepted, PreviousFrame, StoredStep};
use super::{ChannelPoolBatch, ChannelPoolError};
use crate::channel::{ChannelSnapshot, ChannelSnapshotData};
use crate::storage::{LevelPoolBatch, LevelPoolSnapshot, LevelPoolSnapshotData};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use std::sync::Arc;

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
        + AndersonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Restore trusted native continuation while retaining every destination child context and client.
    pub fn restore(&self, snapshot: &ChannelPoolSnapshot<R>) -> Result<Self, ChannelPoolError> {
        require(
            same_setup(&self.context.setup, &snapshot.context.setup)
                && same_initial(&self.context.initial, &snapshot.context.initial),
            "immutable setup or original initial-state bits differ",
        )?;
        require(
            self.context
                .client
                .device()
                .is_same(snapshot.context.client.device()),
            "trusted tensor device differs from the destination client",
        )?;
        let reaches = self.restore_channels(&snapshot.reaches)?;
        let pools = self.restore_pools(&snapshot.pools)?;
        let last = snapshot
            .last
            .as_ref()
            .map(|step| {
                let previous_reaches = self.restore_channels(&step.previous.reaches)?;
                let previous_pools = self.restore_pools(&step.previous.pools)?;
                Ok::<_, ChannelPoolError>(Arc::new(StoredStep {
                    previous: PreviousFrame {
                        reaches: previous_reaches
                            .iter()
                            .map(|reach| reach.snapshot())
                            .collect(),
                        pools: previous_pools.snapshot(),
                        counters: step.previous.counters.clone(),
                    },
                    forcing: Arc::clone(&step.forcing),
                    requested_end_s: step.requested_end_s,
                    iterations: step.iterations,
                    retries: step.retries,
                    interval: step.interval.clone(),
                    accounting: step.accounting.clone(),
                }))
            })
            .transpose()?;
        let candidate = Self {
            context: Arc::clone(&self.context),
            accepted: Accepted {
                reaches,
                pools,
                counters: snapshot.counters.clone(),
                last,
            },
        };
        candidate.check_parent_state()?;
        require(
            (candidate.accepted_steps() == 0) == candidate.accepted.last.is_none(),
            "accepted count and final interval presence disagree",
        )?;
        Ok(candidate)
    }

    /// Replay the complete original final trial before accepting a caller-owned continuation package.
    ///
    /// Applications own authenticated history and content hashes for intervals preceding this bounded proof.
    pub fn import_snapshot(&self, data: &ChannelPoolSnapshotData) -> Result<Self, ChannelPoolError>
    where
        R::Client: numr::ops::TensorOps<R>
            + numr::ops::NextRepresentableOps<R>
            + solvr::integrate::CertifiedSimpsonAlgorithms<R>
            + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
            + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
            + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
    {
        require(
            same_setup(&self.context.setup, &data.setup)
                && same_initial(&self.context.initial, &data.initial),
            "immutable setup or original initial-state bits differ",
        )?;
        require(
            (data.pools.accepted_steps == 0) == data.last_interval.is_none(),
            "accepted count and final interval presence disagree",
        )?;
        check_counter_data(
            &data.counters,
            self.context.setup.connections.len(),
            self.context.topology.external_endpoints.len(),
            self.context.setup.pools.len(),
            data.pools.accepted_steps,
        )?;
        let current_counters = import_counters(&self.context.client, &data.counters)?;
        current_counters.check(
            &self.context.client,
            self.context.setup.connections.len(),
            self.context.topology.external_endpoints.len(),
            self.context.setup.pools.len(),
            data.pools.accepted_steps,
        )?;
        self.context.check_boundary_history(
            &data.reaches,
            &current_counters,
            data.pools.accepted_steps,
        )?;
        let candidate = if let Some(last) = &data.last_interval {
            require(
                last.previous.pools.accepted_steps.checked_add(1)
                    == Some(data.pools.accepted_steps),
                "final interval does not continue the previous accepted count",
            )?;
            let previous = self.import_frame(
                &last.previous.reaches,
                &last.previous.pools,
                &last.previous.counters,
            )?;
            require(
                previous.time() < data.pools.time
                    && last.forcing.start_s.to_bits() == previous.time().to_bits(),
                "final forcing does not begin at the previous accepted time",
            )?;
            let forcing = previous.bind_host_forcing(last.requested_end_s, &last.forcing)?;
            previous
                .trial_bound(last.requested_end_s, forcing)?
                .into_next()
        } else {
            let original = self.import_frame(&data.reaches, &data.pools, &data.counters)?;
            require(
                original.accepted_steps() == 0
                    && original.time().to_bits() == self.context.initial.time_s.to_bits(),
                "zero-history package differs from the original epoch",
            )?;
            original
        };
        require(
            same_data(&candidate.snapshot_data()?, data),
            "accepted children, counters, packets, heads, transfers, or bounds differ from complete replay",
        )?;
        Ok(candidate)
    }

    fn check_parent_state(&self) -> Result<(), ChannelPoolError> {
        self.context
            .check_epochs(&self.accepted.reaches, &self.accepted.pools)?;
        self.context.water_balance(&self.accepted)?;
        Ok(())
    }

    fn import_frame(
        &self,
        reaches: &[ChannelSnapshotData],
        pools: &LevelPoolSnapshotData,
        counters: &ChannelPoolCounterData,
    ) -> Result<Self, ChannelPoolError>
    where
        R::Client: numr::ops::TensorOps<R>
            + numr::ops::NextRepresentableOps<R>
            + solvr::integrate::CertifiedSimpsonAlgorithms<R>
            + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
            + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
            + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
    {
        require(
            reaches.len() == self.context.setup.reaches.len(),
            "channel package count differs from setup",
        )?;
        require(
            pools.source_policies.is_none()
                && pools.element_ledgers.len() == self.context.initial.pools.len()
                && pools
                    .element_ledgers
                    .iter()
                    .zip(&self.context.initial.pools)
                    .all(|(ledger, initial)| {
                        ledger.initial_volume_m3.to_bits() == initial.volume_m3.to_bits()
                    }),
            "pool package changes source support or original storage bits",
        )?;
        let native_reaches = self
            .accepted
            .reaches
            .iter()
            .zip(reaches)
            .zip(&self.context.setup.reaches)
            .map(|((destination, data), definition)| {
                destination
                    .import_snapshot(data)
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: definition.id.clone(),
                        source,
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        // Owned pool import constructs a checked context. Rebind its trusted state to the destination cut-plan context.
        let mut imported = self.accepted.pools.clone();
        imported.restore_data(pools)?;
        let pools = self.restore_pools(&imported.snapshot())?;
        check_counter_data(
            counters,
            self.context.setup.connections.len(),
            self.context.topology.external_endpoints.len(),
            self.context.setup.pools.len(),
            pools.accepted_steps(),
        )?;
        let counters = import_counters(&self.context.client, counters)?;
        self.context
            .check_boundary_history(reaches, &counters, pools.accepted_steps())?;
        let candidate = Self {
            context: Arc::clone(&self.context),
            accepted: Accepted {
                reaches: native_reaches,
                pools,
                counters,
                last: None,
            },
        };
        candidate.check_parent_state()?;
        Ok(candidate)
    }

    fn restore_channels(
        &self,
        snapshots: &[ChannelSnapshot<R>],
    ) -> Result<Vec<crate::channel::ChannelReach<R>>, ChannelPoolError> {
        require(
            snapshots.len() == self.context.setup.reaches.len(),
            "trusted channel count differs from setup",
        )?;
        self.accepted
            .reaches
            .iter()
            .zip(snapshots)
            .zip(&self.context.setup.reaches)
            .map(|((destination, snapshot), definition)| {
                destination
                    .restore(snapshot)
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: definition.id.clone(),
                        source,
                    })
            })
            .collect()
    }

    fn restore_pools(
        &self,
        snapshot: &LevelPoolSnapshot<R>,
    ) -> Result<LevelPoolBatch<R>, ChannelPoolError> {
        let mut pools = self.accepted.pools.clone();
        pools.restore(snapshot)?;
        Ok(pools)
    }
}

fn require(condition: bool, detail: &str) -> Result<(), ChannelPoolError> {
    if condition {
        Ok(())
    } else {
        Err(ChannelPoolError::input(format!(
            "checkpoint {detail}: supply an unchanged complete same-build package"
        )))
    }
}

fn check_counter_data(
    data: &ChannelPoolCounterData,
    connections: usize,
    external: usize,
    pools: usize,
    steps: usize,
) -> Result<(), ChannelPoolError> {
    for (values, count) in [
        (&data.connection_in_m3, connections),
        (&data.connection_out_m3, connections),
        (&data.connection_in_bound_m3, connections),
        (&data.connection_out_bound_m3, connections),
        (&data.external_in_m3, external),
        (&data.external_out_m3, external),
        (&data.external_in_bound_m3, external),
        (&data.external_out_bound_m3, external),
        (&data.pool_incoming_assembly_bound_m3, pools),
        (&data.pool_outgoing_assembly_bound_m3, pools),
        (&data.historical_physical_charge_m3, 3),
    ] {
        require(
            values.len() == count,
            "parent counter support differs from immutable topology",
        )?;
        if steps == 0 {
            require(
                values.iter().all(|value| value.to_bits() == 0),
                "zero-history parent counters require original positive-zero bits",
            )?;
        }
    }
    Ok(())
}
