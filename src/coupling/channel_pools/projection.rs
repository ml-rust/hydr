use super::context_types::Context;
use super::snapshot::{
    ChannelPoolIntervalSnapshotData, ChannelPoolPreviousData, ChannelPoolSnapshot,
    ChannelPoolSnapshotData,
};
use super::state::{BoundForcing, StoredStep};
use super::{
    ChannelPoolBatch, ChannelPoolConnectionLedger, ChannelPoolDonorCut, ChannelPoolError,
    ChannelPoolForcingInterval, ChannelPoolIterationData, ChannelPoolReachForcing,
    ChannelPoolStepDiagnostics, ChannelPoolTransfer, ChannelPoolWaterBalance,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
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
    /// Project lifetime water accounting without counting internal connection traffic as external water.
    pub fn ledger(&self) -> Result<ChannelPoolWaterBalance, ChannelPoolError> {
        let b = self.context.water_balance(&self.accepted)?;
        Ok(ChannelPoolWaterBalance {
            initial_storage_m3: b.initial_storage_m3.item::<f64>()?,
            storage_m3: b.storage_m3.item::<f64>()?,
            external_boundary_in_m3: b.external_in_m3.item::<f64>()?,
            external_boundary_out_m3: b.external_out_m3.item::<f64>()?,
            lateral_in_m3: b.lateral_in_m3.item::<f64>()?,
            lateral_out_m3: b.lateral_out_m3.item::<f64>()?,
            residual_m3: b.residual_m3.item::<f64>()?,
            roundoff_bound_m3: b.roundoff_bound_m3.item::<f64>()?,
        })
    }

    /// Project cumulative directional connection volumes and their counter arithmetic bounds.
    pub fn connection_ledgers(&self) -> Result<Vec<ChannelPoolConnectionLedger>, ChannelPoolError> {
        let incoming = self
            .accepted
            .counters
            .connection_in_m3
            .try_to_vec::<f64>()?;
        let outgoing = self
            .accepted
            .counters
            .connection_out_m3
            .try_to_vec::<f64>()?;
        let bounds = crate::numerics::water_transfers::upward_add(
            &self.context.client,
            &self.accepted.counters.connection_in_bound_m3,
            &self.accepted.counters.connection_out_bound_m3,
        )?
        .try_to_vec::<f64>()?;
        Ok(self
            .context
            .setup
            .connections
            .iter()
            .enumerate()
            .map(|(i, connection)| ChannelPoolConnectionLedger {
                connection_id: connection.id.clone(),
                channel_to_pool_volume_m3: incoming[i],
                pool_to_channel_volume_m3: outgoing[i],
                counter_roundoff_bound_m3: bounds[i],
            })
            .collect())
    }

    /// Project the complete final accepted common interval.
    pub fn last_step(&self) -> Result<Option<ChannelPoolStepDiagnostics>, ChannelPoolError> {
        self.accepted
            .last
            .as_ref()
            .map(|step| self.context.project_step(step, self.time()))
            .transpose()
    }

    /// Retain complete native continuation without numerical field readback.
    pub fn snapshot(&self) -> ChannelPoolSnapshot<R> {
        ChannelPoolSnapshot {
            context: Arc::clone(&self.context),
            reaches: self
                .accepted
                .reaches
                .iter()
                .map(|reach| reach.snapshot())
                .collect(),
            pools: self.accepted.pools.snapshot(),
            counters: self.accepted.counters.clone(),
            last: self.accepted.last.as_ref().map(Arc::clone),
        }
    }

    /// Project complete continuation values for caller-owned persistence.
    pub fn snapshot_data(&self) -> Result<ChannelPoolSnapshotData, ChannelPoolError> {
        self.snapshot().data()
    }
}

impl<R: Runtime<DType = DType>> ChannelPoolSnapshot<R>
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
    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.pools.time()
    }

    /// Project complete owned values, including the bounded original common calculation.
    pub fn data(&self) -> Result<ChannelPoolSnapshotData, ChannelPoolError> {
        let reaches = self
            .reaches
            .iter()
            .zip(&self.context.setup.reaches)
            .map(|(reach, definition)| {
                reach.data().map_err(|source| ChannelPoolError::Channel {
                    reach_id: definition.id.clone(),
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ChannelPoolSnapshotData {
            setup: self.context.setup.clone(),
            initial: self.context.initial.clone(),
            reaches,
            pools: self.pools.data()?,
            counters: self.counters.data()?,
            last_interval: self
                .last
                .as_ref()
                .map(|step| project_interval(&self.context.setup, &self.context.client, step))
                .transpose()?,
        })
    }
}

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
        + AndersonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(super) fn project_step(
        &self,
        step: &StoredStep<R>,
        end_s: f64,
    ) -> Result<ChannelPoolStepDiagnostics, ChannelPoolError> {
        let interval = step.interval.data()?;
        let a = &step.accounting;
        let pools = self.setup.pools.len();
        let connections = self.setup.connections.len();
        let packet = &interval.input_packet;
        if packet.len() != 2 * (pools + connections) {
            return Err(ChannelPoolError::numerical(
                "retained iteration packet has incomplete support",
            ));
        }
        let indices = self.topology.connection_pool_indices.try_to_vec::<i64>()?;
        let transfers = self
            .setup
            .connections
            .iter()
            .enumerate()
            .map(|(i, connection)| {
                let pool = usize::try_from(indices[i]).map_err(|_| {
                    ChannelPoolError::numerical("retained connection pool index is negative")
                })?;
                let head = interval
                    .midpoint_pool_stages_m
                    .get(pool)
                    .copied()
                    .ok_or_else(|| {
                        ChannelPoolError::numerical(
                            "retained connection lacks its used midpoint head",
                        )
                    })?;
                Ok(ChannelPoolTransfer {
                    connection_id: connection.id.clone(),
                    signed_volume_m3: interval.connection_volume_m3[i],
                    midpoint_pool_elevation_m: head,
                    longitudinal_momentum_impulse_m4_per_s: interval.connection_momentum_m4_per_s
                        [i],
                })
            })
            .collect::<Result<Vec<_>, ChannelPoolError>>()?;
        let donor_cuts = self
            .topology
            .cuts
            .iter()
            .enumerate()
            .map(|(i, cut)| ChannelPoolDonorCut {
                pool_id: self.setup.pools[cut.pool_index].id.clone(),
                bed_elevation_m: cut.elevation_m,
                available_lower_m3: interval.cut_available_lower_m3[i],
                outgoing_upper_m3: interval.cut_outgoing_upper_m3[i],
            })
            .collect();
        Ok(ChannelPoolStepDiagnostics {
            start_s: step.previous.pools.time(),
            end_s,
            requested_end_s: step.requested_end_s,
            iterations: step.iterations,
            retries: step.retries,
            head_residual_m: step.interval.head_residual_m.item::<f64>()?,
            transfer_residual_m3: step.interval.transfer_residual_m3.item::<f64>()?,
            storage_before_m3: a.storage_before_m3.item::<f64>()?,
            storage_after_m3: a.storage_after_m3.item::<f64>()?,
            external_boundary_in_m3: a.external_in_m3.item::<f64>()?,
            external_boundary_out_m3: a.external_out_m3.item::<f64>()?,
            lateral_in_m3: a.lateral_in_m3.item::<f64>()?,
            lateral_out_m3: a.lateral_out_m3.item::<f64>()?,
            mass_residual_m3: a.residual_m3.item::<f64>()?,
            aggregation_roundoff_bound_m3: a.roundoff_bound_m3.item::<f64>()?,
            forcing_interval: project_forcing(&self.setup, &self.client, &step.forcing)?,
            iteration: ChannelPoolIterationData {
                end_stages_m: packet[..pools].to_vec(),
                signed_volumes_m3: packet[pools..pools + connections].to_vec(),
                pool_wet: packet[pools + connections..2 * pools + connections]
                    .iter()
                    .map(|value| *value == 1.0)
                    .collect(),
                attachment_wet: packet[2 * pools + connections..]
                    .iter()
                    .map(|value| *value == 1.0)
                    .collect(),
            },
            transfers,
            donor_cuts,
        })
    }
}

pub(super) fn project_forcing<R: Runtime<DType = DType>>(
    setup: &super::ChannelPoolSetup,
    client: &R::Client,
    forcing: &BoundForcing<R>,
) -> Result<ChannelPoolForcingInterval, ChannelPoolError> {
    if forcing.reaches.len() != setup.reaches.len() {
        return Err(ChannelPoolError::numerical(
            "retained external forcing has incomplete reach support",
        ));
    }
    let reaches = forcing
        .reaches
        .iter()
        .zip(&setup.reaches)
        .map(|(bound, reach)| {
            let data =
                bound
                    .project_external(client)
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: reach.id.clone(),
                        source,
                    })?;
            if data.start_s.to_bits() != forcing.start_s.to_bits()
                || data.end_s.to_bits() != forcing.end_s.to_bits()
            {
                return Err(ChannelPoolError::numerical(
                    "retained child forcing span differs from common external forcing",
                ));
            }
            Ok(ChannelPoolReachForcing {
                reach_id: reach.id.clone(),
                left: data.left,
                right: data.right,
                lateral_inputs: data.lateral_inputs,
            })
        })
        .collect::<Result<Vec<_>, ChannelPoolError>>()?;
    Ok(ChannelPoolForcingInterval {
        start_s: forcing.start_s,
        end_s: forcing.end_s,
        reaches,
    })
}

fn project_interval<R: Runtime<DType = DType>>(
    setup: &super::ChannelPoolSetup,
    client: &R::Client,
    step: &StoredStep<R>,
) -> Result<ChannelPoolIntervalSnapshotData, ChannelPoolError>
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
    let reaches = step
        .previous
        .reaches
        .iter()
        .zip(&setup.reaches)
        .map(|(snapshot, reach)| {
            snapshot.data().map_err(|source| ChannelPoolError::Channel {
                reach_id: reach.id.clone(),
                source,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ChannelPoolIntervalSnapshotData {
        previous: ChannelPoolPreviousData {
            reaches,
            pools: step.previous.pools.data()?,
            counters: step.previous.counters.data()?,
        },
        forcing: project_forcing(setup, client, &step.forcing)?,
        requested_end_s: step.requested_end_s,
        iterations: step.iterations,
        retries: step.retries,
        interval: step.interval.data()?,
        accounting: step.accounting.data()?,
    })
}
