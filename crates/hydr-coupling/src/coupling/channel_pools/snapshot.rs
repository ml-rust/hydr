use super::context_types::Context;
use super::state::{ParentCounters, StoredStep};
use super::{ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolSetup};
use crate::channel::{ChannelSnapshot, ChannelSnapshotData};
use crate::coupling::checkpoint;
use crate::storage::{LevelPoolSnapshot, LevelPoolSnapshotData};
use hydr_core::contract::package_identity;
use numr::{dtype::DType, runtime::Runtime};
use std::sync::Arc;

/// Schema identity of the channel pool snapshot package.
pub const CHANNEL_POOL_SNAPSHOT_SCHEMA: &str = "hydr-channel-pool-checkpoint-1";
/// Numerical contract identity of the channel pool snapshot package.
pub const CHANNEL_POOL_CONTRACT_VERSION: &str = "hydr-channel-pool-numr-cpu-1";

/// Complete caller-owned continuation. Preserve every floating-point bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Immutable geometry, topology, evidence, and numerical controls.
    pub setup: ChannelPoolSetup,
    /// Original common epoch and authoritative original child states.
    pub initial: ChannelPoolInitialState,
    /// Complete accepted channel packages in setup order.
    pub reaches: Vec<ChannelSnapshotData>,
    /// Complete accepted exchange-only storage package.
    pub pools: LevelPoolSnapshotData,
    /// Authoritative parent counters and historical arithmetic bounds.
    pub counters: ChannelPoolCounterData,
    /// Bounded final common interval, absent before the first accepted step.
    pub last_interval: Option<ChannelPoolIntervalSnapshotData>,
}

impl ChannelPoolSnapshotData {
    /// Check this package identity, then the identity of every nested channel and pool package.
    pub(super) fn check_identities(&self) -> Result<(), String> {
        package_identity(
            "channel pool",
            &self.schema,
            CHANNEL_POOL_SNAPSHOT_SCHEMA,
            &self.contract,
            CHANNEL_POOL_CONTRACT_VERSION,
        )?;
        self.check_frame("", &self.reaches, &self.pools)?;
        if let Some(last) = &self.last_interval {
            self.check_frame(
                "last_interval.previous.",
                &last.previous.reaches,
                &last.previous.pools,
            )?;
        }
        Ok(())
    }

    fn check_frame(
        &self,
        prefix: &str,
        reaches: &[ChannelSnapshotData],
        pools: &LevelPoolSnapshotData,
    ) -> Result<(), String> {
        for (index, reach) in reaches.iter().enumerate() {
            let path = match self.setup.reaches.get(index) {
                Some(definition) => format!(
                    "channel pool checkpoint {prefix}reaches[{index}] ({:?})",
                    definition.id
                ),
                None => format!("channel pool checkpoint {prefix}reaches[{index}]"),
            };
            checkpoint::channel(&path, reach)?;
        }
        checkpoint::level_pool(&format!("channel pool checkpoint {prefix}pools"), pools)
    }
}

/// Parent directional counters and retained physical arithmetic charges.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolCounterData {
    /// Channel-to-pool cumulative volumes in connection order.
    pub connection_in_m3: Vec<f64>,
    /// Pool-to-channel cumulative volumes in connection order.
    pub connection_out_m3: Vec<f64>,
    /// Channel-to-pool counter addition bounds in connection order.
    pub connection_in_bound_m3: Vec<f64>,
    /// Pool-to-channel counter addition bounds in connection order.
    pub connection_out_bound_m3: Vec<f64>,
    /// Cumulative external inflow in unconnected endpoint order.
    pub external_in_m3: Vec<f64>,
    /// Cumulative external outflow in unconnected endpoint order.
    pub external_out_m3: Vec<f64>,
    /// External inflow counter addition bounds.
    pub external_in_bound_m3: Vec<f64>,
    /// External outflow counter addition bounds.
    pub external_out_bound_m3: Vec<f64>,
    /// Cumulative pool incoming-exchange assembly bounds in pool order.
    pub pool_incoming_assembly_bound_m3: Vec<f64>,
    /// Cumulative pool outgoing-exchange assembly bounds in pool order.
    pub pool_outgoing_assembly_bound_m3: Vec<f64>,
    /// Channel transfer, channel assembly, and pool state assembly charges, in that order.
    pub historical_physical_charge_m3: Vec<f64>,
}

/// Nonrecursive start frame for replaying one complete common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolPreviousData {
    /// Complete previous channel packages, including their bounded native continuation.
    pub reaches: Vec<ChannelSnapshotData>,
    /// Complete previous pool package.
    pub pools: LevelPoolSnapshotData,
    /// Previous authoritative parent accounting.
    pub counters: ChannelPoolCounterData,
}

/// Original forcing and retained native operands for the final common calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolIntervalSnapshotData {
    /// Accepted children and counters before the complete common trial.
    pub previous: ChannelPoolPreviousData,
    /// Original external forcing, preserving absent connected endpoint policies.
    pub forcing: ChannelPoolForcingInterval,
    /// Original requested endpoint, including any whole-interval retries.
    pub requested_end_s: f64,
    /// Fixed-point iterations in the accepted attempt.
    pub iterations: usize,
    /// Complete common recalculations before acceptance.
    pub retries: usize,
    /// Final supplied iterate and the exact retained map evaluation.
    pub interval: ChannelPoolNativeIntervalData,
    /// Native interval water accounting and physical charges.
    pub accounting: ChannelPoolIntervalAccountingData,
}

/// Retained numerical packets, used heads, transfers, support, and donor bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolNativeIntervalData {
    /// Final supplied packet: pool stages, connection volumes, pool wetness, attachment wetness.
    pub input_packet: Vec<f64>,
    /// Reevaluated packet with the same layout as `input_packet`.
    pub output_packet: Vec<f64>,
    /// Actual midpoint pool heads used by the channel trials, in pool order.
    pub midpoint_pool_stages_m: Vec<f64>,
    /// Original signed channel-to-pool face volumes in connection order.
    pub connection_volume_m3: Vec<f64>,
    /// Actual channel momentum contributions in connection order.
    pub connection_momentum_m4_per_s: Vec<f64>,
    /// Outward face volumes in unconnected endpoint order.
    pub external_volume_m3: Vec<f64>,
    /// Downward-bounded original cut availability in declared cut order.
    pub cut_available_lower_m3: Vec<f64>,
    /// Upward-bounded cut export demand in declared cut order.
    pub cut_outgoing_upper_m3: Vec<f64>,
    /// Incoming exchange assembly bounds in pool order.
    pub pool_incoming_bound_m3: Vec<f64>,
    /// Outgoing exchange assembly bounds in pool order.
    pub pool_outgoing_bound_m3: Vec<f64>,
    /// Final head residual, containing one scalar.
    pub head_residual_m: Vec<f64>,
    /// Final transfer residual, containing one scalar.
    pub transfer_residual_m3: Vec<f64>,
}

/// Original interval reduction results and retained physical arithmetic charges.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolIntervalAccountingData {
    /// Total represented storage before the interval, containing one scalar.
    pub storage_before_m3: Vec<f64>,
    /// Total represented storage after the interval, containing one scalar.
    pub storage_after_m3: Vec<f64>,
    /// External inflow over the interval, containing one scalar.
    pub external_in_m3: Vec<f64>,
    /// External outflow over the interval, containing one scalar.
    pub external_out_m3: Vec<f64>,
    /// Original lateral inflow over the interval, containing one scalar.
    pub lateral_in_m3: Vec<f64>,
    /// Original lateral withdrawal over the interval, containing one scalar.
    pub lateral_out_m3: Vec<f64>,
    /// Interval physical residual, containing one scalar.
    pub residual_m3: Vec<f64>,
    /// Complete current arithmetic bound, containing one scalar.
    pub roundoff_bound_m3: Vec<f64>,
    /// Channel transfer, channel assembly, and pool state assembly charges, in that order.
    pub physical_charge_m3: Vec<f64>,
}

/// Trusted native continuation without projecting numerical fields to the host.
#[derive(Clone)]
pub struct ChannelPoolSnapshot<R: Runtime<DType = DType>> {
    pub(super) context: Arc<Context<R>>,
    pub(super) reaches: Vec<ChannelSnapshot<R>>,
    pub(super) pools: LevelPoolSnapshot<R>,
    pub(super) counters: ParentCounters<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for ChannelPoolSnapshot<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChannelPoolSnapshot")
            .field("setup", &self.context.setup)
            .field("initial", &self.context.initial)
            .field("has_last_interval", &self.last.is_some())
            .finish()
    }
}
