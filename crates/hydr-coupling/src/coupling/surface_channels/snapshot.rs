//! Owned surface/channel checkpoint packages and the trusted in-memory snapshot.

use super::context::Context;
use super::operating::OperatingChildren;
use super::pools::PoolChild;
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelConfig, SurfaceChannelDefinition, SurfaceChannelForcingInterval,
    SurfaceChannelInitialState, SurfaceChannelLink, SurfaceChannelLinkLedger,
    SurfaceChannelOperating, SurfaceChannelOperatingData, SurfaceChannelPoolEnd,
    SurfaceChannelPoolEndLedger, SurfaceChannelPort, SurfaceChannelPortLedger, SurfaceChannelReach,
    SurfaceChannelRunoff, SurfaceChannelRunoffLedger, SurfaceChannelSetup,
    SurfaceChannelStepDiagnostics, SurfaceChannelStructure, SurfaceChannelStructureLedger,
};
use crate::channel::{ChannelSnapshot, ChannelSnapshotData};
use crate::coupling::checkpoint;
use crate::coupling::level_pools::{
    CoupledPoolInflowBinding, CoupledPoolInflowLedger, CoupledPoolSupport,
};
use crate::runoff::{SubcatchmentBatch, SubcatchmentSnapshotData};
use crate::solver::{Snapshot, SnapshotData, SolverSetup};
use crate::storage::{
    LevelPoolConfig, LevelPoolElement, LevelPoolSnapshotData, LevelPoolSourcePolicy,
};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Schema identity of the surface/channel snapshot package.
pub const SURFACE_CHANNEL_SNAPSHOT_SCHEMA: &str = "hydr-surface-channel-checkpoint-11";
/// Numerical contract identity of the surface/channel snapshot package.
pub const SURFACE_CHANNEL_CONTRACT_VERSION: &str = "hydr-surface-channel-numr-cpu-13";

/// Complete caller-owned continuation. Preserve every floating-point bit when encoding.
///
/// The nested surface package holds the surface definition and its bounded forcing window.
/// Import checks consistency and replays the last interval, decision completion included.
/// Port ghost states rebuild from the start-frame reach packages, so no package holds them.
/// It establishes no authenticity.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Every setup part except the surface definition, which lives in `surface`.
    pub setup: SurfaceChannelSetupData,
    /// Original common epoch and authoritative original child states.
    pub initial: SurfaceChannelInitialState,
    /// Complete accepted surface package. It carries coordinator link and port water.
    pub surface: SnapshotData,
    /// Complete accepted `(reach id, package)` pairs in setup order.
    pub reaches: Vec<(String, ChannelSnapshotData)>,
    /// Complete accepted pool package, or `None` without pools.
    pub pools: Option<LevelPoolSnapshotData>,
    /// Complete accepted operating packages, or `None` without operating children.
    pub operating: Option<SurfaceChannelOperatingData>,
    /// Complete accepted runoff package, or `None` without active runoff.
    pub runoff: Option<SubcatchmentSnapshotData>,
    /// Lifetime link, structure, port, pool-end, pool inflow, and runoff counters, accepted
    /// count, and common time.
    pub counters: SurfaceChannelCounterData,
    /// Active geometry epoch per schedule at `counters.time_s`. Recomputed and checked at import.
    pub geometry_epochs: Vec<usize>,
    /// Active pool curve epoch per schedule at `counters.time_s`. Recomputed and checked at import.
    pub pool_curve_epochs: Vec<usize>,
    /// Last common interval with its start frame, absent before the first accepted interval.
    pub last_interval: Option<SurfaceChannelIntervalSnapshotData>,
}

/// Every setup part except the surface definition, which the nested surface package holds.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelSetupData {
    /// Caller reach definitions in setup order, without synthesized bindings.
    pub reaches: Vec<SurfaceChannelReach>,
    /// Bank-overflow links in declaration order.
    pub links: Vec<SurfaceChannelLink>,
    /// Reach-end ports in declaration order.
    pub ports: Vec<SurfaceChannelPort>,
    /// Level pools in declaration order.
    pub pools: Vec<LevelPoolElement>,
    /// Storage support of every pool, in pool order.
    pub pool_support: Vec<CoupledPoolSupport>,
    /// Local source policy of every pool, in pool order, or `None` without pool sources.
    pub pool_sources: Option<Vec<LevelPoolSourcePolicy>>,
    /// External pool inflow bindings in declaration order.
    pub pool_inflows: Vec<CoupledPoolInflowBinding>,
    /// Dimensional controls of the pool child.
    pub storage: LevelPoolConfig,
    /// Structure links in declaration order.
    pub structures: Vec<SurfaceChannelStructure>,
    /// Pool ends in declaration order.
    pub pool_ends: Vec<SurfaceChannelPoolEnd>,
    /// Operating children as declared, or `None` without them.
    pub operating: Option<SurfaceChannelOperating>,
    /// Native runoff as declared, or `None` without it.
    pub runoff: Option<SurfaceChannelRunoff>,
    /// Prescribed structure geometry schedules in declaration order.
    pub geometry_schedules: Vec<crate::structures::StructureGeometrySchedule>,
    /// Prescribed pool stage-volume curve epoch schedules in declaration order.
    pub pool_curve_schedules: Vec<crate::storage::PoolCurveSchedule>,
    /// Evidence that surface cells, channel cells, and pools store no water twice.
    pub storage_exclusion_evidence: String,
    /// Explicit tolerances, duration limits, and work limits.
    pub config: SurfaceChannelConfig,
}

impl SurfaceChannelSetupData {
    /// Copies every part of `setup` except the surface definition.
    pub fn of(setup: &SurfaceChannelSetup) -> Self {
        Self {
            reaches: setup.reaches.clone(),
            links: setup.links.clone(),
            ports: setup.ports.clone(),
            pools: setup.pools.clone(),
            pool_support: setup.pool_support.clone(),
            pool_sources: setup.pool_sources.clone(),
            pool_inflows: setup.pool_inflows.clone(),
            storage: setup.storage.clone(),
            structures: setup.structures.clone(),
            pool_ends: setup.pool_ends.clone(),
            operating: setup.operating.clone(),
            runoff: setup.runoff.clone(),
            geometry_schedules: setup.geometry_schedules.clone(),
            pool_curve_schedules: setup.pool_curve_schedules.clone(),
            storage_exclusion_evidence: setup.storage_exclusion_evidence.clone(),
            config: setup.config.clone(),
        }
    }

    /// Complete setup with `surface` as the surface definition.
    pub fn with_surface(&self, surface: SolverSetup) -> SurfaceChannelSetup {
        SurfaceChannelSetup {
            surface,
            reaches: self.reaches.clone(),
            links: self.links.clone(),
            ports: self.ports.clone(),
            pools: self.pools.clone(),
            pool_support: self.pool_support.clone(),
            pool_sources: self.pool_sources.clone(),
            pool_inflows: self.pool_inflows.clone(),
            storage: self.storage.clone(),
            structures: self.structures.clone(),
            pool_ends: self.pool_ends.clone(),
            operating: self.operating.clone(),
            runoff: self.runoff.clone(),
            geometry_schedules: self.geometry_schedules.clone(),
            pool_curve_schedules: self.pool_curve_schedules.clone(),
            storage_exclusion_evidence: self.storage_exclusion_evidence.clone(),
            config: self.config.clone(),
        }
    }
}

/// Authoritative coordinator counters at one accepted time.
///
/// Every value is `+0.0` before the first accepted interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelCounterData {
    /// Common accepted time, in seconds.
    pub time_s: f64,
    /// Number of accepted common intervals.
    pub accepted_steps: usize,
    /// Lifetime directional bank link volumes in ascending link identity.
    pub links: Vec<SurfaceChannelLinkLedger>,
    /// Lifetime port volumes and port totals. Empty and zero without ports.
    pub ports: SurfaceChannelPortCounterData,
    /// Lifetime directional structure volumes in ascending structure identity.
    pub structures: Vec<SurfaceChannelStructureLedger>,
    /// Lifetime directional pool-end volumes in ascending pool-end identity.
    pub pool_ends: Vec<SurfaceChannelPoolEndLedger>,
    /// Lifetime external pool inflow ledgers in declared binding order.
    pub pool_inflows: Vec<CoupledPoolInflowLedger>,
    /// Lifetime runoff delivery counters. Empty and zero without runoff.
    pub runoff: SurfaceChannelRunoffCounterData,
}

/// Authoritative runoff delivery counters at one accepted time.
///
/// Every value is `+0.0` before the first accepted interval.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceChannelRunoffCounterData {
    /// Lifetime delivered volume of every delivery lane, in delivery-lane order.
    pub ledgers: Vec<SurfaceChannelRunoffLedger>,
    /// Upward-rounded sum of every interval's span split bound, in m³.
    pub split_bound_m3: f64,
}

/// Authoritative port counters at one accepted time.
///
/// Every value is `+0.0` before the first accepted interval.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceChannelPortCounterData {
    /// Lifetime directional reach volumes in ascending port identity.
    pub ledgers: Vec<SurfaceChannelPortLedger>,
    /// Signed volume of each port over the last accepted interval, in ledger order, in m³.
    ///
    /// The next interval's `reversed` flag compares against it.
    pub last_signed_volume_m3: Vec<f64>,
    /// Lifetime port water the surface exchange totals hold, into the surface, in m³.
    pub surface_in_m3: f64,
    /// Lifetime port water the surface exchange totals hold, out of the surface, in m³.
    pub surface_out_m3: f64,
    /// Upward-rounded sum of every interval's port pairing bound, in m³.
    pub paired_bound_m3: f64,
}

/// Nonrecursive start frame of the last common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPreviousData {
    /// Complete surface package at the interval start.
    pub surface: SnapshotData,
    /// Complete `(reach id, package)` pairs at the interval start, in setup order.
    pub reaches: Vec<(String, ChannelSnapshotData)>,
    /// Complete pool package at the interval start, or `None` without pools.
    pub pools: Option<LevelPoolSnapshotData>,
    /// Complete operating packages at the interval start, or `None` without operating children.
    pub operating: Option<SurfaceChannelOperatingData>,
    /// Complete runoff package at the interval start, or `None` without active runoff.
    pub runoff: Option<SubcatchmentSnapshotData>,
    /// Coordinator counters at the interval start.
    pub counters: SurfaceChannelCounterData,
}

/// Last common interval: its start frame, caller forcing, and accepted receipt.
///
/// Import recalculates the interval from `previous`, decision completion included, and compares
/// every result bit.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelIntervalSnapshotData {
    /// Accepted children and counters before the interval.
    pub previous: SurfaceChannelPreviousData,
    /// Caller channel and pool forcing exactly as the trial received it.
    pub forcing: SurfaceChannelForcingInterval,
    /// Requested and accepted end, retries, iterations, per-link, per-structure, per-port, and
    /// per-pool-end transfers, pools, and midpoint stages.
    pub diagnostics: SurfaceChannelStepDiagnostics,
}

impl SurfaceChannelSnapshotData {
    /// Checks this package identity, then every nested identity. Errors name the package path.
    pub(super) fn check_identities(&self) -> Result<(), String> {
        package_identity(
            "surface-channel",
            &self.schema,
            SURFACE_CHANNEL_SNAPSHOT_SCHEMA,
            &self.contract,
            SURFACE_CHANNEL_CONTRACT_VERSION,
        )?;
        check_frame(
            "",
            (&self.surface, &self.reaches),
            (self.pools.as_ref(), self.operating.as_ref()),
        )?;
        check_runoff("", self.runoff.as_ref())?;
        if let Some(last) = &self.last_interval {
            let previous = &last.previous;
            check_frame(
                "last_interval.previous.",
                (&previous.surface, &previous.reaches),
                (previous.pools.as_ref(), previous.operating.as_ref()),
            )?;
            check_runoff("last_interval.previous.", previous.runoff.as_ref())?;
        }
        Ok(())
    }
}

/// Checks the runoff package identity of one frame.
fn check_runoff(prefix: &str, runoff: Option<&SubcatchmentSnapshotData>) -> Result<(), String> {
    runoff.map_or(Ok(()), |runoff| {
        checkpoint::subcatchment(
            &format!("surface-channel checkpoint {prefix}runoff"),
            runoff,
        )
    })
}

/// Checks the surface, reach, pool, and operating package identities of one frame.
fn check_frame(
    prefix: &str,
    (surface, reaches): (&SnapshotData, &[(String, ChannelSnapshotData)]),
    (pools, operating): (
        Option<&LevelPoolSnapshotData>,
        Option<&SurfaceChannelOperatingData>,
    ),
) -> Result<(), String> {
    checkpoint::surface(
        &format!("surface-channel checkpoint {prefix}surface"),
        surface,
    )?;
    for (index, (id, reach)) in reaches.iter().enumerate() {
        checkpoint::channel(
            &format!("surface-channel checkpoint {prefix}reaches[{index}] ({id:?})"),
            reach,
        )?;
    }
    if let Some(pools) = pools {
        checkpoint::level_pool(&format!("surface-channel checkpoint {prefix}pools"), pools)?;
    }
    if let Some(operating) = operating {
        operating.check_identities(&format!("surface-channel checkpoint {prefix}operating"))?;
    }
    Ok(())
}

/// Trusted native continuation without host projection of numerical fields.
#[derive(Clone)]
pub struct SurfaceChannelSnapshot<R: Runtime<DType = DType>> {
    pub(super) definition: Arc<SurfaceChannelDefinition>,
    pub(super) context: Arc<Context<R>>,
    pub(super) initial: Arc<SurfaceChannelInitialState>,
    pub(super) time_s: f64,
    pub(super) surface: Snapshot<R>,
    /// Reaches in setup order.
    pub(super) reaches: Vec<ChannelSnapshot<R>>,
    pub(super) accepted_steps: usize,
    pub(super) counters: LinkCounters<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
    /// Accepted pools, or `None` without pools.
    pub(super) pools: Option<PoolChild<R>>,
    /// Accepted operating children, or `None` without them.
    pub(super) operating: Option<OperatingChildren<R>>,
    /// Accepted runoff child, or `None` without active runoff.
    pub(super) runoff: Option<SubcatchmentBatch<R>>,
}

impl<R: Runtime<DType = DType>> SurfaceChannelSnapshot<R> {
    /// Common accepted time, in seconds.
    pub fn time(&self) -> f64 {
        self.time_s
    }

    /// Number of accepted common intervals.
    pub fn accepted_steps(&self) -> usize {
        self.accepted_steps
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for SurfaceChannelSnapshot<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SurfaceChannelSnapshot")
            .field("setup", self.definition.setup())
            .field("time_s", &self.time_s)
            .field("accepted_steps", &self.accepted_steps)
            .field("has_last_interval", &self.last.is_some())
            .finish()
    }
}
