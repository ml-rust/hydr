//! Owned surface/channel checkpoint packages and the trusted in-memory snapshot.

use super::context::Context;
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelConfig, SurfaceChannelDefinition, SurfaceChannelForcingInterval,
    SurfaceChannelInitialState, SurfaceChannelLink, SurfaceChannelLinkLedger, SurfaceChannelReach,
    SurfaceChannelSetup, SurfaceChannelStepDiagnostics,
};
use crate::channel::{ChannelSnapshot, ChannelSnapshotData};
use crate::coupling::checkpoint;
use crate::solver::{Snapshot, SnapshotData, SolverSetup};
use hydr_core::contract::package_identity;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Schema identity of the surface/channel snapshot package.
pub const SURFACE_CHANNEL_SNAPSHOT_SCHEMA: &str = "hydr-surface-channel-checkpoint-3";
/// Numerical contract identity of the surface/channel snapshot package.
pub const SURFACE_CHANNEL_CONTRACT_VERSION: &str = "hydr-surface-channel-numr-cpu-3";

/// Complete caller-owned continuation. Preserve every floating-point bit when encoding.
///
/// The nested surface package holds the surface definition and its bounded forcing window.
/// Import checks consistency and replays the last interval. It establishes no authenticity.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Reaches, links, evidence, and common controls. The surface setup lives in `surface`.
    pub setup: SurfaceChannelSetupData,
    /// Original common epoch and authoritative original child states.
    pub initial: SurfaceChannelInitialState,
    /// Complete accepted surface package. It carries coordinator exchange water.
    pub surface: SnapshotData,
    /// Complete accepted `(reach id, package)` pairs in setup order.
    pub reaches: Vec<(String, ChannelSnapshotData)>,
    /// Lifetime link counters, accepted count, and common time.
    pub counters: SurfaceChannelCounterData,
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
    /// Evidence that surface cells and channel cells store no water twice.
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
            storage_exclusion_evidence: self.storage_exclusion_evidence.clone(),
            config: self.config.clone(),
        }
    }
}

/// Authoritative coordinator counters at one accepted time.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelCounterData {
    /// Common accepted time, in seconds.
    pub time_s: f64,
    /// Number of accepted common intervals.
    pub accepted_steps: usize,
    /// Lifetime directional link volumes in ascending link identity.
    pub links: Vec<SurfaceChannelLinkLedger>,
}

/// Nonrecursive start frame of the last common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPreviousData {
    /// Complete surface package at the interval start.
    pub surface: SnapshotData,
    /// Complete `(reach id, package)` pairs at the interval start, in setup order.
    pub reaches: Vec<(String, ChannelSnapshotData)>,
    /// Coordinator counters at the interval start.
    pub counters: SurfaceChannelCounterData,
}

/// Last common interval: its start frame, caller forcing, and accepted receipt.
///
/// Import recalculates the interval from `previous` and compares every result bit.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelIntervalSnapshotData {
    /// Accepted children and counters before the interval.
    pub previous: SurfaceChannelPreviousData,
    /// Caller channel forcing exactly as the trial received it.
    pub forcing: SurfaceChannelForcingInterval,
    /// Requested and accepted end, retries, iterations, per-link transfers, and midpoint stages.
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
        check_frame("", &self.surface, &self.reaches)?;
        if let Some(last) = &self.last_interval {
            check_frame(
                "last_interval.previous.",
                &last.previous.surface,
                &last.previous.reaches,
            )?;
        }
        Ok(())
    }
}

/// Checks the surface and reach package identities of one frame.
fn check_frame(
    prefix: &str,
    surface: &SnapshotData,
    reaches: &[(String, ChannelSnapshotData)],
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
