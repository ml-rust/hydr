use super::super::{
    ChannelBoundaryInterval, ChannelError, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, ChannelStepDiagnostics, ChannelWaterLedger,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

/// Complete caller-owned continuation values. Preserve every floating-point bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSnapshotData {
    /// Exact immutable geometry, identities, assumptions, and numerical controls.
    pub definition: ChannelReachDefinition,
    /// Original epoch and authoritative initial conserved state.
    pub initial: ChannelInitialState,
    /// Accepted elapsed time, in seconds.
    pub time_s: f64,
    /// Authoritative current cell water volumes, in m³.
    pub volume_m3: Vec<f64>,
    /// Authoritative current longitudinal momenta, in m⁴/s.
    pub longitudinal_momentum_m4_per_s: Vec<f64>,
    /// Cumulative external-water accounting.
    pub ledger: ChannelWaterLedger,
    /// Number of accepted intervals since the original epoch.
    pub accepted_steps: usize,
    /// Complete bounded final interval, absent before any accepted step.
    pub last_interval: Option<ChannelIntervalSnapshotData>,
}

/// One nonrecursive start frame and its accepted interval receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelIntervalSnapshotData {
    /// Previous accepted elapsed time, in seconds.
    pub previous_time_s: f64,
    /// Previous authoritative water volumes, in m³.
    pub previous_volume_m3: Vec<f64>,
    /// Previous authoritative longitudinal momenta, in m⁴/s.
    pub previous_longitudinal_momentum_m4_per_s: Vec<f64>,
    /// Previous cumulative external-water accounting.
    pub previous_ledger: ChannelWaterLedger,
    /// Accepted count before this interval.
    pub previous_accepted_steps: usize,
    /// Original held endpoint conditions and their exact time support.
    pub forcing: ChannelBoundaryInterval,
    /// Original requested endpoint before automatic interval shortening, in seconds.
    pub requested_end_s: f64,
    /// Accepted scalar diagnostics and original integrated face and source operands.
    pub diagnostics: ChannelStepDiagnostics,
}

/// Trusted complete continuation clone without host projection.
#[derive(Clone)]
pub struct ChannelSnapshot<R: Runtime<DType = DType>> {
    pub(super) reach: ChannelReach<R>,
}

impl<R> ChannelSnapshot<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Project complete values for caller-owned persistence.
    pub fn data(&self) -> Result<ChannelSnapshotData, ChannelError> {
        self.reach.snapshot_data()
    }

    /// Accepted elapsed time, in seconds.
    pub fn time(&self) -> f64 {
        self.reach.time()
    }
}

impl<R> ChannelReach<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    /// Clone complete accepted continuation state without host projection.
    pub fn snapshot(&self) -> ChannelSnapshot<R> {
        ChannelSnapshot {
            reach: self.clone(),
        }
    }

    /// Project complete bounded continuation values on explicit request.
    pub fn snapshot_data(&self) -> Result<ChannelSnapshotData, ChannelError> {
        let last_interval = if let Some(last) = &self.accepted.last {
            Some(ChannelIntervalSnapshotData {
                previous_time_s: last.previous_time,
                previous_volume_m3: last.previous_volume.try_to_vec::<f64>()?,
                previous_longitudinal_momentum_m4_per_s: last
                    .previous_momentum
                    .try_to_vec::<f64>()?,
                previous_ledger: last.previous_ledger.clone(),
                previous_accepted_steps: last.previous_steps,
                forcing: last.forcing.clone(),
                requested_end_s: last.requested_end,
                diagnostics: last.project()?,
            })
        } else {
            None
        };
        Ok(ChannelSnapshotData {
            definition: (*self.context.definition).clone(),
            initial: self.context.original_initial.clone(),
            time_s: self.accepted.time,
            volume_m3: self.accepted.volume.try_to_vec::<f64>()?,
            longitudinal_momentum_m4_per_s: self.accepted.momentum.try_to_vec::<f64>()?,
            ledger: self.accepted.ledger.clone(),
            accepted_steps: self.accepted.steps,
            last_interval,
        })
    }
}
