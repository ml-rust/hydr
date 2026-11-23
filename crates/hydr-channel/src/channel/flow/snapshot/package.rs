use super::super::{
    ChannelError, ChannelForcingInterval, ChannelInitialState, ChannelLateralStateData,
    ChannelReach, ChannelReachDefinition, ChannelStepDiagnostics, ChannelWaterLedger,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

/// Schema identity of the channel snapshot package.
pub const CHANNEL_SNAPSHOT_SCHEMA: &str = "hydr-channel-checkpoint-3";
/// Numerical contract identity of the channel snapshot package.
pub const CHANNEL_CONTRACT_VERSION: &str = "hydr-channel-1d-numr-cpu-4";

/// Complete caller-owned continuation values. Preserve every floating-point bit when encoding.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelSnapshotData {
    /// Package schema identity.
    pub schema: String,
    /// Numerical contract identity.
    pub contract: String,
    /// Exact immutable geometry, identities, assumptions, and numerical controls.
    pub definition: ChannelReachDefinition,
    /// Original epoch and authoritative initial conserved state.
    pub initial: ChannelInitialState,
    /// Accepted elapsed time, in seconds.
    pub time_s: f64,
    /// Authoritative current cell water volumes, in m³.
    pub volume_m3: Vec<f64>,
    /// Normalized low parts of the current volumes, in m³.
    pub volume_low_m3: Vec<f64>,
    /// Authoritative current longitudinal momenta, in m⁴/s.
    pub longitudinal_momentum_m4_per_s: Vec<f64>,
    /// Normalized low parts of the current momenta, in m⁴/s.
    pub longitudinal_momentum_low_m4_per_s: Vec<f64>,
    /// Cumulative external-water accounting.
    pub ledger: ChannelWaterLedger,
    /// Authoritative lateral counters and their conservative numerical bounds.
    pub lateral: ChannelLateralStateData,
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
    /// Normalized low parts of the previous volumes, in m³.
    pub previous_volume_low_m3: Vec<f64>,
    /// Previous authoritative longitudinal momenta, in m⁴/s.
    pub previous_longitudinal_momentum_m4_per_s: Vec<f64>,
    /// Normalized low parts of the previous momenta, in m⁴/s.
    pub previous_longitudinal_momentum_low_m4_per_s: Vec<f64>,
    /// Previous cumulative external-water accounting.
    pub previous_ledger: ChannelWaterLedger,
    /// Previous authoritative lateral counters and numerical bounds.
    pub previous_lateral: ChannelLateralStateData,
    /// Accepted count before this interval.
    pub previous_accepted_steps: usize,
    /// Original held endpoint conditions and their exact time support.
    pub forcing: ChannelForcingInterval,
    /// Original requested endpoint before automatic interval shortening, in seconds.
    pub requested_end_s: f64,
    /// Accepted scalar diagnostics and original integrated face and source operands.
    pub diagnostics: ChannelStepDiagnostics,
}

impl ChannelSnapshotData {
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        let interval = match (&self.last_interval, &other.last_interval) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                a.previous_time_s.to_bits() == b.previous_time_s.to_bits()
                    && super::check::values(&a.previous_volume_m3, &b.previous_volume_m3)
                    && super::check::values(&a.previous_volume_low_m3, &b.previous_volume_low_m3)
                    && super::check::values(
                        &a.previous_longitudinal_momentum_low_m4_per_s,
                        &b.previous_longitudinal_momentum_low_m4_per_s,
                    )
                    && super::check::values(
                        &a.previous_longitudinal_momentum_m4_per_s,
                        &b.previous_longitudinal_momentum_m4_per_s,
                    )
                    && super::check::ledger(&a.previous_ledger, &b.previous_ledger)
                    && super::check::lateral_state(&a.previous_lateral, &b.previous_lateral)
                    && a.previous_accepted_steps == b.previous_accepted_steps
                    && super::check::forcing(&a.forcing, &b.forcing)
                    && a.requested_end_s.to_bits() == b.requested_end_s.to_bits()
                    && super::check::diagnostics(&a.diagnostics, &b.diagnostics)
            }
            _ => false,
        };
        interval
            && self.schema == other.schema
            && self.contract == other.contract
            && super::check::definition(&self.definition, &other.definition)
            && super::check::initial(&self.initial, &other.initial)
            && self.time_s.to_bits() == other.time_s.to_bits()
            && super::check::values(&self.volume_m3, &other.volume_m3)
            && super::check::values(&self.volume_low_m3, &other.volume_low_m3)
            && super::check::values(
                &self.longitudinal_momentum_low_m4_per_s,
                &other.longitudinal_momentum_low_m4_per_s,
            )
            && super::check::values(
                &self.longitudinal_momentum_m4_per_s,
                &other.longitudinal_momentum_m4_per_s,
            )
            && super::check::ledger(&self.ledger, &other.ledger)
            && super::check::lateral_state(&self.lateral, &other.lateral)
            && self.accepted_steps == other.accepted_steps
    }
}

/// Trusted complete continuation clone without host projection.
#[derive(Clone)]
pub struct ChannelSnapshot<R: Runtime<DType = DType>> {
    pub(in crate::channel::flow) reach: ChannelReach<R>,
}

impl<R> ChannelSnapshot<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
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
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
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
                previous_volume_m3: last.previous_volume.high.try_to_vec::<f64>()?,
                previous_volume_low_m3: last.previous_volume.low.try_to_vec::<f64>()?,
                previous_longitudinal_momentum_m4_per_s: last
                    .previous_momentum
                    .high
                    .try_to_vec::<f64>()?,
                previous_longitudinal_momentum_low_m4_per_s: last
                    .previous_momentum
                    .low
                    .try_to_vec::<f64>()?,
                previous_ledger: last.previous_ledger.clone(),
                previous_lateral: self
                    .context
                    .lateral
                    .state_data(&self.context.client, &last.previous_lateral)?,
                previous_accepted_steps: last.previous_steps,
                forcing: last.forcing.project(&self.context.client)?,
                requested_end_s: last.requested_end,
                diagnostics: self.last_step()?.ok_or_else(|| {
                    ChannelError::input(
                        "channel checkpoint lacks its final receipt: inspect accepted state",
                    )
                })?,
            })
        } else {
            None
        };
        Ok(ChannelSnapshotData {
            schema: CHANNEL_SNAPSHOT_SCHEMA.into(),
            contract: CHANNEL_CONTRACT_VERSION.into(),
            definition: (*self.context.definition).clone(),
            initial: self.context.original_initial.clone(),
            time_s: self.accepted.time,
            volume_m3: self.accepted.volume.high.try_to_vec::<f64>()?,
            volume_low_m3: self.accepted.volume.low.try_to_vec::<f64>()?,
            longitudinal_momentum_m4_per_s: self.accepted.momentum.high.try_to_vec::<f64>()?,
            longitudinal_momentum_low_m4_per_s: self.accepted.momentum.low.try_to_vec::<f64>()?,
            ledger: self.accepted.ledger.clone(),
            lateral: self
                .context
                .lateral
                .state_data(&self.context.client, &self.accepted.lateral)?,
            accepted_steps: self.accepted.steps,
            last_interval,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::snapshot::fixture::{advanced, walled_reach};

    #[test]
    fn snapshot_round_trip_keeps_current_and_previous_low_bits() -> Result<(), ChannelError> {
        let origin = walled_reach()?;
        let current = advanced(&advanced(&origin)?)?;
        let data = current.snapshot_data()?;
        let interval = data
            .last_interval
            .as_ref()
            .ok_or_else(|| ChannelError::input("test interval absent: retain the accepted step"))?;
        assert_eq!(data.volume_low_m3[0].to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            data.longitudinal_momentum_low_m4_per_s[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            interval.previous_volume_low_m3[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(
            interval.previous_longitudinal_momentum_low_m4_per_s[0].to_bits(),
            0.0_f64.to_bits()
        );
        assert_eq!(origin.import_snapshot(&data)?.snapshot_data()?, data);
        Ok(())
    }

    #[test]
    fn state_volume_equals_the_high_bits() -> Result<(), ChannelError> {
        let reach = advanced(&walled_reach()?)?;
        let data = reach.snapshot_data()?;
        let state = reach.state()?;
        assert_eq!(state[0].volume_m3.to_bits(), data.volume_m3[0].to_bits());
        Ok(())
    }
}
