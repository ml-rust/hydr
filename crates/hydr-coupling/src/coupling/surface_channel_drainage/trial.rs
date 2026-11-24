//! Uncommitted 3-way coupled trial and portable snapshot header validation.

use super::batch::SurfaceChannelDrainageBatch;
use super::error::SurfaceChannelDrainageError;
use super::state::{
    SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION, SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA,
    SurfaceChannelDrainageSnapshotData, SurfaceChannelDrainageStepDiagnostics,
};
use crate::coupling::SurfaceChannelBatch;
use crate::drainage::SwmmDrainageSnapshotData;
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Validates schema, contract, vertical datum, transfer vector, and drainage time consistency.
pub(super) fn validate_snapshot_header(
    data: &SurfaceChannelDrainageSnapshotData,
    expected_datum: &str,
    inlet_count: usize,
) -> Result<(), SurfaceChannelDrainageError> {
    if data.schema != SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA {
        return Err(SurfaceChannelDrainageError::input(format!(
            "snapshot schema `{}` does not match `{SURFACE_CHANNEL_DRAINAGE_SNAPSHOT_SCHEMA}`",
            data.schema
        )));
    }
    if data.contract != SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION {
        return Err(SurfaceChannelDrainageError::input(format!(
            "snapshot contract `{}` does not match `{SURFACE_CHANNEL_DRAINAGE_CONTRACT_VERSION}`",
            data.contract
        )));
    }
    if data.vertical_datum != expected_datum {
        return Err(SurfaceChannelDrainageError::input(format!(
            "snapshot vertical_datum `{}` does not match `{expected_datum}`",
            data.vertical_datum
        )));
    }
    if data.cumulative_inlet_transfers_m3.len() != inlet_count
        || !data
            .cumulative_inlet_transfers_m3
            .iter()
            .all(|v| v.is_finite())
    {
        return Err(SurfaceChannelDrainageError::input(
            "snapshot cumulative_inlet_transfers_m3 length or finiteness invalid",
        ));
    }
    if (data.drainage.time_s - data.time_s).abs() > 1e-3 {
        return Err(SurfaceChannelDrainageError::input(format!(
            "snapshot drainage time_s ({}) disagrees with coordinator time_s ({})",
            data.drainage.time_s, data.time_s
        )));
    }
    Ok(())
}

/// Uncommitted conservative 3-way coupled step of a [`SurfaceChannelDrainageBatch`].
///
/// Call [`Self::accept`] to commit surface, channels, pools, structures, and SWMM drainage
/// atomically. Dropping without accepting leaves `surface_channels` unchanged and rolls back
/// the SWMM worker bit-for-bit.
pub struct SurfaceChannelDrainageTrial<'a, R: Runtime<DType = DType>> {
    pub(super) batch: &'a mut SurfaceChannelDrainageBatch<R>,
    pub(super) drainage_backup: SwmmDrainageSnapshotData,
    pub(super) next_sc: Option<SurfaceChannelBatch<R>>,
    pub(super) step_transfers_m3: Vec<f64>,
    pub(super) diagnostics: SurfaceChannelDrainageStepDiagnostics,
    pub(super) accepted: bool,
}

impl<R: Runtime<DType = DType>> SurfaceChannelDrainageTrial<'_, R> {
    /// Proposed 3-way step diagnostics, channel transfers, and inlet transfers.
    pub fn diagnostics(&self) -> &SurfaceChannelDrainageStepDiagnostics {
        &self.diagnostics
    }

    /// Commits 2D surface, 1D channels/pools/structures, and SWMM drainage atomically.
    pub fn accept(mut self) -> SurfaceChannelDrainageStepDiagnostics {
        self.accepted = true;
        if let Some(next_sc) = self.next_sc.take() {
            self.batch
                .commit_trial(next_sc, self.diagnostics.end_s, &self.step_transfers_m3);
        }
        self.diagnostics.clone()
    }
}

impl<R: Runtime<DType = DType>> Drop for SurfaceChannelDrainageTrial<'_, R> {
    fn drop(&mut self) {
        if !self.accepted {
            self.batch.rollback_drainage(&self.drainage_backup);
        }
    }
}
