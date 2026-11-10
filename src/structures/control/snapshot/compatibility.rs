//! Snapshot-specific admission using shared exact immutable control identity.

use super::super::OperatingControlBatch;
pub(super) use super::super::compatibility::{command, option_time, sample};
use crate::structures::{StructureError, StructureInputErrorKind};

pub(super) fn require(condition: bool, message: &str) -> Result<(), StructureError> {
    if condition {
        Ok(())
    } else {
        Err(StructureError::input(
            StructureInputErrorKind::Definition,
            message,
        ))
    }
}
pub(super) fn check(
    batch: &OperatingControlBatch,
    data: &super::OperatingControlSnapshotData,
) -> Result<(), StructureError> {
    require(
        super::super::compatibility::same_original(
            &batch.context,
            &data.elements,
            &data.definitions,
            &data.initial_states,
            &data.schedule,
            &data.timing,
        ),
        "control snapshot definitions differ: restore into an identically constructed actual structure/control batch",
    )
}
/// Checked schedules increase strictly. Total ordering preserves original signed-zero identity.
pub(super) fn contains_original_time(times: &[f64], time: f64) -> bool {
    times
        .binary_search_by(|candidate| candidate.total_cmp(&time))
        .is_ok()
}

pub(super) fn check_batch(
    a: &OperatingControlBatch,
    b: &OperatingControlBatch,
) -> Result<(), StructureError> {
    require(
        super::super::compatibility::same_context(&a.context, &b.context),
        "trusted control snapshot definitions differ from destination context",
    )
}
