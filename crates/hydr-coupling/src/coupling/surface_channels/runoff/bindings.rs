//! Receiving bindings synthesized for runoff delivery lanes.

use super::admission::{CheckedRunoff, RunoffTarget};
use crate::channel::{ChannelLateralBinding, ChannelLateralDirection, ChannelReachDefinition};
use crate::runoff::Subarea;

/// Binding identity prefix reserved for coordinator runoff delivery.
///
/// Caller lateral bindings, caller lateral inputs, and caller pool inflow bindings reject it.
pub(in super::super) const RUNOFF_ATTACHMENT_BINDING_PREFIX: &str = "runoff-attachment:";

/// Pool inflow boundary identity of one lane of `outlet_id`.
pub(super) fn pool_binding_id(outlet_id: &str, subarea: Subarea) -> String {
    format!("{RUNOFF_ATTACHMENT_BINDING_PREFIX}{outlet_id}:{subarea:?}")
}

/// Lateral boundary identity of one lane of `outlet_id` at reach cell `cell`.
pub(super) fn lateral_binding_id(outlet_id: &str, subarea: Subarea, cell: usize) -> String {
    format!("{RUNOFF_ATTACHMENT_BINDING_PREFIX}{outlet_id}:{subarea:?}:{cell}")
}

/// Appends one inflow binding per channel delivery lane of reach `reach` to `definition`.
///
/// `cell_ids` holds the reach's ordered cell identities. Bindings follow delivery-lane order.
pub(in super::super) fn append_lateral_bindings(
    runoff: &CheckedRunoff,
    reach: usize,
    cell_ids: &[&str],
    definition: &mut ChannelReachDefinition,
) {
    for lane in &runoff.lanes {
        let (
            RunoffTarget::Channel {
                reach: target,
                cell,
            },
            Some(boundary_id),
        ) = (lane.target, &lane.binding)
        else {
            continue;
        };
        if target != reach {
            continue;
        }
        let Some(cell_id) = cell_ids.get(cell) else {
            continue;
        };
        definition.lateral_bindings.push(ChannelLateralBinding {
            boundary_id: boundary_id.clone(),
            cell_id: (*cell_id).to_owned(),
            direction: ChannelLateralDirection::Inflow,
            ownership_assertion: lane.ownership.clone(),
        });
    }
}
