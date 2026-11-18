//! Structure links between a surface cell and a channel cell: declarations and admission.
//!
//! A structure link carries one admitted structure law between two coordinator nodes. It shares
//! the link table with bank links: one fixed point, one donor tree, one levelling cap, one
//! response step, and one ledger shape. Structure water enters either side with zero velocity.
//!
//! Endpoint A is a surface cell and endpoint B a channel cell. A one-way law sets its direction
//! through the element's direction policy.

mod admission;
mod laws;
#[cfg(test)]
pub(super) mod scenario;

pub(super) use admission::check_structures;
pub(super) use laws::LinkLaw;

use crate::structures::{StructureElement, StructureOperatingPoint};

/// One coordinator node a structure attaches to.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "in-crate callers construct structure links")
)]
pub(crate) enum SurfaceChannelNode {
    /// Active global row-major surface cell index.
    SurfaceCell(usize),
    /// Ordered cell index inside the named setup reach.
    ChannelCell {
        /// Exact identity of the setup reach.
        reach_id: String,
        /// Ordered cell index inside the reach.
        cell: usize,
    },
}

impl SurfaceChannelNode {
    /// `surface-cell:<n>` or `reach:<id>:cell:<k>`.
    pub(crate) fn identity(&self) -> String {
        match self {
            Self::SurfaceCell(cell) => format!("surface-cell:{cell}"),
            Self::ChannelCell { reach_id, cell } => format!("reach:{reach_id}:cell:{cell}"),
        }
    }
}

/// One structure endpoint: its node and the bed of its connection.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SurfaceChannelAttachment {
    pub node: SurfaceChannelNode,
    /// Connection bed in the common datum, in metres. At least the node bed.
    pub bed_elevation_m: f64,
    /// Evidence that the structure connects to the node at this bed.
    pub physical_connection_assertion: String,
}

/// The fixed blockage or condition scenario whose geometry the law parameters hold.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct StructureScenario {
    /// Unique nonempty scenario identity.
    pub id: String,
    /// Evidence for the scenario geometry.
    pub evidence: String,
}

/// Owner of a structure's operation.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "in-crate callers construct structure links")
)]
pub(crate) enum SurfaceChannelOperationOwner {
    /// One supplied operation held over every interval.
    Manual(StructureOperatingPoint),
}

/// One structure link between a surface cell and a channel cell.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SurfaceChannelStructure {
    /// Law, direction, and datum. Its endpoint strings equal the node identities.
    pub element: StructureElement,
    /// Surface-cell endpoint.
    pub endpoint_a: SurfaceChannelAttachment,
    /// Channel-cell endpoint.
    pub endpoint_b: SurfaceChannelAttachment,
    pub scenario: StructureScenario,
    pub operation: SurfaceChannelOperationOwner,
}

/// One admitted structure link with resolved indices and synthesized binding identities.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedStructure {
    pub structure: SurfaceChannelStructure,
    /// Active global row-major surface cell index of endpoint A.
    pub surface_cell: usize,
    /// Index into the setup's reach list.
    pub reach_index: usize,
    /// Ordered cell index of endpoint B inside its reach.
    pub channel_cell: usize,
    /// Identity of the channel cell.
    pub cell_id: String,
    /// Attachment beds `[A, B]`, in metres.
    pub beds: [f64; 2],
    /// Operation of every interval.
    pub operating_point: StructureOperatingPoint,
    /// Synthesized inflow binding identity.
    pub inflow_binding: String,
    /// Synthesized withdrawal binding identity.
    pub withdrawal_binding: String,
}

impl CheckedStructure {
    /// Exact structure identity.
    pub fn id(&self) -> &str {
        &self.structure.element.id
    }
}
