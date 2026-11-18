//! Structure links between coordinator nodes: declarations and admission.
//!
//! A structure link carries one admitted structure law between two coordinator nodes. It shares
//! the link table with bank links: one fixed point, one donor tree, one levelling cap, one
//! response step, and one ledger shape. Structure water enters either side with zero velocity.
//!
//! The two endpoints join a surface cell and a channel cell, a surface cell and a pool, a channel
//! cell and a pool, or two pools, in either order. A one-way law sets its direction through the
//! element's direction policy.

mod admission;
mod laws;
mod operations;
#[cfg(test)]
pub(super) mod scenario;

pub(super) use admission::check_structures;
pub(super) use laws::LinkLaw;
pub(super) use operations::LinkOperations;

use crate::structures::{StructureElement, StructureOperatingPoint};

/// One coordinator node a structure attaches to or a control signal samples.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SurfaceChannelNode {
    /// Active global row-major surface cell index.
    SurfaceCell(usize),
    /// Ordered cell index inside the named setup reach.
    ChannelCell {
        /// Exact identity of the setup reach.
        reach_id: String,
        /// Ordered cell index inside the reach.
        cell: usize,
    },
    /// Exact identity of a coordinator pool.
    Pool(String),
}

impl SurfaceChannelNode {
    /// `surface-cell:<n>`, `reach:<id>:cell:<k>`, or `pool:<id>`.
    ///
    /// A structure element's endpoint strings must equal the identities of its nodes.
    pub fn identity(&self) -> String {
        match self {
            Self::SurfaceCell(cell) => format!("surface-cell:{cell}"),
            Self::ChannelCell { reach_id, cell } => format!("reach:{reach_id}:cell:{cell}"),
            Self::Pool(id) => format!("pool:{id}"),
        }
    }
}

/// One structure endpoint: its node and the bed of its connection.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelAttachment {
    /// Node the endpoint joins.
    pub node: SurfaceChannelNode,
    /// Finite connection bed in the common datum, in metres.
    ///
    /// It sits at or above the node bed, or at or above the curve origin of a pool. A weir
    /// crest, gate sill, or orifice sill sits at or above both attachment beds.
    pub bed_elevation_m: f64,
    /// Evidence that the structure connects to the node at this bed.
    pub physical_connection_assertion: String,
}

/// The fixed blockage or condition scenario whose geometry the law parameters hold.
///
/// A scenario is caller geometry with evidence. The coordinator infers no blockage and changes
/// no geometry over time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StructureScenario {
    /// Nonempty scenario identity, unique across the structure links.
    pub id: String,
    /// Nonempty evidence for the scenario geometry.
    pub evidence: String,
}

/// Owner of a structure's operation.
///
/// A dynamic owner binds the structure to one child of the setup's
/// [`super::SurfaceChannelOperating`]. Each structure has exactly one owner.
#[derive(Clone, Debug, PartialEq)]
pub enum SurfaceChannelOperationOwner {
    /// One supplied operation held over every interval. It must drive the law: `Passive` for a
    /// weir or an orifice, a gate opening for a gate, a running or stopped pump for a pump.
    Manual(StructureOperatingPoint),
    /// A gate on a prescribed trajectory. Each interval takes the midpoint of its linear path.
    Prescribed,
    /// A gate on a controlled actuator. Each interval takes the midpoint of its actuator path.
    Actuated,
    /// A pump on a sampled controller. Each interval holds its accepted operation.
    ControlledPump,
}

impl SurfaceChannelOperationOwner {
    /// True for an owner bound to an operating child.
    pub fn is_dynamic(&self) -> bool {
        !matches!(self, Self::Manual(_))
    }
}

/// One structure link between two distinct coordinator nodes.
///
/// The two endpoints join a surface cell and a channel cell, a surface cell or a channel cell and
/// a pool, or two pools, in either order. Water enters either side with zero velocity. A channel
/// endpoint synthesizes `structure-exchange:<id>:inflow` and `structure-exchange:<id>:withdrawal`
/// at its cell.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelStructure {
    /// Unique identity, law, direction, and datum.
    ///
    /// Its endpoint strings equal the node identities of `endpoint_a` and `endpoint_b`. Its datum
    /// equals the surface vertical datum. A pump takes `Forward`; a bidirectional law needs
    /// symmetric reversal evidence.
    pub element: StructureElement,
    /// Endpoint A. Positive signed volumes move water from A to B.
    pub endpoint_a: SurfaceChannelAttachment,
    /// Endpoint B.
    pub endpoint_b: SurfaceChannelAttachment,
    /// Fixed scenario whose geometry the law parameters hold.
    pub scenario: StructureScenario,
    /// Owner of the structure's operation.
    pub operation: SurfaceChannelOperationOwner,
}

/// One admitted structure endpoint with its resolved node index.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum CheckedNode {
    /// Active global row-major surface cell index.
    Surface(usize),
    /// Ordered cell of a setup reach.
    Channel {
        /// Index into the setup's reach list.
        reach_index: usize,
        /// Ordered cell index inside the reach.
        cell: usize,
        /// Identity of the channel cell.
        cell_id: String,
    },
    /// Position of the pool in ascending pool identity.
    Pool(usize),
}

/// One admitted structure link with resolved nodes and synthesized binding identities.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedStructure {
    pub structure: SurfaceChannelStructure,
    /// Resolved endpoints `[A, B]`.
    pub nodes: [CheckedNode; 2],
    /// Attachment beds `[A, B]`, in metres.
    pub beds: [f64; 2],
    /// Operation of every interval under a manual owner.
    ///
    /// A dynamic owner holds its closed state here: a gate opening of zero, or a stopped pump.
    /// Every attempt then replaces it with the interval operation of its operating child.
    pub operating_point: StructureOperatingPoint,
    /// Synthesized inflow binding identity at the channel endpoint.
    pub inflow_binding: String,
    /// Synthesized withdrawal binding identity at the channel endpoint.
    pub withdrawal_binding: String,
}

impl CheckedStructure {
    /// Exact structure identity.
    pub fn id(&self) -> &str {
        &self.structure.element.id
    }

    /// Active surface cell of either endpoint, if any.
    pub fn surface_cell(&self) -> Option<usize> {
        self.nodes.iter().find_map(|node| match node {
            CheckedNode::Surface(cell) => Some(*cell),
            _ => None,
        })
    }

    /// `(reach index, cell, cell identity)` of the channel endpoint, if any.
    pub fn channel(&self) -> Option<(usize, usize, &str)> {
        self.nodes.iter().find_map(|node| match node {
            CheckedNode::Channel {
                reach_index,
                cell,
                cell_id,
            } => Some((*reach_index, *cell, cell_id.as_str())),
            _ => None,
        })
    }
}
