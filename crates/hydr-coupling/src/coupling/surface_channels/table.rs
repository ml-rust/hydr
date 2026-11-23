//! One link table for bank links and structure links, in ascending identity.
//!
//! A bank link is a bidirectional passive weir between its surface cell and channel cell, on the
//! cell beds, with tangent momentum closure. A structure link carries its own law on its
//! attachment beds between any admitted node pair, and its water enters either side without
//! velocity.

use super::SurfaceChannelDefinition;
use super::structures::CheckedNode;
use crate::structures::{
    DirectionPolicy, StructureCondition, StructureElement, StructureLaw, StructureOperatingPoint,
};
use std::collections::BTreeMap;

/// One endpoint of a table link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum TableEnd {
    /// Active global row-major surface cell index.
    Surface(usize),
    /// Ordered cell of a setup reach.
    Channel { reach_index: usize, cell: usize },
    /// Position of the pool in ascending pool identity.
    Pool(usize),
}

impl TableEnd {
    fn of(node: &CheckedNode) -> Self {
        match node {
            CheckedNode::Surface(cell) => Self::Surface(*cell),
            CheckedNode::Channel {
                reach_index, cell, ..
            } => Self::Channel {
                reach_index: *reach_index,
                cell: *cell,
            },
            CheckedNode::Pool(position) => Self::Pool(*position),
        }
    }
}

/// One entry of the link table: a bank link or a structure link.
pub(super) struct TableLink<'a> {
    pub id: &'a str,
    /// Endpoints `[A, B]`.
    pub ends: [TableEnd; 2],
    pub element: StructureElement,
    pub operation: StructureOperatingPoint,
    /// Attachment beds `[A, B]`, or `None` for a bank link on the cell beds.
    pub beds: Option<[f64; 2]>,
    pub tangent: [f64; 2],
    /// Synthesized `(inflow, withdrawal)` channel binding identities.
    pub bindings: (&'a str, &'a str),
    pub bank: bool,
}

impl TableLink<'_> {
    /// Side `0` (A) or `1` (B) of the channel endpoint, if any.
    pub fn channel_side(&self) -> Option<usize> {
        self.ends
            .iter()
            .position(|end| matches!(end, TableEnd::Channel { .. }))
    }
}

/// Bank and structure links of `definition` in ascending identity.
pub(super) fn table(definition: &SurfaceChannelDefinition) -> Vec<TableLink<'_>> {
    let setup = definition.setup();
    let datum = setup.surface.domain.reference().vertical_datum();
    let mut table: BTreeMap<&str, TableLink<'_>> = BTreeMap::new();
    for link in &definition.links {
        let declared = &setup.links[link.setup_index];
        table.insert(
            declared.id.as_str(),
            TableLink {
                id: declared.id.as_str(),
                ends: [
                    TableEnd::Surface(declared.surface_cell),
                    TableEnd::Channel {
                        reach_index: link.reach_index,
                        cell: link.channel_cell,
                    },
                ],
                element: StructureElement {
                    id: declared.id.clone(),
                    endpoint_a: format!("surface-cell:{}", declared.surface_cell),
                    endpoint_b: format!(
                        "reach:{}:cell:{}",
                        setup.reaches[link.reach_index].id, link.channel_cell
                    ),
                    datum: datum.to_string(),
                    direction: DirectionPolicy::Bidirectional,
                    law: StructureLaw::SuppressedRectangularWeir(link.weir.clone()),
                    condition: StructureCondition::AsBuilt,
                },
                operation: StructureOperatingPoint::Passive,
                beds: None,
                tangent: declared.reach_tangent,
                bindings: (&link.inflow_binding, &link.withdrawal_binding),
                bank: true,
            },
        );
    }
    for structure in &definition.structures {
        table.insert(
            structure.id(),
            TableLink {
                id: structure.id(),
                ends: structure.nodes.each_ref().map(TableEnd::of),
                element: structure.structure.element.clone(),
                operation: structure.operating_point,
                beds: Some(structure.beds),
                tangent: [0.0; 2],
                bindings: (&structure.inflow_binding, &structure.withdrawal_binding),
                bank: false,
            },
        );
    }
    table.into_values().collect()
}
