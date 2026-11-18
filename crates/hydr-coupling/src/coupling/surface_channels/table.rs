//! One link table for bank links and structure links, in ascending identity.
//!
//! A bank link is a bidirectional passive weir between its surface cell and channel cell, on the
//! cell beds, with tangent momentum closure. A structure link carries its own law on its
//! attachment beds, and its water enters either side without velocity.

use super::SurfaceChannelDefinition;
use crate::structures::{DirectionPolicy, StructureElement, StructureLaw, StructureOperatingPoint};
use std::collections::BTreeMap;

/// One entry of the link table: a bank link or a structure link.
pub(super) struct TableLink<'a> {
    pub id: &'a str,
    pub reach_index: usize,
    pub surface_lane: usize,
    pub channel_cell: usize,
    pub element: StructureElement,
    pub operation: StructureOperatingPoint,
    /// Attachment beds `[A, B]`, or `None` for a bank link on the cell beds.
    pub beds: Option<[f64; 2]>,
    pub tangent: [f64; 2],
    /// Synthesized `(inflow, withdrawal)` channel binding identities.
    pub bindings: (&'a str, &'a str),
    pub bank: bool,
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
                reach_index: link.reach_index,
                surface_lane: link.surface_lane,
                channel_cell: link.channel_cell,
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
                },
                operation: StructureOperatingPoint::Passive,
                beds: None,
                tangent: declared.reach_tangent,
                bindings: (&link.inflow_binding, &link.withdrawal_binding),
                bank: true,
            },
        );
    }
    let cells = definition.surface_cells();
    for structure in &definition.structures {
        table.insert(
            structure.id(),
            TableLink {
                id: structure.id(),
                reach_index: structure.reach_index,
                surface_lane: cells.partition_point(|&cell| cell < structure.surface_cell),
                channel_cell: structure.channel_cell,
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
