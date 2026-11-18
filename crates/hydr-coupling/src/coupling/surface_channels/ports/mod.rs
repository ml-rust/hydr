//! Coordinator reach-end ports: declarations, admission, static topology, and execution.
//!
//! A port binds one contiguous line of surface faces to one end of a rectangular reach.
//! The coordinator owns that end: the caller supplies no endpoint policy there.
//!
//! Each attempt runs one surface HLL flux per port face against the reach end's accepted-start
//! state. The ported reach end then takes the summed rates as a `FluxPort`. The flux stays
//! fixed for the attempt and recalculates after shortening.

mod attempt;
mod binding;
mod counters;
mod endpoint;
mod faces;
#[cfg(test)]
pub(super) mod fixture;
mod ghosts;
mod receipt;
#[cfg(test)]
pub(super) mod scenario;
mod topology;

use super::{
    SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE, SurfaceChannelError, SurfaceChannelPort,
    SurfaceChannelSetup,
};
use crate::channel::{ChannelEnd, ChannelGeometry};
use crate::solver::BoundaryFace;
use faces::{FACE_OWNERSHIP, check_faces};
use std::collections::{BTreeMap, BTreeSet};

pub(super) use binding::PortBinding;
pub(super) use counters::PortCounters;
pub(super) use endpoint::check_endpoint;
pub(super) use ghosts::PortStart;
pub(super) use receipt::PortInterval;
pub(super) use topology::PortTopology;

/// One admitted port with its resolved reach, end cell, and face line.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedPort {
    /// Index into the declared port list.
    pub setup_index: usize,
    /// Index into the setup's reach list.
    pub reach_index: usize,
    pub end: ChannelEnd,
    /// End cell index in reach cell order.
    pub cell: usize,
    /// Faces sorted along their face line.
    pub faces: Vec<BoundaryFace>,
    /// End cell width `b`, in m. The face widths sum to it within the width tolerance.
    pub width_m: f64,
    /// End cell bed elevation in the common datum, in m.
    pub ghost_bed_m: f64,
    /// The ghost lies on the faces' negative side.
    pub ghost_left: bool,
    /// Bed of each face's active surface cell, in face order, in m.
    pub face_beds_m: Vec<f64>,
    /// Gravity of the ported reach, bit-identical to the surface gravity, in m/s².
    pub gravity: f64,
}

/// Position of `end` in a per-reach `[left, right]` pair.
pub(super) fn end_slot(end: ChannelEnd) -> usize {
    match end {
        ChannelEnd::Left => 0,
        ChannelEnd::Right => 1,
    }
}

fn end_name(end: ChannelEnd) -> &'static str {
    match end {
        ChannelEnd::Left => "left",
        ChannelEnd::Right => "right",
    }
}

/// Rejects empty port identity or evidence text and text with control characters.
fn port_text(entry: &str, id: &str, name: &str, value: &str) -> Result<(), SurfaceChannelError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(SurfaceChannelError::input(format!(
            "{entry}: port={id:?} {name}={value:?}: supply nonempty identity and evidence without control characters"
        )));
    }
    Ok(())
}

/// Checks every port of `setup`. Returns the admitted ports in ascending identity order.
///
/// `reach_index` maps each checked reach identity to its setup index.
pub(super) fn check_ports(
    setup: &SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
) -> Result<Vec<CheckedPort>, SurfaceChannelError> {
    let ports: &[SurfaceChannelPort] = &setup.ports;
    let link_ids: BTreeSet<&str> = setup.links.iter().map(|link| link.id.as_str()).collect();
    let open: BTreeSet<BoundaryFace> = setup
        .surface
        .boundaries
        .open_faces()
        .iter()
        .copied()
        .collect();
    let mut claimed = BTreeSet::new();
    let mut ends: BTreeMap<(usize, usize), &str> = BTreeMap::new();
    let mut checked = BTreeMap::new();
    for (position, port) in ports.iter().enumerate() {
        let id = port.id.as_str();
        port_text("missing-link-evidence", id, "id", id)?;
        if checked.contains_key(id) || link_ids.contains(id) {
            return Err(SurfaceChannelError::input(format!(
                "duplicate-links: port id={id:?} repeats a port or link identity: give each port and link a unique identity"
            )));
        }
        port_text(
            "missing-link-evidence",
            id,
            "mapping_evidence",
            &port.mapping_evidence,
        )?;
        let Some(&reach) = reach_index.get(port.reach_id.as_str()) else {
            return Err(SurfaceChannelError::input(format!(
                "{FACE_OWNERSHIP}: port={id:?} reach_id={:?} names no setup reach: use a declared reach identity",
                port.reach_id
            )));
        };
        let end = end_name(port.end);
        let cells = match &setup.reaches[reach].definition.geometry {
            ChannelGeometry::Rectangular { cells } => cells,
            ChannelGeometry::PrismaticTriangular { .. } => {
                return Err(SurfaceChannelError::input(format!(
                    "port-section-mismatch: port={id:?} reach={:?} geometry=triangular: a port needs a rectangular end cell of the face width; use a rectangular reach or connect this reach through bank links",
                    port.reach_id
                )));
            }
            ChannelGeometry::PrismaticSurveyed { .. } => {
                return Err(SurfaceChannelError::input(format!(
                    "surveyed-port-end: port={id:?} reach={:?} geometry=surveyed: surveyed reaches couple through bank links only; use a rectangular reach for a port",
                    port.reach_id
                )));
            }
        };
        let cell = match port.end {
            ChannelEnd::Left => 0,
            ChannelEnd::Right => cells.len().saturating_sub(1),
        };
        let Some(end_cell) = cells.get(cell) else {
            return Err(SurfaceChannelError::input(format!(
                "port-section-mismatch: port={id:?} reach={:?} has no cells: declare the reach cells",
                port.reach_id
            )));
        };
        if let Some(other) = ends.insert((reach, end_slot(port.end)), id) {
            return Err(SurfaceChannelError::input(format!(
                "{FACE_OWNERSHIP}: port={id:?} reach={:?} end={end} is already bound to port={other:?}: bind one port per reach end",
                port.reach_id
            )));
        }
        let line = check_faces(&setup.surface.domain, &open, &mut claimed, id, &port.faces)?;
        let b = end_cell.width_m;
        let tolerance = SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE * b;
        let gap = (line.width_m - b).abs();
        if !b.is_finite() || b <= 0.0 || gap.is_nan() || gap > tolerance {
            return Err(SurfaceChannelError::input(format!(
                "port-width-mismatch: port={id:?} face width={}, reach={:?} end={end} cell={:?} width_m={b}, tolerance={tolerance}: set the end cell width to the face width sum",
                line.width_m, port.reach_id, end_cell.id
            )));
        }
        checked.insert(
            id,
            CheckedPort {
                setup_index: position,
                reach_index: reach,
                end: port.end,
                cell,
                faces: line.faces,
                width_m: b,
                ghost_bed_m: end_cell.bed_elevation_m,
                ghost_left: line.ghost_left,
                face_beds_m: line.beds_m,
                gravity: setup.reaches[reach].definition.config.gravity,
            },
        );
    }
    Ok(checked.into_values().collect())
}

#[cfg(test)]
mod tests {
    use super::fixture::{
        ALPHA_BED, BETA_BED, REACH_WIDTH, TestResult, case, contains, four_ports, ported,
        rejection, x_face, y_face,
    };
    use super::*;
    use ChannelEnd::{Left, Right};

    fn definition(
        setup: SurfaceChannelSetup,
        ports: Vec<SurfaceChannelPort>,
    ) -> Result<super::super::SurfaceChannelDefinition, SurfaceChannelError> {
        super::super::SurfaceChannelDefinition::new(ported(setup, ports))
    }

    #[test]
    fn a_port_on_each_reach_end_admits_with_both_ghost_sides() -> TestResult {
        let (setup, _) = case(&[])?;
        let checked = definition(setup, four_ports())?;
        // Ascending identity: alpha-left, alpha-right, beta-left, beta-right.
        let expected = [
            (
                1,
                0,
                Left,
                0,
                vec![x_face(0, 0), x_face(0, 1)],
                ALPHA_BED,
                true,
            ),
            (
                0,
                0,
                Right,
                2,
                vec![x_face(4, 0), x_face(4, 1)],
                ALPHA_BED,
                false,
            ),
            (
                2,
                1,
                Left,
                0,
                vec![y_face(0, 2), y_face(1, 2)],
                BETA_BED,
                false,
            ),
            (
                3,
                1,
                Right,
                2,
                vec![y_face(0, 0), y_face(1, 0)],
                BETA_BED,
                true,
            ),
        ];
        assert_eq!(checked.ports.len(), expected.len());
        for (port, (setup_index, reach, end, cell, faces, bed, ghost_left)) in
            checked.ports.iter().zip(expected)
        {
            assert_eq!(port.setup_index, setup_index);
            assert_eq!(port.reach_index, reach);
            assert_eq!(port.end, end);
            assert_eq!(port.cell, cell);
            assert_eq!(port.faces, faces);
            assert_eq!(port.width_m.to_bits(), REACH_WIDTH.to_bits());
            assert_eq!(port.ghost_bed_m.to_bits(), bed.to_bits());
            assert_eq!(port.ghost_left, ghost_left);
        }
        // Ports synthesize no lateral bindings.
        assert_eq!(checked.coordinated[1], checked.setup.reaches[1].definition);
        Ok(())
    }

    #[test]
    fn links_may_be_empty_only_beside_ports() -> TestResult {
        let (mut setup, _) = case(&[])?;
        setup.links.clear();
        let checked = definition(setup.clone(), four_ports())?;
        assert!(checked.links.is_empty() && checked.surface_cells.is_empty());
        contains(
            &rejection(setup, Vec::new())?,
            &["links is empty", "supply at least one surface-channel link"],
        )
    }
}
