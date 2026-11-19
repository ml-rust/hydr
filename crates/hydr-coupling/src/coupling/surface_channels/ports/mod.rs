//! Coordinator reach-end ports: declarations, admission, static topology, and execution.
//!
//! A port binds one contiguous line of surface faces to one end of a rectangular reach
//! or of a surveyed reach whose section is a rectangle.
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

use super::bindings::channel_cells;
use super::{
    SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE, SurfaceChannelError, SurfaceChannelPort,
    SurfaceChannelSetup,
};
use crate::channel::{ChannelEnd, ChannelGeometry};
use crate::solver::BoundaryFace;
use faces::{FACE_OWNERSHIP, check_faces};
use hydr_channel::internal::surveyed_rectangle_width;
use std::collections::{BTreeMap, BTreeSet};

pub(super) use binding::PortBinding;
pub(super) use counters::{PortCounters, two_sum};
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
        let definition = &setup.reaches[reach].definition;
        let (count, surveyed_width) = match &definition.geometry {
            ChannelGeometry::Rectangular { cells } => (cells.len(), None),
            ChannelGeometry::PrismaticTriangular { .. } => {
                return Err(SurfaceChannelError::input(format!(
                    "port-section-mismatch: port={id:?} reach={:?} geometry=triangular: a port needs a rectangular end cell of the face width; use a rectangular reach or connect this reach through bank links",
                    port.reach_id
                )));
            }
            ChannelGeometry::PrismaticSurveyed { profile, cells, .. } => {
                let Some(width) = surveyed_rectangle_width(profile) else {
                    return Err(SurfaceChannelError::input(format!(
                        "port-section-mismatch: port={id:?} reach={:?} geometry=surveyed: a port needs a rectangular surveyed section; use a rectangular profile or connect this reach through bank links",
                        port.reach_id
                    )));
                };
                (cells.len(), Some(width))
            }
        };
        let cell = match port.end {
            ChannelEnd::Left => 0,
            ChannelEnd::Right => count.saturating_sub(1),
        };
        let end_cell = match (&definition.geometry, surveyed_width) {
            (ChannelGeometry::Rectangular { cells }, _) => cells
                .get(cell)
                .map(|c| (c.id.as_str(), c.width_m, c.bed_elevation_m)),
            (_, Some(width)) => channel_cells(definition)
                .get(cell)
                .map(|&(cell_id, bed)| (cell_id, width, bed)),
            _ => None,
        };
        let Some((cell_id, b, bed_elevation_m)) = end_cell else {
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
        let tolerance = SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE * b;
        let gap = (line.width_m - b).abs();
        if !b.is_finite() || b <= 0.0 || gap.is_nan() || gap > tolerance {
            return Err(SurfaceChannelError::input(format!(
                "port-width-mismatch: port={id:?} face width={}, reach={:?} end={end} cell={:?} width_m={b}, tolerance={tolerance}: set the end cell width to the face width sum. A surveyed rectangle takes its bottom width as the end cell width",
                line.width_m, port.reach_id, cell_id
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
                ghost_bed_m: bed_elevation_m,
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
        ALPHA, ALPHA_BED, BETA_BED, REACH_WIDTH, TestResult, case, contains, four_ports, port,
        ported, rectangular, rejection, surveyed, surveyed_rectangle, x_face, y_face,
    };
    use super::*;
    use crate::channel::{ChannelReachDefinition, TriangularChannelCell};
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

    /// Admission of a port on the left end of a surveyed `alpha` with the declared `definition`.
    ///
    /// Frozen criteria, fixed before the first run. The port covers the two west faces, 4 m.
    /// - A rectangle of the face width admits. Its port holds width 4 m and the section bed.
    /// - A three-band end and a triangle end reject with `port-section-mismatch`.
    /// - A rectangle of 3 m rejects with `port-width-mismatch`.
    /// - Each rejection leaves the declared setup equal to its copy from before the call.
    fn alpha_left_on(
        definition: ChannelReachDefinition,
    ) -> TestResult<Result<Vec<CheckedPort>, String>> {
        let (mut setup, _) = case(&[])?;
        setup.reaches[0].definition = definition;
        let before = setup.clone();
        let ports = vec![port("p", ALPHA, Left, vec![x_face(0, 0), x_face(0, 1)])];
        let outcome =
            match super::super::SurfaceChannelDefinition::new(ported(setup.clone(), ports)) {
                Ok(checked) => Ok(checked.ports),
                Err(SurfaceChannelError::Input(message)) => Err(message),
                Err(other) => {
                    return Err(format!("expected an input rejection, got {other}").into());
                }
            };
        assert_eq!(setup, before);
        Ok(outcome)
    }

    #[test]
    fn a_surveyed_rectangle_of_the_face_width_admits() -> TestResult {
        let admitted = alpha_left_on(surveyed_rectangle(3, ALPHA_BED, REACH_WIDTH))?;
        let ports = admitted.map_err(|message| format!("a rectangle rejected: {message}"))?;
        let [port] = ports.as_slice() else {
            return Err("one port admits".into());
        };
        assert_eq!(port.cell, 0);
        assert_eq!(port.width_m.to_bits(), REACH_WIDTH.to_bits());
        assert_eq!(port.ghost_bed_m.to_bits(), ALPHA_BED.to_bits());
        Ok(())
    }

    #[test]
    fn a_surveyed_rectangle_of_the_face_width_within_tolerance_admits() -> TestResult {
        let within = REACH_WIDTH * (1.0 + SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE / 2.0);
        let admitted = alpha_left_on(surveyed_rectangle(3, ALPHA_BED, within))?;
        admitted.map_err(|message| format!("a rectangle rejected: {message}"))?;
        Ok(())
    }

    #[test]
    fn a_surveyed_rectangle_off_the_face_width_rejects() -> TestResult {
        let Err(message) = alpha_left_on(surveyed_rectangle(3, ALPHA_BED, 3.0))? else {
            return Err("a 3 m rectangle admitted against 4 m of faces".into());
        };
        contains(
            &message,
            &[
                "port-width-mismatch",
                "port=\"p\"",
                "face width=4",
                "reach=\"alpha\" end=left cell=\"cell-0\" width_m=3",
            ],
        )
    }

    #[test]
    fn a_three_band_surveyed_end_rejects_as_a_section_mismatch() -> TestResult {
        let top = ALPHA_BED + 4.0;
        let three_bands = [
            (0.0, top),
            (1.0, 2.5),
            (1.5, 1.5),
            (2.0, ALPHA_BED),
            (2.5, ALPHA_BED),
            (3.0, 1.5),
            (3.5, 2.5),
            (4.0, top),
        ];
        let Err(message) = alpha_left_on(surveyed(3, ALPHA_BED, &three_bands))? else {
            return Err("a three-band end admitted".into());
        };
        contains(
            &message,
            &[
                "port-section-mismatch",
                "port=\"p\"",
                "reach=\"alpha\"",
                "geometry=surveyed",
                "a port needs a rectangular surveyed section",
            ],
        )
    }

    #[test]
    fn a_triangular_end_rejects_as_a_section_mismatch() -> TestResult {
        let triangle = ChannelReachDefinition {
            geometry: ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: ALPHA_BED,
                cells: (0..3)
                    .map(|i| TriangularChannelCell {
                        id: format!("cell-{i}"),
                        length_m: 2.0,
                        maximum_depth_m: 4.0,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            ..rectangular(3, ALPHA_BED)
        };
        let Err(message) = alpha_left_on(triangle)? else {
            return Err("a triangle end admitted".into());
        };
        contains(&message, &["port-section-mismatch", "geometry=triangular"])
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
