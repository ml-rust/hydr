//! Coordinator lateral bindings synthesized from bank and structure links.

use super::SurfaceChannelReach;
use crate::channel::{
    ChannelGeometry, ChannelLateralBinding, ChannelLateralDirection, ChannelReachDefinition,
};
use hydr_channel::internal::SURFACE_EXCHANGE_BINDING_PREFIX;

/// Lateral binding identity prefix reserved for coordinator structure exchange.
///
/// Caller definitions and caller forcing reject it.
pub(super) const STRUCTURE_EXCHANGE_BINDING_PREFIX: &str = "structure-exchange:";

/// One link's synthesized bindings at its channel cell.
#[derive(Clone, Copy)]
pub(super) struct LinkBindings<'a> {
    /// `surface-channel link` or `surface-channel structure`.
    pub owner: &'static str,
    pub id: &'a str,
    pub cell_id: &'a str,
    pub inflow: &'a str,
    pub withdrawal: &'a str,
}

/// Ordered cell identities and bed elevations of a reach.
///
/// A surveyed cell's bed is the minimum elevation of the reach's surveyed section.
/// The reach reports stage above that elevation, so links compare crests against it.
pub(super) fn channel_cells(definition: &ChannelReachDefinition) -> Vec<(&str, f64)> {
    match &definition.geometry {
        ChannelGeometry::Rectangular { cells } => cells
            .iter()
            .map(|cell| (cell.id.as_str(), cell.bed_elevation_m))
            .collect(),
        ChannelGeometry::PrismaticTriangular {
            bed_elevation_m,
            cells,
            ..
        } => cells
            .iter()
            .map(|cell| (cell.id.as_str(), *bed_elevation_m))
            .collect(),
        ChannelGeometry::PrismaticSurveyed { profile, cells, .. } => {
            let bed = profile
                .points
                .iter()
                .map(|point| point.elevation_m)
                .reduce(f64::min)
                .unwrap_or(profile.maximum_stage_m);
            cells.iter().map(|cell| (cell.id.as_str(), bed)).collect()
        }
    }
}

/// Inflow and withdrawal binding identities of one link.
pub(super) fn binding_ids(link_id: &str) -> (String, String) {
    (
        format!("{SURFACE_EXCHANGE_BINDING_PREFIX}{link_id}:inflow"),
        format!("{SURFACE_EXCHANGE_BINDING_PREFIX}{link_id}:withdrawal"),
    )
}

/// Inflow and withdrawal binding identities of one structure link.
pub(super) fn structure_binding_ids(structure_id: &str) -> (String, String) {
    (
        format!("{STRUCTURE_EXCHANGE_BINDING_PREFIX}{structure_id}:inflow"),
        format!("{STRUCTURE_EXCHANGE_BINDING_PREFIX}{structure_id}:withdrawal"),
    )
}

/// Caller definition plus two bindings per link, appended in the order of `links`.
///
/// `links` holds each link of the reach, already sorted by link identity.
pub(super) fn coordinated_definition(
    reach: &SurfaceChannelReach,
    links: &[LinkBindings<'_>],
    storage_exclusion_evidence: &str,
) -> ChannelReachDefinition {
    let mut definition = reach.definition.clone();
    definition.lateral_bindings.reserve(2 * links.len());
    for link in links {
        let ownership_assertion =
            format!("{} {:?}: {storage_exclusion_evidence}", link.owner, link.id);
        for (boundary_id, direction) in [
            (link.inflow, ChannelLateralDirection::Inflow),
            (link.withdrawal, ChannelLateralDirection::Withdrawal),
        ] {
            definition.lateral_bindings.push(ChannelLateralBinding {
                boundary_id: boundary_id.to_string(),
                cell_id: link.cell_id.to_string(),
                direction,
                ownership_assertion: ownership_assertion.clone(),
            });
        }
    }
    definition
}
