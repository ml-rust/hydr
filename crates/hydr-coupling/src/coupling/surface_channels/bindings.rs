//! Coordinator lateral bindings synthesized from bank-overflow links.

use super::{SurfaceChannelLink, SurfaceChannelReach};
use crate::channel::{
    ChannelGeometry, ChannelLateralBinding, ChannelLateralDirection, ChannelReachDefinition,
};
use hydr_channel::internal::SURFACE_EXCHANGE_BINDING_PREFIX;

/// Ordered cell identities and bed elevations of a rectangular or triangular reach.
///
/// Returns `None` for surveyed geometry, which links do not support yet.
pub(super) fn prismatic_cells(definition: &ChannelReachDefinition) -> Option<Vec<(&str, f64)>> {
    match &definition.geometry {
        ChannelGeometry::Rectangular { cells } => Some(
            cells
                .iter()
                .map(|cell| (cell.id.as_str(), cell.bed_elevation_m))
                .collect(),
        ),
        ChannelGeometry::PrismaticTriangular {
            bed_elevation_m,
            cells,
            ..
        } => Some(
            cells
                .iter()
                .map(|cell| (cell.id.as_str(), *bed_elevation_m))
                .collect(),
        ),
        ChannelGeometry::PrismaticSurveyed { .. } => None,
    }
}

/// Inflow and withdrawal binding identities of one link.
pub(super) fn binding_ids(link_id: &str) -> (String, String) {
    (
        format!("{SURFACE_EXCHANGE_BINDING_PREFIX}{link_id}:inflow"),
        format!("{SURFACE_EXCHANGE_BINDING_PREFIX}{link_id}:withdrawal"),
    )
}

/// Caller definition plus two bindings per link, appended in the order of `links`.
///
/// `links` holds each link with its channel cell identity, already sorted by link identity.
pub(super) fn coordinated_definition(
    reach: &SurfaceChannelReach,
    links: &[(&SurfaceChannelLink, &str)],
    storage_exclusion_evidence: &str,
) -> ChannelReachDefinition {
    let mut definition = reach.definition.clone();
    definition.lateral_bindings.reserve(2 * links.len());
    for (link, cell_id) in links {
        let (inflow, withdrawal) = binding_ids(&link.id);
        let ownership_assertion = format!(
            "surface-channel link {:?}: {storage_exclusion_evidence}",
            link.id
        );
        for (boundary_id, direction) in [
            (inflow, ChannelLateralDirection::Inflow),
            (withdrawal, ChannelLateralDirection::Withdrawal),
        ] {
            definition.lateral_bindings.push(ChannelLateralBinding {
                boundary_id,
                cell_id: (*cell_id).to_string(),
                direction,
                ownership_assertion: ownership_assertion.clone(),
            });
        }
    }
    definition
}
