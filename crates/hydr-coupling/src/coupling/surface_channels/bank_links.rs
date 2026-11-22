//! Admission of bank links: identities, cells, weir geometry, tangent, and crest.

use super::bindings::channel_cells;
use super::validation::text;
use super::{
    SURFACE_CHANNEL_TANGENT_TOLERANCE, SurfaceChannelError, SurfaceChannelLink, SurfaceChannelSetup,
};
use crate::structures::SuppressedRectangularWeirParameters;
use std::collections::{BTreeMap, BTreeSet};

/// One link after its checks, with resolved reach, channel cell identity, and weir law.
pub(super) struct Resolved<'a> {
    pub setup_index: usize,
    pub link: &'a SurfaceChannelLink,
    pub reach_index: usize,
    pub cell_id: &'a str,
    pub weir: SuppressedRectangularWeirParameters,
}

/// Checks every link. Returns the resolved links in ascending link-identity order.
///
/// The links may be empty only beside a port, a structure link, a pool end, or native runoff.
/// `uncoupled` is true without the last three.
pub(super) fn check_links<'a>(
    setup: &'a SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
    uncoupled: bool,
) -> Result<Vec<Resolved<'a>>, SurfaceChannelError> {
    if setup.links.is_empty() && setup.ports.is_empty() && uncoupled {
        return Err(SurfaceChannelError::input(
            "links is empty and ports is empty: supply at least one surface-channel link or port, or native runoff",
        ));
    }
    let domain = &setup.surface.domain;
    let cell_count = domain.grid().cell_count();
    let mut resolved = BTreeMap::new();
    let mut triples = BTreeSet::new();
    for (position, link) in setup.links.iter().enumerate() {
        text("link id", &link.id)?;
        if resolved.contains_key(link.id.as_str()) {
            return Err(SurfaceChannelError::input(format!(
                "link id={:?} repeats: give each link a unique identity",
                link.id
            )));
        }
        let id = &link.id;
        text("mapping_evidence", &link.mapping_evidence)?;
        if !triples.insert((link.surface_cell, link.reach_id.as_str(), link.channel_cell)) {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} repeats surface_cell={}, reach_id={:?}, channel_cell={}: merge the duplicate links into one",
                link.surface_cell, link.reach_id, link.channel_cell
            )));
        }
        if link.surface_cell >= cell_count || !domain.active()[link.surface_cell] {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} surface_cell={}, cells={cell_count}: link an active cell inside the grid",
                link.surface_cell
            )));
        }
        let Some(&reach) = reach_index.get(link.reach_id.as_str()) else {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} reach_id={:?} names no setup reach: use a declared reach identity",
                link.reach_id
            )));
        };
        let cells = channel_cells(&setup.reaches[reach].definition);
        let Some(&(cell_id, channel_bed)) = cells.get(link.channel_cell) else {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} channel_cell={}, reach={:?} cells={}: link an existing reach cell",
                link.channel_cell,
                link.reach_id,
                cells.len()
            )));
        };
        for (name, value) in [
            ("crest_length_m", link.crest_length_m),
            ("weir_coefficient", link.weir_coefficient),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(SurfaceChannelError::input(format!(
                    "link={id:?} {name}={value}: supply a finite positive value"
                )));
            }
        }
        if !link.crest_elevation_m.is_finite() {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} crest_elevation_m={}: supply a finite crest elevation",
                link.crest_elevation_m
            )));
        }
        let [x, y] = link.reach_tangent;
        let length = x.hypot(y);
        if !x.is_finite()
            || !y.is_finite()
            || (length - 1.0).abs() > SURFACE_CHANNEL_TANGENT_TOLERANCE
        {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} reach_tangent=[{x},{y}] has length {length}: supply a finite unit tangent within {SURFACE_CHANNEL_TANGENT_TOLERANCE}"
            )));
        }
        let surface_bed = domain.bed()[link.surface_cell];
        if link.crest_elevation_m < surface_bed || link.crest_elevation_m < channel_bed {
            return Err(SurfaceChannelError::input(format!(
                "link={id:?} crest_elevation_m={} is below surface bed={surface_bed} or channel bed={channel_bed}: an unobstructed connection is unsupported; raise the crest to at least both beds",
                link.crest_elevation_m
            )));
        }
        let weir = weir(link)?;
        resolved.insert(
            link.id.as_str(),
            Resolved {
                setup_index: position,
                link,
                reach_index: reach,
                cell_id,
                weir,
            },
        );
    }
    Ok(resolved.into_values().collect())
}

/// Weir law of a link, checked by the structures crate.
///
/// Links carry water both ways, so the coefficient must hold under reversed flow.
fn weir(
    link: &SurfaceChannelLink,
) -> Result<SuppressedRectangularWeirParameters, SurfaceChannelError> {
    if !link.applicability.symmetric_reversal {
        return Err(SurfaceChannelError::input(format!(
            "link={:?} symmetric_reversal=false: links exchange water in both directions; supply a coefficient that holds under reversed flow",
            link.id
        )));
    }
    SuppressedRectangularWeirParameters::new(
        link.crest_length_m,
        link.crest_elevation_m,
        link.weir_coefficient,
        link.applicability.clone(),
    )
    .map_err(|error| SurfaceChannelError::input(format!("link={:?} weir: {error}", link.id)))
}
