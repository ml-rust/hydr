//! Construction checks that turn a setup into a checked surface/channel definition.

use super::bindings::{binding_ids, coordinated_definition, prismatic_cells};
use super::{
    SURFACE_CHANNEL_TANGENT_TOLERANCE, SurfaceChannelError, SurfaceChannelLink, SurfaceChannelSetup,
};
use crate::channel::ChannelReachDefinition;
use crate::structures::SuppressedRectangularWeirParameters;
use hydr_channel::internal::SURFACE_EXCHANGE_BINDING_PREFIX;
use std::collections::{BTreeMap, BTreeSet};

/// One checked link with resolved indices, weir law, and synthesized binding identities.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckedLink {
    /// Index into the setup's declared link list.
    pub setup_index: usize,
    /// Index into the setup's reach list.
    pub reach_index: usize,
    /// Lane of the linked surface cell in `SurfaceChannelDefinition::surface_cells`.
    pub surface_lane: usize,
    /// Ordered cell index inside the linked reach.
    pub channel_cell: usize,
    /// Checked weir geometry, coefficient, and applicability.
    pub weir: SuppressedRectangularWeirParameters,
    /// Synthesized inflow binding identity.
    pub inflow_binding: String,
    /// Synthesized withdrawal binding identity.
    pub withdrawal_binding: String,
}

/// Checked surface/channel setup with synthesized coordinator reach definitions.
///
/// [`SurfaceChannelDefinition::new`] runs every construction check that needs no runtime.
/// The coordinator batch builds its children from this value alone.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelDefinition {
    pub(super) setup: SurfaceChannelSetup,
    /// Coordinator reach definitions in setup reach order.
    pub(super) coordinated: Vec<ChannelReachDefinition>,
    /// Checked links in ascending link-identity order.
    pub(super) links: Vec<CheckedLink>,
    /// Unique linked surface cells in ascending order. Surface exchange lanes follow it.
    pub(super) surface_cells: Vec<usize>,
}

impl SurfaceChannelDefinition {
    /// Checks `setup` and synthesizes two coordinator lateral bindings per link.
    ///
    /// Each linked reach receives `surface-exchange:<link id>:inflow` and
    /// `surface-exchange:<link id>:withdrawal` at the link's channel cell.
    /// They follow the caller bindings, in ascending link-identity order.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Surface`] when the surface config fails its own check.
    /// Returns [`SurfaceChannelError::Input`] for any rule below:
    /// - config tolerances, durations, or work limits fail [`super::SurfaceChannelConfig::check`]
    /// - evidence or identity text is empty or holds control characters
    /// - no reaches, no links, or a repeated reach or link identity
    /// - a repeated `(surface_cell, reach_id, channel_cell)` triple
    /// - a surface cell outside the grid or inactive
    /// - an unknown reach identity or a channel cell out of range
    /// - a linked reach with surveyed geometry
    /// - a nonfinite or nonpositive crest length or weir coefficient, or a nonfinite crest
    /// - incomplete weir applicability evidence
    /// - a reach tangent that is nonfinite or not unit length
    /// - a crest below the surface bed or the channel bed
    /// - gravity or vertical datum differing between the surface and a reach
    /// - a caller lateral binding using the `surface-exchange:` prefix
    /// - `maximum_step_s` above the surface or a reach `max_step`
    /// - `minimum_step_s` below the surface or a reach `min_step`
    /// - `retry_limit` above the surface `max_retries`
    pub fn new(setup: SurfaceChannelSetup) -> Result<Self, SurfaceChannelError> {
        setup.config.check()?;
        text(
            "storage_exclusion_evidence",
            &setup.storage_exclusion_evidence,
        )?;
        let surface = &setup.surface;
        surface.config.check()?;
        let config = &setup.config;
        if config.maximum_step_s > surface.config.max_step {
            return Err(SurfaceChannelError::input(format!(
                "config maximum_step_s={}, surface max_step={}: lower maximum_step_s to at most the surface max_step",
                config.maximum_step_s, surface.config.max_step
            )));
        }
        if config.minimum_step_s < surface.config.min_step {
            return Err(SurfaceChannelError::input(format!(
                "config minimum_step_s={}, surface min_step={}: raise minimum_step_s to at least the surface min_step",
                config.minimum_step_s, surface.config.min_step
            )));
        }
        if config.retry_limit > surface.config.max_retries {
            return Err(SurfaceChannelError::input(format!(
                "config retry_limit={}, surface max_retries={}: lower retry_limit to at most the surface max_retries",
                config.retry_limit, surface.config.max_retries
            )));
        }
        let reach_index = check_reaches(&setup)?;
        let resolved = check_links(&setup, &reach_index)?;
        let surface_cells: Vec<usize> = setup
            .links
            .iter()
            .map(|link| link.surface_cell)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut links = Vec::with_capacity(resolved.len());
        let mut per_reach: Vec<Vec<(&SurfaceChannelLink, &str)>> =
            vec![Vec::new(); setup.reaches.len()];
        for resolved in resolved {
            let link = resolved.link;
            if let Some(reach_links) = per_reach.get_mut(resolved.reach_index) {
                reach_links.push((link, resolved.cell_id));
            }
            let (inflow_binding, withdrawal_binding) = binding_ids(&link.id);
            links.push(CheckedLink {
                setup_index: resolved.setup_index,
                reach_index: resolved.reach_index,
                surface_lane: surface_cells.partition_point(|&cell| cell < link.surface_cell),
                channel_cell: link.channel_cell,
                weir: resolved.weir,
                inflow_binding,
                withdrawal_binding,
            });
        }
        let coordinated = setup
            .reaches
            .iter()
            .zip(&per_reach)
            .map(|(reach, links)| {
                coordinated_definition(reach, links, &setup.storage_exclusion_evidence)
            })
            .collect();
        Ok(Self {
            setup,
            coordinated,
            links,
            surface_cells,
        })
    }

    /// Immutable caller setup.
    pub fn setup(&self) -> &SurfaceChannelSetup {
        &self.setup
    }

    /// Coordinator definition of the named reach, with synthesized bindings appended.
    ///
    /// A reach without links keeps the caller definition unchanged.
    pub fn coordinated_reach(&self, id: &str) -> Option<&ChannelReachDefinition> {
        self.setup
            .reaches
            .iter()
            .position(|reach| reach.id == id)
            .and_then(|index| self.coordinated.get(index))
    }

    /// Unique linked surface cells in ascending order.
    pub fn surface_cells(&self) -> &[usize] {
        &self.surface_cells
    }

    /// Links in ascending link-identity order, the coordinator's exchange order.
    pub fn links(&self) -> impl ExactSizeIterator<Item = &SurfaceChannelLink> {
        self.links
            .iter()
            .map(|link| &self.setup.links[link.setup_index])
    }
}

/// Rejects empty identity or evidence text and text with control characters.
fn text(name: &str, value: &str) -> Result<(), SurfaceChannelError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(SurfaceChannelError::input(format!(
            "{name}={value:?}: supply nonempty identity and evidence without control characters"
        )));
    }
    Ok(())
}

/// Checks reach identities and surface compatibility. Returns each reach's index by identity.
fn check_reaches(
    setup: &SurfaceChannelSetup,
) -> Result<BTreeMap<&str, usize>, SurfaceChannelError> {
    if setup.reaches.is_empty() {
        return Err(SurfaceChannelError::input(
            "reaches is empty: supply at least one channel reach",
        ));
    }
    let surface = &setup.surface;
    let datum = surface.domain.reference().vertical_datum();
    let gravity = surface.config.gravity;
    let mut index = BTreeMap::new();
    for (position, reach) in setup.reaches.iter().enumerate() {
        text("reach id", &reach.id)?;
        if index.insert(reach.id.as_str(), position).is_some() {
            return Err(SurfaceChannelError::input(format!(
                "reach id={:?} repeats: give each reach a unique identity",
                reach.id
            )));
        }
        let definition = &reach.definition;
        if definition.vertical_datum != datum {
            return Err(SurfaceChannelError::input(format!(
                "reach={:?} vertical_datum={:?}, surface vertical_datum={datum:?}: express both in one vertical datum",
                reach.id, definition.vertical_datum
            )));
        }
        if definition.config.gravity.to_bits() != gravity.to_bits() {
            return Err(SurfaceChannelError::input(format!(
                "reach={:?} gravity={}, surface gravity={gravity}: supply bit-identical gravity",
                reach.id, definition.config.gravity
            )));
        }
        if let Some(binding) = definition.lateral_bindings.iter().find(|binding| {
            binding
                .boundary_id
                .starts_with(SURFACE_EXCHANGE_BINDING_PREFIX)
        }) {
            return Err(SurfaceChannelError::input(format!(
                "reach={:?} lateral binding {:?} uses the reserved prefix {SURFACE_EXCHANGE_BINDING_PREFIX:?}: rename the binding; the coordinator synthesizes surface-exchange bindings from links",
                reach.id, binding.boundary_id
            )));
        }
        let config = &setup.config;
        if config.maximum_step_s > definition.config.max_step {
            return Err(SurfaceChannelError::input(format!(
                "config maximum_step_s={}, reach={:?} max_step={}: lower maximum_step_s to at most every reach max_step",
                config.maximum_step_s, reach.id, definition.config.max_step
            )));
        }
        if config.minimum_step_s < definition.config.min_step {
            return Err(SurfaceChannelError::input(format!(
                "config minimum_step_s={}, reach={:?} min_step={}: raise minimum_step_s to at least every reach min_step",
                config.minimum_step_s, reach.id, definition.config.min_step
            )));
        }
    }
    Ok(index)
}

/// Ordered cells of a linked reach. Rejects surveyed geometry.
fn linked_cells(
    setup: &SurfaceChannelSetup,
    reach: usize,
) -> Result<Vec<(&str, f64)>, SurfaceChannelError> {
    let reach = &setup.reaches[reach];
    prismatic_cells(&reach.definition).ok_or_else(|| {
        SurfaceChannelError::input(format!(
            "reach={:?} has surveyed geometry: surface links support rectangular and triangular reaches only; use a rectangular or triangular reach",
            reach.id
        ))
    })
}

/// One link after its checks, with resolved reach, channel cell identity, and weir law.
struct Resolved<'a> {
    setup_index: usize,
    link: &'a SurfaceChannelLink,
    reach_index: usize,
    cell_id: &'a str,
    weir: SuppressedRectangularWeirParameters,
}

/// Checks every link. Returns the resolved links in ascending link-identity order.
fn check_links<'a>(
    setup: &'a SurfaceChannelSetup,
    reach_index: &BTreeMap<&str, usize>,
) -> Result<Vec<Resolved<'a>>, SurfaceChannelError> {
    if setup.links.is_empty() {
        return Err(SurfaceChannelError::input(
            "links is empty: supply at least one surface-channel link",
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
        let cells = linked_cells(setup, reach)?;
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
fn weir(
    link: &SurfaceChannelLink,
) -> Result<SuppressedRectangularWeirParameters, SurfaceChannelError> {
    SuppressedRectangularWeirParameters::new(
        link.crest_length_m,
        link.crest_elevation_m,
        link.weir_coefficient,
        link.applicability.clone(),
    )
    .map_err(|error| SurfaceChannelError::input(format!("link={:?} weir: {error}", link.id)))
}
