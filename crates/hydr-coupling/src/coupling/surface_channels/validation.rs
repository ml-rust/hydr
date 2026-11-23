//! Construction checks that turn a setup into a checked surface/channel definition.

use super::bank_links::check_links;
use super::bindings::{
    LinkBindings, STRUCTURE_EXCHANGE_BINDING_PREFIX, binding_ids, channel_cells,
    coordinated_definition,
};
use super::operating::check_manual;
use super::pool_ends::{CheckedPoolEnd, check_pool_ends};
use super::pools::{CheckedPools, SurfaceChannelPools, check_attached, check_pools};
use super::ports::{CheckedPort, check_ports};
use super::runoff::{
    CheckedRunoff, RUNOFF_ATTACHMENT_BINDING_PREFIX, append_lateral_bindings,
    check_caller_pool_inflows, check_runoff,
};
use super::structures::{CheckedStructure, check_structures};
use super::{SurfaceChannelError, SurfaceChannelLink, SurfaceChannelSetup};
use crate::channel::ChannelReachDefinition;
use crate::solver::ForcingSchedule;
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
    /// Admitted ports in ascending port-identity order.
    pub(super) ports: Vec<CheckedPort>,
    /// Admitted structure links in ascending identity order.
    pub(super) structures: Vec<CheckedStructure>,
    /// Admitted pools in ascending identity order, or `None` without pools.
    pub(super) pools: Option<CheckedPools>,
    /// Admitted pool ends in ascending identity order.
    pub(super) pool_ends: Vec<CheckedPoolEnd>,
    /// Admitted runoff with its compiled plan, or `None` without runoff.
    pub(super) runoff: Option<CheckedRunoff>,
}

impl SurfaceChannelDefinition {
    /// Checks `setup` and synthesizes two coordinator lateral bindings per bank link and per
    /// structure link with a channel endpoint.
    ///
    /// Each linked reach receives `surface-exchange:<link id>:inflow` and
    /// `surface-exchange:<link id>:withdrawal` at a bank link's channel cell, and
    /// `structure-exchange:<id>:inflow` and `structure-exchange:<id>:withdrawal` at a structure
    /// link's channel cell. They follow the caller bindings, in ascending link identity, bank and
    /// structure links together. Ports and pool ends synthesize no bindings.
    ///
    /// Operating children build and admit at batch construction, since they need a runtime.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Surface`] when the surface config fails its own check.
    /// Returns [`SurfaceChannelError::Input`] for any rule below:
    /// - config tolerances, durations, or work limits fail [`super::SurfaceChannelConfig::check`]
    /// - evidence or identity text is empty or holds control characters
    /// - no reaches, no links, ports, structures, pool ends, or native runoff, or a repeated
    ///   reach or link identity
    /// - a repeated `(surface_cell, reach_id, channel_cell)` triple
    /// - a surface cell outside the grid or inactive
    /// - an unknown reach identity or a channel cell out of range
    /// - a nonfinite or nonpositive crest length or weir coefficient, or a nonfinite crest
    /// - incomplete weir applicability evidence, or a coefficient without symmetric reversal
    /// - a reach tangent that is nonfinite or not unit length
    /// - a crest below the surface bed or the channel bed; a surveyed channel bed is the
    ///   minimum elevation of its section
    /// - gravity or vertical datum differing between the surface and a reach
    /// - a caller lateral binding using the `surface-exchange:` prefix, the
    ///   `structure-exchange:` prefix, or the `runoff-attachment:` prefix, or a caller pool
    ///   inflow binding using the `runoff-attachment:` prefix (`reserved-binding-prefix`)
    /// - `maximum_step_s` above the surface or a reach `max_step`
    /// - `minimum_step_s` below the surface or a reach `min_step`
    /// - `retry_limit` above the surface `max_retries`
    ///
    /// Each port also rejects for:
    /// - an empty or repeated identity, one equal to a link identity, or empty evidence
    ///   (`duplicate-links`, `missing-link-evidence`)
    /// - a triangular or empty reach, or a surveyed reach whose section is not a rectangle
    ///   (`port-section-mismatch`)
    /// - face widths whose sum differs from the end cell width, or from the width of a surveyed
    ///   rectangle, by more than
    ///   [`super::SURFACE_CHANNEL_PORT_WIDTH_TOLERANCE`] times that width (`port-width-mismatch`)
    /// - an unknown reach, a second port on one reach end, a face with no active cell or two,
    ///   a caller open face, a face in two ports, faces off one axis, face line, or surface
    ///   side, or a gap in the face line (`port-face-ownership`)
    ///
    /// Each structure also rejects for:
    /// - an empty or repeated identity, one equal to a link, port, or pool identity, or a
    ///   repeated scenario identity (`duplicate-links`)
    /// - empty scenario or connection evidence (`missing-link-evidence`)
    /// - an endpoint on no active surface cell, no existing channel cell, or no pool, both
    ///   endpoints on one node, or an element endpoint that differs from its node identity
    ///   (`structure-endpoint-identity`)
    /// - an element datum other than the surface datum, or a gate or orifice gravity other than
    ///   the surface gravity (`datum-or-gravity-mismatch`)
    /// - an attachment bed below its node bed or pool origin, or a weir crest, gate sill, or
    ///   orifice sill below either attachment bed (`unobstructed-pool-connection`)
    /// - a bidirectional pump, or a bidirectional law without symmetric reversal evidence
    ///   (`non-symmetric-structure`)
    /// - an operation that does not drive its law, or a dynamic owner without `operating`
    ///   (`operation-ownership`)
    ///
    /// Pools reject for:
    /// - an empty or repeated identity, or one equal to a link or port identity
    ///   (`duplicate-links`, `missing-link-evidence`)
    /// - a curve outside the surface datum (`datum-or-gravity-mismatch`)
    /// - a repeated storage support, or support, source policy, or origin that differs from the
    ///   pools, support, sources, or inflows without pools, or a pool that joins no structure and
    ///   no pool end (`pool-support`)
    /// - a `storage` config that fails its own check
    ///
    /// Each pool end rejects for:
    /// - an empty or repeated identity, one equal to a link, port, structure, or pool identity,
    ///   or empty evidence (`duplicate-links`, `missing-link-evidence`)
    /// - an unknown reach or pool, or a reach without cells (`structure-endpoint-identity`)
    /// - an end cell bed below the pool origin (`unobstructed-pool-connection`)
    /// - a reach end with a port or a second pool end (`reach-end-ownership`)
    /// - a nonfinite or negative entrance loss (`pool-end-regime`)
    ///
    /// Runoff rejects for:
    /// - a support plan that fails against the surface domain and open faces (`runoff-plan`)
    /// - nonzero surface rain on a suppressed cell in any interval (`suppressed-rain`)
    /// - pool rain support of an unknown pool (`pool-support`)
    /// - an outlet that names no active modelled outlet, or an active outlet without a
    ///   destination (`unmapped-runoff-outlet`), or an outlet bound twice
    ///   (`duplicate-runoff-outlet`)
    /// - an empty outlet identity or evidence, or a nonfinite invert (`runoff-free-outlet`)
    /// - an inactive, off-grid, or repeated surface cell, an unknown reach, an out-of-range or
    ///   repeated reach cell, an unknown pool, an empty target list, or a weight that is
    ///   nonfinite or not positive (`runoff-destination`)
    /// - a caller pool inflow binding whose physical support names an active modelled source
    ///   or one of its atoms (`runoff-duplicate-writer`)
    pub fn new(setup: SurfaceChannelSetup) -> Result<Self, SurfaceChannelError> {
        let pools = SurfaceChannelPools::of(&setup);
        let structures = setup.structures.clone();
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
        let resolved = check_links(
            &setup,
            &reach_index,
            structures.is_empty() && pools.ends.is_empty() && setup.runoff.is_none(),
        )?;
        let ports = check_ports(&setup, &reach_index)?;
        let taken: BTreeSet<&str> = setup
            .links
            .iter()
            .map(|link| link.id.as_str())
            .chain(setup.ports.iter().map(|port| port.id.as_str()))
            .collect();
        let datum = setup.surface.domain.reference().vertical_datum();
        let pools = check_pools(datum, &taken, pools)?;
        let mut taken = taken;
        if let Some(pools) = &pools {
            taken.extend(pools.ids.iter().map(String::as_str));
        }
        let structures =
            check_structures(&setup, &reach_index, pools.as_ref(), &taken, structures)?;
        let structure_beds: Vec<[f64; 2]> = setup
            .structures
            .iter()
            .map(|s| [s.endpoint_a.bed_elevation_m, s.endpoint_b.bed_elevation_m])
            .collect();
        super::geometry::check_geometry_schedules(
            &setup.structures,
            &setup.geometry_schedules,
            &structure_beds,
            setup.surface.config.gravity,
        )?;
        taken.extend(structures.iter().map(CheckedStructure::id));
        let pool_ends = match &pools {
            Some(checked) => check_pool_ends(
                &setup,
                &reach_index,
                Some(checked),
                &ports,
                &taken,
                &checked.declared.ends,
            )?,
            None => Vec::new(),
        };
        if let Some(pools) = &pools {
            check_attached(pools, &structures, &pool_ends)?;
        }
        super::geometry::check_pool_curve_schedules(
            pools.as_ref(),
            &structures,
            &pool_ends,
            &setup.pool_curve_schedules,
        )?;
        super::geometry::check_surface_bed_schedules(&setup)?;
        let runoff = check_runoff(&setup, &reach_index, pools.as_ref())?;
        check_caller_pool_inflows(&setup.pool_inflows, runoff.as_ref())?;
        let surface_cells: Vec<usize> = setup
            .links
            .iter()
            .map(|link| link.surface_cell)
            .chain(structures.iter().filter_map(CheckedStructure::surface_cell))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut links = Vec::with_capacity(resolved.len());
        let mut per_reach: Vec<BTreeMap<&str, LinkBindings<'_>>> =
            (0..setup.reaches.len()).map(|_| BTreeMap::new()).collect();
        for resolved in &resolved {
            let link = resolved.link;
            let (inflow_binding, withdrawal_binding) = binding_ids(&link.id);
            links.push(CheckedLink {
                setup_index: resolved.setup_index,
                reach_index: resolved.reach_index,
                surface_lane: surface_cells.partition_point(|&cell| cell < link.surface_cell),
                channel_cell: link.channel_cell,
                weir: resolved.weir.clone(),
                inflow_binding,
                withdrawal_binding,
            });
        }
        for (link, resolved) in links.iter().zip(&resolved) {
            if let Some(reach_links) = per_reach.get_mut(link.reach_index) {
                reach_links.insert(
                    resolved.link.id.as_str(),
                    LinkBindings {
                        owner: "surface-channel link",
                        id: resolved.link.id.as_str(),
                        cell_id: resolved.cell_id,
                        inflow: &link.inflow_binding,
                        withdrawal: &link.withdrawal_binding,
                    },
                );
            }
        }
        for structure in &structures {
            if let Some((reach_index, _, cell_id)) = structure.channel()
                && let Some(reach_links) = per_reach.get_mut(reach_index)
            {
                reach_links.insert(
                    structure.id(),
                    LinkBindings {
                        owner: "surface-channel structure",
                        id: structure.id(),
                        cell_id,
                        inflow: &structure.inflow_binding,
                        withdrawal: &structure.withdrawal_binding,
                    },
                );
            }
        }
        let coordinated = setup
            .reaches
            .iter()
            .zip(&per_reach)
            .enumerate()
            .map(|(index, (reach, links))| {
                let links: Vec<LinkBindings<'_>> = links.values().copied().collect();
                let mut definition =
                    coordinated_definition(reach, &links, &setup.storage_exclusion_evidence);
                if let Some(runoff) = &runoff {
                    let cells = channel_cells(&reach.definition);
                    let ids: Vec<&str> = cells.iter().map(|(id, _)| *id).collect();
                    append_lateral_bindings(runoff, index, &ids, &mut definition);
                }
                definition
            })
            .collect();
        let definition = Self {
            setup,
            coordinated,
            links,
            surface_cells,
            ports,
            structures,
            pools,
            pool_ends,
            runoff,
        };
        if definition.setup.operating.is_none() {
            check_manual(&definition)?;
        }
        Ok(definition)
    }

    /// Immutable caller setup.
    pub fn setup(&self) -> &SurfaceChannelSetup {
        &self.setup
    }

    /// This definition with `forcing` as the surface forcing schedule.
    ///
    /// Forcing enters no coordinator check, so the checked links and reaches carry over.
    pub(super) fn with_surface_forcing(&self, forcing: ForcingSchedule) -> Self {
        let mut definition = self.clone();
        definition.setup.surface.forcing = forcing;
        definition
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
pub(super) fn text(name: &str, value: &str) -> Result<(), SurfaceChannelError> {
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
        for (prefix, source) in [
            (
                STRUCTURE_EXCHANGE_BINDING_PREFIX,
                "structure-exchange bindings from structure links",
            ),
            (
                RUNOFF_ATTACHMENT_BINDING_PREFIX,
                "runoff bindings from runoff outlets",
            ),
        ] {
            if let Some(binding) = definition
                .lateral_bindings
                .iter()
                .find(|binding| binding.boundary_id.starts_with(prefix))
            {
                return Err(SurfaceChannelError::input(format!(
                    "reserved-binding-prefix: reach={:?} lateral binding {:?} uses the reserved prefix {prefix:?}: rename the binding; the coordinator synthesizes {source}",
                    reach.id, binding.boundary_id
                )));
            }
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
