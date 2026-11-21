//! Runoff admission: plan compile, outlet ownership, destinations, and delivery lanes.
//!
//! Each rejection names its rule first.

use super::SurfaceChannelRunoff;
use super::bindings::{lateral_binding_id, pool_binding_id};
use super::targets::targets;
use crate::coupling::level_pools::CoupledPoolInflowBinding;
use crate::coupling::surface_channels::pools::CheckedPools;
use crate::coupling::surface_channels::validation::text;
use crate::coupling::surface_channels::{SurfaceChannelError, SurfaceChannelSetup};
use crate::runoff::{ContributionKind, Subarea, SupportError, SupportPlan};
use std::collections::BTreeMap;

/// One receiving node of a delivery lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) enum RunoffTarget {
    /// Global row-major surface cell index.
    Surface { cell: usize },
    /// Setup reach index and ordered cell index inside the reach.
    Channel { reach: usize, cell: usize },
    /// Pool position in ascending pool identity.
    Pool { position: usize },
}

/// One `(lane, target)` pair: one subarea lane of one active outlet into one receiving node.
#[derive(Clone, Debug, PartialEq)]
pub(in super::super) struct DeliveryLane {
    /// Index into the setup's declared outlets.
    pub outlet: usize,
    /// Position of the outlet's definition among the plan's active definitions.
    pub definition: usize,
    /// Potential subarea of that definition.
    pub subarea: Subarea,
    pub target: RunoffTarget,
    /// Normalized weight. Exactly one for a single target.
    pub weight: f64,
    /// Synthesized binding identity of a channel or pool target. `None` on the surface.
    pub binding: Option<String>,
    /// Ownership evidence carried by the synthesized binding.
    pub ownership: String,
}

/// Checked runoff with its compiled plan and delivery lanes.
#[derive(Clone, Debug)]
pub(in super::super) struct CheckedRunoff {
    /// Plan compiled against the surface domain and open faces.
    pub plan: SupportPlan,
    /// Delivery lanes by active definition, then subarea, then declared target.
    pub lanes: Vec<DeliveryLane>,
    /// One synthesized inflow binding per pool delivery lane, in delivery-lane order.
    pub pool_bindings: Vec<CoupledPoolInflowBinding>,
}

/// The plan follows from the setup, so lanes and bindings decide equality.
impl PartialEq for CheckedRunoff {
    fn eq(&self, other: &Self) -> bool {
        self.plan.input() == other.plan.input()
            && self.lanes == other.lanes
            && self.pool_bindings == other.pool_bindings
    }
}

/// Rejects with `entry` for outlet `id`.
pub(super) fn rejected(
    entry: &str,
    id: &str,
    detail: impl std::fmt::Display,
) -> SurfaceChannelError {
    SurfaceChannelError::input(format!("{entry}: runoff outlet={id:?} {detail}"))
}

/// A plan rejection as a coordinator input rejection under `entry`.
pub(super) fn plan_error(entry: &str, error: SupportError) -> SurfaceChannelError {
    match error {
        SupportError::Subcatchment(error) => SurfaceChannelError::Runoff(error),
        other => SurfaceChannelError::input(format!("{entry}: {other}")),
    }
}

/// Checks the runoff of `setup` and builds its delivery lanes. Returns `None` without runoff.
///
/// `reaches` maps reach identity to setup index. `pools` holds the admitted pools.
pub(in super::super) fn check_runoff(
    setup: &SurfaceChannelSetup,
    reaches: &BTreeMap<&str, usize>,
    pools: Option<&CheckedPools>,
) -> Result<Option<CheckedRunoff>, SurfaceChannelError> {
    let Some(runoff) = &setup.runoff else {
        return Ok(None);
    };
    let surface = &setup.surface;
    let plan = SupportPlan::new(
        runoff.support.clone(),
        &surface.domain,
        &surface.boundaries,
        &runoff.definitions,
    )
    .map_err(|error| plan_error("runoff-plan", error))?;
    plan.check_surface_schedule(&surface.forcing)
        .map_err(|error| plan_error("suppressed-rain: setup surface forcing", error))?;
    for support in &runoff.support.pool_rain {
        if pools
            .and_then(|pools| pools.find(&support.pool_id))
            .is_none()
        {
            return Err(SurfaceChannelError::input(format!(
                "pool-support: runoff pool rain pool_id={:?} names no coordinator pool: declare rain support for setup pools only",
                support.pool_id
            )));
        }
    }
    let bound = bind_outlets(runoff, &plan)?;
    let mut lanes = Vec::new();
    let mut pool_bindings = Vec::new();
    for (definition, (outlet_index, contribution_id)) in bound.into_iter().enumerate() {
        let outlet = &runoff.outlets[outlet_index];
        let targets = targets(setup, reaches, pools, outlet)?;
        let ownership = format!(
            "surface-channel runoff {:?}: {}",
            outlet.outlet_id, setup.storage_exclusion_evidence
        );
        let parameters = &plan.active_definitions()[definition].parameters;
        for subarea in Subarea::ALL {
            if parameters.area_m2(subarea) <= 0.0 || parameters.donor_fraction(subarea) >= 1.0 {
                continue;
            }
            for &(target, weight) in &targets {
                let binding = match target {
                    RunoffTarget::Surface { .. } => None,
                    RunoffTarget::Channel { cell, .. } => {
                        Some(lateral_binding_id(&outlet.outlet_id, subarea, cell))
                    }
                    RunoffTarget::Pool { position } => {
                        let boundary_id = pool_binding_id(&outlet.outlet_id, subarea);
                        let pool_id = pools
                            .and_then(|pools| pools.ids.get(position))
                            .cloned()
                            .unwrap_or_default();
                        pool_bindings.push(CoupledPoolInflowBinding {
                            boundary_id: boundary_id.clone(),
                            pool_id,
                            input_id: boundary_id.clone(),
                            physical_support_id: contribution_id.clone(),
                            boundary_assertion: outlet.free_outlet.evidence.clone(),
                            ownership_assertion: ownership.clone(),
                        });
                        Some(boundary_id)
                    }
                };
                lanes.push(DeliveryLane {
                    outlet: outlet_index,
                    definition,
                    subarea,
                    target,
                    weight,
                    binding,
                    ownership: ownership.clone(),
                });
            }
        }
    }
    Ok(Some(CheckedRunoff {
        plan,
        lanes,
        pool_bindings,
    }))
}

/// The declared outlet index and contribution identity of each active definition, in order.
///
/// Rejects outlet text and inverts, unknown or inactive outlets, outlets bound twice, and active
/// outlets left unbound.
fn bind_outlets(
    runoff: &SurfaceChannelRunoff,
    plan: &SupportPlan,
) -> Result<Vec<(usize, String)>, SurfaceChannelError> {
    let mut active = BTreeMap::new();
    for (position, contribution) in plan.active_contributions().enumerate() {
        if let ContributionKind::ModelledRunoff { outlet_id, .. } = &contribution.kind {
            active.insert(outlet_id.as_str(), (position, contribution.id.as_str()));
        }
    }
    let mut bound: Vec<Option<(usize, String)>> = vec![None; active.len()];
    for (index, outlet) in runoff.outlets.iter().enumerate() {
        let id = outlet.outlet_id.as_str();
        text("runoff-free-outlet: runoff outlet_id", id)?;
        text(
            &format!("runoff-free-outlet: runoff outlet={id:?} free_outlet evidence"),
            &outlet.free_outlet.evidence,
        )?;
        let invert = outlet.free_outlet.invert_elevation_m;
        if !invert.is_finite() {
            return Err(rejected(
                "runoff-free-outlet",
                id,
                format!("invert_elevation_m={invert}: supply a finite invert in the surface datum"),
            ));
        }
        let Some(&(position, contribution)) = active.get(id) else {
            return Err(rejected(
                "unmapped-runoff-outlet",
                id,
                "names no active modelled outlet of the plan: bind active outlets only",
            ));
        };
        match bound.get_mut(position) {
            Some(slot @ None) => *slot = Some((index, contribution.to_owned())),
            _ => {
                return Err(rejected(
                    "duplicate-runoff-outlet",
                    id,
                    "is bound twice: declare one destination per outlet",
                ));
            }
        }
    }
    for (outlet_id, (position, _)) in &active {
        if bound.get(*position).is_some_and(Option::is_none) {
            return Err(rejected(
                "unmapped-runoff-outlet",
                outlet_id,
                "has no destination: declare one destination for every active outlet",
            ));
        }
    }
    Ok(bound.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::super::scenario::{
        OUTLET, POND, SOURCE, build, outlet, parts, reach_outlet, rejection, starts,
    };
    use super::super::{SurfaceChannelRunoffDestination, SurfaceChannelRunoffOutlet};
    use crate::coupling::surface_channels::SurfaceChannelSetup;
    use crate::coupling::surface_channels::structures::scenario::{REACH, TestResult};

    /// The default case with its outlets replaced by `outlets`.
    fn with_outlets(outlets: Vec<SurfaceChannelRunoffOutlet>) -> TestResult<SurfaceChannelSetup> {
        let (mut setup, _) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.outlets = outlets;
        }
        Ok(setup)
    }

    #[test]
    fn an_unknown_outlet_or_an_unbound_active_outlet_is_unmapped() -> TestResult {
        let mut unknown = reach_outlet();
        unknown.outlet_id = "elsewhere".into();
        let message = rejection(with_outlets(vec![unknown])?)?;
        starts(&message, "unmapped-runoff-outlet");
        assert!(message.contains("\"elsewhere\""), "{message}");
        let message = rejection(with_outlets(Vec::new())?)?;
        starts(&message, "unmapped-runoff-outlet");
        assert!(message.contains(OUTLET), "{message}");
        Ok(())
    }

    #[test]
    fn an_outlet_bound_twice_is_a_duplicate() -> TestResult {
        let message = rejection(with_outlets(vec![reach_outlet(), reach_outlet()])?)?;
        starts(&message, "duplicate-runoff-outlet");
        assert!(message.contains(OUTLET), "{message}");
        Ok(())
    }

    #[test]
    fn free_outlet_text_and_invert_are_checked() -> TestResult {
        let mut evidence = reach_outlet();
        evidence.free_outlet.evidence = " ".into();
        starts(
            &rejection(with_outlets(vec![evidence])?)?,
            "runoff-free-outlet",
        );
        let mut invert = reach_outlet();
        invert.free_outlet.invert_elevation_m = f64::NAN;
        starts(
            &rejection(with_outlets(vec![invert])?)?,
            "runoff-free-outlet",
        );
        let mut empty = reach_outlet();
        empty.outlet_id = String::new();
        starts(
            &rejection(with_outlets(vec![empty])?)?,
            "runoff-free-outlet",
        );
        Ok(())
    }

    #[test]
    fn a_failing_plan_and_unknown_pool_rain_reject() -> TestResult {
        let (mut setup, _) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.support.partition.complete_disjoint = false;
        }
        starts(&rejection(setup)?, "runoff-plan");
        let (mut setup, _) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.support.pool_rain[0].pool_id = "lake".into();
        }
        starts(&rejection(setup)?, "pool-support");
        Ok(())
    }

    #[test]
    fn construction_synthesizes_one_lateral_binding_per_channel_lane() -> TestResult {
        let batch = build(parts()?)?;
        let definition = batch.definition();
        let Some(reach) = definition.coordinated_reach(REACH) else {
            return Err("missing coordinated reach".into());
        };
        let id = format!("runoff-attachment:{OUTLET}:ImperviousZeroStorage:0");
        let synthesized: Vec<_> = reach
            .lateral_bindings
            .iter()
            .filter(|binding| binding.boundary_id.starts_with("runoff-attachment:"))
            .collect();
        assert_eq!(synthesized.len(), 1);
        assert_eq!(synthesized[0].boundary_id, id);
        assert_eq!(synthesized[0].cell_id, "cell-0");
        let Some(runoff) = &definition.runoff else {
            return Err("missing checked runoff".into());
        };
        assert_eq!(runoff.lanes.len(), 1);
        assert_eq!(runoff.lanes[0].weight.to_bits(), 1.0_f64.to_bits());
        assert!(runoff.pool_bindings.is_empty());
        Ok(())
    }

    #[test]
    fn a_pool_destination_synthesizes_one_pool_binding_per_lane() -> TestResult {
        let (mut setup, initial) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.outlets = vec![outlet(SurfaceChannelRunoffDestination::Pool {
                pool_id: POND.into(),
            })];
        }
        let batch = build((setup, initial))?;
        let Some(runoff) = &batch.definition().runoff else {
            return Err("missing checked runoff".into());
        };
        assert_eq!(runoff.pool_bindings.len(), 1);
        let binding = &runoff.pool_bindings[0];
        assert_eq!(
            binding.boundary_id,
            format!("runoff-attachment:{OUTLET}:ImperviousZeroStorage")
        );
        assert_eq!(binding.pool_id, POND);
        assert_eq!(binding.physical_support_id, SOURCE);
        let Some(reach) = batch.definition().coordinated_reach(REACH) else {
            return Err("missing coordinated reach".into());
        };
        assert!(
            !reach
                .lateral_bindings
                .iter()
                .any(|binding| binding.boundary_id.starts_with("runoff-attachment:"))
        );
        Ok(())
    }
}
