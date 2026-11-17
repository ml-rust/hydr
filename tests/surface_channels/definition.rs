use super::fixture::{self, *};
use hydr::channel::*;
use hydr::coupling::*;

#[test]
fn valid_setup_builds_every_child() -> Result {
    let (setup, initial) = case()?;
    let spare = setup.reaches[1].definition.clone();
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    assert_eq!(batch.time().to_bits(), 0.0_f64.to_bits());
    assert_eq!(batch.definition().surface_cells(), [0, 1, 2]);
    let ids: Vec<&str> = batch
        .definition()
        .links()
        .map(|link| link.id.as_str())
        .collect();
    assert_eq!(ids, ["link-a", "link-b", "link-c"]);
    let reach = batch.reach(REACH).ok_or("linked reach is missing")?;
    assert_eq!(reach.definition().lateral_bindings.len(), 7);
    let unlinked = batch.reach(SPARE).ok_or("unlinked reach is missing")?;
    assert_eq!(unlinked.definition(), &spare);
    assert!(batch.reach("absent").is_none());
    assert_eq!(batch.surface().time().to_bits(), 0.0_f64.to_bits());
    Ok(())
}

#[test]
fn synthesized_bindings_follow_link_identity_order() -> Result {
    let (setup, _) = case()?;
    let mut reversed = setup.clone();
    reversed.links.reverse();
    let forward = SurfaceChannelDefinition::new(setup)?;
    let backward = SurfaceChannelDefinition::new(reversed)?;
    let definition = forward
        .coordinated_reach(REACH)
        .ok_or("linked reach is missing")?;
    assert_eq!(Some(definition), backward.coordinated_reach(REACH));
    let found: Vec<(&str, &str, ChannelLateralDirection)> = definition
        .lateral_bindings
        .iter()
        .map(|binding| {
            (
                binding.boundary_id.as_str(),
                binding.cell_id.as_str(),
                binding.direction,
            )
        })
        .collect();
    use ChannelLateralDirection::{Inflow, Withdrawal};
    assert_eq!(
        found,
        [
            (CALLER_BINDING, "cell-3", Inflow),
            ("surface-exchange:link-a:inflow", "cell-0", Inflow),
            ("surface-exchange:link-a:withdrawal", "cell-0", Withdrawal),
            ("surface-exchange:link-b:inflow", "cell-1", Inflow),
            ("surface-exchange:link-b:withdrawal", "cell-1", Withdrawal),
            ("surface-exchange:link-c:inflow", "cell-2", Inflow),
            ("surface-exchange:link-c:withdrawal", "cell-2", Withdrawal),
        ]
    );
    assert_eq!(
        forward.coordinated_reach(SPARE),
        Some(&reach_definition(2, CHANNEL_BED))
    );
    Ok(())
}

/// Returns the rejection message of the fixture setup after `edit`.
fn reject(edit: impl FnOnce(&mut SurfaceChannelSetup)) -> Result<String> {
    let (mut setup, _) = case()?;
    edit(&mut setup);
    input_rejection(setup)
}

#[test]
fn rejects_repeated_link_identity() -> Result {
    let message = reject(|s| s.links[1].id = "link-b".into())?;
    contains(&message, &["link id=\"link-b\" repeats", "unique identity"])
}

#[test]
fn rejects_repeated_link_triple() -> Result {
    let message = reject(|s| s.links.push(fixture::link("link-d", 1, 1)))?;
    contains(
        &message,
        &["link=\"link-d\" repeats surface_cell=1", "merge"],
    )
}

#[test]
fn rejects_surface_cell_outside_grid() -> Result {
    let message = reject(|s| s.links[0].surface_cell = 8)?;
    contains(
        &message,
        &["surface_cell=8, cells=8", "active cell inside the grid"],
    )
}

#[test]
fn rejects_inactive_surface_cell() -> Result {
    let message = reject(|s| s.links[0].surface_cell = INACTIVE_CELL)?;
    contains(&message, &["surface_cell=7", "active cell inside the grid"])
}

#[test]
fn rejects_unknown_reach() -> Result {
    let message = reject(|s| s.links[0].reach_id = "absent".into())?;
    contains(&message, &["reach_id=\"absent\" names no setup reach"])
}

#[test]
fn rejects_channel_cell_out_of_range() -> Result {
    let message = reject(|s| s.links[0].channel_cell = 4)?;
    contains(
        &message,
        &["channel_cell=4", "cells=4", "existing reach cell"],
    )
}

#[test]
fn rejects_nonpositive_crest_length() -> Result {
    let message = reject(|s| s.links[0].crest_length_m = 0.0)?;
    contains(&message, &["crest_length_m=0", "finite positive"])
}

#[test]
fn rejects_nonfinite_weir_coefficient() -> Result {
    let message = reject(|s| s.links[0].weir_coefficient = f64::NAN)?;
    contains(&message, &["weir_coefficient=NaN", "finite positive"])
}

#[test]
fn rejects_nonfinite_crest_elevation() -> Result {
    let message = reject(|s| s.links[0].crest_elevation_m = f64::INFINITY)?;
    contains(&message, &["crest_elevation_m=inf", "finite crest"])
}

#[test]
fn rejects_incomplete_weir_applicability() -> Result {
    let message = reject(|s| s.links[0].applicability.coefficient_provenance = " ".into())?;
    contains(
        &message,
        &["link=\"link-b\" weir", "coefficient_provenance"],
    )
}

#[test]
fn rejects_invalid_tolerance() -> Result {
    let message = reject(|s| {
        s.config.head_absolute_m = 0.0;
        s.config.head_relative = 0.0;
    })?;
    contains(
        &message,
        &["head_absolute_m=0", "positive combined allowance"],
    )
}

#[test]
fn rejects_invalid_step_limits() -> Result {
    let message = reject(|s| s.config.minimum_step_s = 2.0)?;
    contains(
        &message,
        &["minimum_step_s=2", "maximum_step_s >= minimum_step_s"],
    )
}

#[test]
fn rejects_zero_iteration_limit() -> Result {
    let message = reject(|s| s.config.iteration_limit = 0)?;
    contains(&message, &["iteration_limit=0"])
}

#[test]
fn rejects_non_unit_reach_tangent() -> Result {
    let message = reject(|s| s.links[0].reach_tangent = [1.0, 1.0])?;
    contains(&message, &["reach_tangent=[1,1]", "unit tangent"])?;
    let message = reject(|s| s.links[0].reach_tangent = [f64::NAN, 1.0])?;
    contains(&message, &["reach_tangent=[NaN,1]", "unit tangent"])
}

#[test]
fn rejects_crest_below_surface_bed() -> Result {
    let message = reject(|s| s.links[0].crest_elevation_m = SURFACE_BED - 0.1)?;
    contains(
        &message,
        &["is below surface bed=1", "unobstructed connection"],
    )
}

#[test]
fn rejects_crest_below_channel_bed() -> Result {
    let message = reject(|s| s.reaches[0].definition = reach_definition(4, 1.5))?;
    contains(&message, &["channel bed=1.5", "unobstructed connection"])
}

#[test]
fn rejects_gravity_mismatch() -> Result {
    let message = reject(|s| s.reaches[0].definition.config.gravity = 9.8)?;
    contains(
        &message,
        &["gravity=9.8, surface gravity=9.81", "bit-identical"],
    )
}

#[test]
fn rejects_vertical_datum_mismatch() -> Result {
    let message = reject(|s| s.reaches[1].definition.vertical_datum = "Other datum".into())?;
    contains(
        &message,
        &[
            "reach=\"spare\" vertical_datum=\"Other datum\"",
            "one vertical datum",
        ],
    )
}

#[test]
fn rejects_step_above_surface_max_step() -> Result {
    let message = reject(|s| {
        s.config.maximum_step_s = 2.0;
        s.reaches[0].definition.config.max_step = 2.0;
        s.reaches[1].definition.config.max_step = 2.0;
    })?;
    contains(&message, &["maximum_step_s=2, surface max_step=1"])
}

#[test]
fn rejects_step_above_reach_max_step() -> Result {
    let message = reject(|s| s.reaches[1].definition.config.max_step = 0.5)?;
    contains(
        &message,
        &["reach=\"spare\" max_step=0.5", "every reach max_step"],
    )
}

#[test]
fn rejects_step_below_surface_min_step() -> Result {
    let message = reject(|s| s.surface.config.min_step = 1e-6)?;
    contains(
        &message,
        &["surface min_step=0.000001", "raise minimum_step_s"],
    )
}

#[test]
fn rejects_step_below_reach_min_step() -> Result {
    let message = reject(|s| s.reaches[0].definition.config.min_step = 1e-6)?;
    contains(
        &message,
        &["reach=\"reach\" min_step=0.000001", "every reach min_step"],
    )
}

#[test]
fn rejects_retry_limit_above_surface() -> Result {
    let message = reject(|s| s.config.retry_limit = 65)?;
    contains(&message, &["retry_limit=65, surface max_retries=64"])
}

#[test]
fn rejects_reserved_caller_binding() -> Result {
    let message = reject(|s| {
        s.reaches[0].definition.lateral_bindings[0].boundary_id = "surface-exchange:x".into();
    })?;
    contains(
        &message,
        &["\"surface-exchange:x\" uses the reserved prefix", "rename"],
    )
}

#[test]
fn rejects_linked_surveyed_reach() -> Result {
    let message = reject(|s| {
        s.reaches[0].definition.geometry = ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "surveyed-rectangle".into(),
                vertical_datum: DATUM.into(),
                points: [(0.0, 4.0), (0.0, 0.0), (1.0, 0.0), (1.0, 4.0)]
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Explicit analytical rectangular survey".into(),
            },
            cells: (0..4)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 2.0,
                    manning_n: 0.0,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        };
    })?;
    contains(
        &message,
        &[
            "reach=\"reach\" has surveyed geometry",
            "rectangular and triangular",
        ],
    )
}

#[test]
fn rejects_empty_evidence_and_collections() -> Result {
    let message = reject(|s| s.storage_exclusion_evidence = String::new())?;
    contains(&message, &["storage_exclusion_evidence=\"\""])?;
    let message = reject(|s| s.links[2].mapping_evidence = "\n".into())?;
    contains(&message, &["mapping_evidence=\"\\n\""])?;
    let message = reject(|s| s.links.clear())?;
    contains(&message, &["links is empty"])?;
    let message = reject(|s| s.reaches[1].id = REACH.into())?;
    contains(&message, &["reach id=\"reach\" repeats"])
}

#[test]
fn rejects_inconsistent_initial_states() -> Result {
    let (setup, mut initial) = case()?;
    initial.reaches.pop();
    let message = batch_input_rejection(setup, initial)?;
    contains(&message, &["reach id=\"reach\" has no state"])?;
    let (setup, mut initial) = case()?;
    initial.reaches[1].1.time_s = 1.0;
    let message = batch_input_rejection(setup, initial)?;
    contains(&message, &["reach=\"reach\" time_s=1", "common time"])?;
    let (setup, mut initial) = case()?;
    initial
        .reaches
        .push((SPARE.into(), initial.reaches[0].1.clone()));
    let message = batch_input_rejection(setup, initial)?;
    contains(&message, &["reach id=\"spare\" repeats"])?;
    let (setup, mut initial) = case()?;
    initial.surface.time = 1.0;
    let message = batch_input_rejection(setup, initial)?;
    contains(&message, &["surface time=1, time_s=0"])
}

#[test]
fn retryable_errors_follow_the_shared_taxonomy() {
    assert!(SurfaceChannelError::Numerical("x".into()).is_retryable());
    assert!(!SurfaceChannelError::Input("x".into()).is_retryable());
    assert!(
        SurfaceChannelError::Surface(hydr::solver::EngineError::LandSource {
            cell: None,
            reason: "x".into()
        })
        .is_retryable()
    );
    assert!(
        !SurfaceChannelError::Surface(hydr::solver::EngineError::Rejected {
            operation: "x",
            reason: "x".into()
        })
        .is_retryable()
    );
    assert!(
        !SurfaceChannelError::Unresolved {
            dt_s: 1.0,
            retries: 1,
            reason: "x".into()
        }
        .is_retryable()
    );
}
