use super::{fixture::*, reference_case};
use hydr::structures::*;

#[test]
fn original_definitions_require_finite_ordered_knots_and_explicit_policies() -> Result {
    for knots in [
        vec![],
        vec![(0.0, 0.0)],
        vec![(0.0, 0.0), (0.0, 1.0)],
        vec![(1.0, 0.0), (0.0, 1.0)],
        vec![(-1.0, 0.0), (1.0, 1.0)],
        vec![(0.0, -1.0), (1.0, 1.0)],
        vec![(0.0, 0.0), (f64::INFINITY, 1.0)],
        vec![(0.0, 0.0), (1.0, f64::NAN)],
    ] {
        expect_error(prescribed(&knots, permissive()), "Definition", None)?;
    }
    let knots = vec![
        GateOpeningKnot {
            time_s: 0.0,
            opening_m: 0.0,
        },
        GateOpeningKnot {
            time_s: 1.0,
            opening_m: 1.0,
        },
    ];
    for value in [-1.0, f64::INFINITY, f64::NAN] {
        for field in 0..4 {
            let mut limits = GateMotionLimits {
                maximum_opening_speed_m_per_s: 1.0,
                maximum_closing_speed_m_per_s: 1.0,
            };
            let mut accuracy = permissive();
            match field {
                0 => limits.maximum_opening_speed_m_per_s = value,
                1 => limits.maximum_closing_speed_m_per_s = value,
                2 => accuracy.maximum_absolute_opening_error_m = value,
                _ => accuracy.maximum_relative_opening_error = value,
            }
            expect_error(
                GateOpeningTrajectory::new(
                    knots.clone(),
                    limits,
                    accuracy,
                    "Analytical provenance".into(),
                ),
                "Definition",
                None,
            )?;
        }
    }
    for provenance in ["", "Invalid\nprovenance"] {
        expect_error(
            GateOpeningTrajectory::new(
                knots.clone(),
                GateMotionLimits {
                    maximum_opening_speed_m_per_s: 1.0,
                    maximum_closing_speed_m_per_s: 1.0,
                },
                permissive(),
                provenance.into(),
            ),
            "Definition",
            None,
        )?;
    }
    Ok(())
}

#[test]
fn binding_requires_the_complete_ordered_gate_subsequence() -> Result {
    let batch = mixed()?;
    let trajectory = stationary(0.5, 0.0, 2.0)?;
    let correct = mixed_assignment(trajectory.clone(), trajectory.clone());
    for assignments in [
        vec![],
        vec![correct[0].clone()],
        vec![correct[0].clone(), correct[0].clone()],
        vec![correct[1].clone(), correct[0].clone()],
        vec![
            GateTrajectoryAssignment {
                structure_id: "passive".into(),
                trajectory: trajectory.clone(),
            },
            correct[1].clone(),
        ],
        vec![
            GateTrajectoryAssignment {
                structure_id: "foreign".into(),
                trajectory: trajectory.clone(),
            },
            correct[1].clone(),
        ],
    ] {
        expect_kind(
            batch.with_gate_trajectories(assignments),
            StructureInputErrorKind::Identity,
        )?;
    }
    let bound = batch.with_gate_trajectories(correct.clone())?;
    assert_eq!(bound.assignments(), correct);
    assert_eq!(bound.elements(), batch.elements());
    let passive_only = StructureBatch::new(vec![passive("passive")?])?;
    expect_error(
        passive_only.with_gate_trajectories(vec![]),
        "Definition",
        None,
    )?;
    Ok(())
}

#[test]
fn binding_uses_complete_gate_geometry_and_requires_common_duration() -> Result {
    let definition = stationary(1.5, 0.0, 2.0)?;
    let small = StructureBatch::new(vec![reference_case::gate("same-id", 1.0)?])?;
    let large = StructureBatch::new(vec![reference_case::gate("same-id", 2.0)?])?;
    let assignment = GateTrajectoryAssignment {
        structure_id: "same-id".into(),
        trajectory: definition,
    };
    expect_error(
        small.with_gate_trajectories(vec![assignment.clone()]),
        "Definition",
        Some(0),
    )?;
    assert_eq!(
        large
            .with_gate_trajectories(vec![assignment])?
            .evaluate(0.5)?
            .outputs()?[0]
            .opening_m,
        1.5
    );
    let error = rejection(mixed()?.with_gate_trajectories(mixed_assignment(
        stationary(0.0, 0.0, 1.0)?,
        stationary(0.0, 1.0, 2.0)?,
    )))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::Definition)
    );
    Ok(())
}
