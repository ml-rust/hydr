use super::fixture::*;
use hydr::structures::*;

#[test]
fn prescribed_subset_leaves_positive_manual_gate_complement_explicit_and_ordered() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate("manual-a")?, gate("path")?, gate("manual-b")?],
    )?;
    let prescribed = trajectory(&context, "path", &[(0.0, 0.0), (1.0, 0.0), (2.0, 0.5)])?;
    let operations = [opening("manual-a", 0.25), opening("manual-b", 0.75)];
    let heads = heads(&context);
    let direct = prescribed.propose_at(0.5, &heads, &operations)?;
    assert_eq!(
        direct.outputs()?[0].operation,
        operations[0].operating_point
    );
    assert_eq!(
        direct.outputs()?[2].operation,
        operations[1].operating_point
    );
    assert_eq!(direct.outputs()?[1].regime, StructureRegime::GateClosed);
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &prescribed,
        },
        OperatingSourceBinding::Manual {
            source_id: "manual".into(),
            structure_ids: vec!["manual-a".into(), "manual-b".into()],
        },
    ])?;
    let mixed = registry.propose_at(0.5, &heads, &operations, &[])?;
    assert_eq!(
        mixed.operations()?,
        vec![
            operations[0].clone(),
            opening("path", 0.0),
            operations[1].clone()
        ]
    );
    explicit_match(&context, &heads, &mixed)?;
    kind(
        prescribed.propose_at(0.5, &heads, &[opening("path", 0.5), operations[1].clone()]),
        StructureInputErrorKind::Identity,
    )?;
    kind(
        context.with_gate_trajectories(prescribed.assignments().to_vec()),
        StructureInputErrorKind::Identity,
    )?;
    Ok(())
}

#[test]
fn source_accuracy_rejection_keeps_original_element_category_and_numerical_context() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate("manual")?, gate("path")?],
    )?;
    let path = GateOpeningTrajectory::new(
        vec![
            GateOpeningKnot {
                time_s: 0.1,
                opening_m: 0.2,
            },
            GateOpeningKnot {
                time_s: 0.9,
                opening_m: 0.8,
            },
        ],
        GateMotionLimits {
            maximum_opening_speed_m_per_s: f64::MAX,
            maximum_closing_speed_m_per_s: f64::MAX,
        },
        GateTrajectoryAccuracy {
            maximum_absolute_opening_error_m: 0.0,
            maximum_relative_opening_error: 0.0,
        },
        "Explicit zero numerical allowance".into(),
    )?;
    let prescribed = context.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
        structure_id: "path".into(),
        trajectory: path,
    }])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Manual {
            source_id: "manual".into(),
            structure_ids: vec!["manual".into()],
        },
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &prescribed,
        },
    ])?;
    let direct = rejected(prescribed.evaluate(0.3))?;
    assert!(matches!(direct, StructureError::Numerical { .. }));
    assert_eq!(direct.element(), Some(1));
    let composed =
        rejected(registry.propose_at(0.3, &heads(&context), &[opening("manual", 0.5)], &[]))?;
    assert_eq!(composed, direct);
    Ok(())
}

#[test]
fn subset_admits_closed_point_intersection_while_exhaustive_legacy_retains_positive_duration()
-> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![gate("a")?, gate("b")?],
    )?;
    let a = trajectory(&context, "a", &[(0.0, 0.25), (1.0, 0.5)])?;
    let b = trajectory(&context, "b", &[(1.0, 0.75), (2.0, 0.25)])?;
    let assignments = vec![a.assignments()[0].clone(), b.assignments()[0].clone()];
    let point = context.with_gate_trajectory_subset(assignments.clone())?;
    assert_eq!(point.support().start_s, 1.0);
    assert_eq!(point.support().end_s, 1.0);
    assert_eq!(point.evaluate(1.0)?.outputs()?.len(), 2);
    assert!(point.next_event_after(1.0)?.is_none());
    kind(
        context.with_gate_trajectories(assignments),
        StructureInputErrorKind::Definition,
    )?;
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Prescribed {
        source_id: "point".into(),
        trajectory: &point,
    }])?;
    let support = registry.support().ok_or("Missing point support")?;
    assert_eq!(support.start_s, 1.0);
    assert_eq!(support.end_s, 1.0);
    let proposal = registry.propose_at(1.0, &heads(&context), &[], &[])?;
    explicit_match(&context, &heads(&context), &proposal)?;
    assert!(registry.next_event_after(1.0, &[])?.is_none());
    for time in [
        f64::from_bits(1.0f64.to_bits() - 1),
        f64::from_bits(1.0f64.to_bits() + 1),
    ] {
        kind(
            registry.propose_at(time, &heads(&context), &[], &[]),
            StructureInputErrorKind::UnsupportedRegime,
        )?;
    }
    Ok(())
}
