use super::fixture::*;
use hydr::structures::*;

#[test]
fn exactly_coincident_knots_keep_registration_order_and_common_terminal() -> Result {
    let context = StructureBatch::new(vec![gate("a")?, gate("b")?])?;
    let a = trajectory(&context, "a", &[(0.0, 0.25), (1.0, 0.5), (3.0, 0.75)])?;
    let b = trajectory(&context, "b", &[(0.0, 0.75), (1.0, 0.5), (2.0, 0.25)])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "b-source".into(),
            trajectory: &b,
        },
        OperatingSourceBinding::Prescribed {
            source_id: "a-source".into(),
            trajectory: &a,
        },
    ])?;
    let support = registry.support().ok_or("Missing bounded support")?;
    assert_eq!(support.start_s, 0.0);
    assert_eq!(support.end_s, 2.0);
    let event = registry
        .next_event_after(0.0, &[])?
        .ok_or("Missing shared knot")?;
    assert_eq!(event.time_s, 1.0);
    assert!(!event.terminal);
    assert_eq!(
        event
            .origins
            .iter()
            .map(|origin| origin.source_id.as_str())
            .collect::<Vec<_>>(),
        ["b-source", "a-source"]
    );
    assert!(
        event
            .origins
            .iter()
            .all(|origin| matches!(origin.kind, OperatingSourceEventKind::PrescribedKnot))
    );
    let terminal = registry
        .next_event_after(1.0, &[])?
        .ok_or("Missing common terminal")?;
    assert_eq!(terminal.time_s, 2.0);
    assert!(terminal.terminal);
    assert_eq!(terminal.origins.len(), 1);
    assert_eq!(terminal.origins[0].source_id, "b-source");
    assert!(registry.next_event_after(2.0, &[])?.is_none());
    kind(
        registry.next_event_after(2.5, &[]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    Ok(())
}

#[test]
fn adjacent_representable_knots_remain_separate_events() -> Result {
    let context = StructureBatch::new(vec![gate("a")?, gate("b")?])?;
    let adjacent = f64::from_bits(1.0f64.to_bits() + 1);
    let a = trajectory(&context, "a", &[(0.0, 0.25), (1.0, 0.5), (2.0, 0.75)])?;
    let b = trajectory(&context, "b", &[(0.0, 0.75), (adjacent, 0.5), (2.0, 0.25)])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "a".into(),
            trajectory: &a,
        },
        OperatingSourceBinding::Prescribed {
            source_id: "b".into(),
            trajectory: &b,
        },
    ])?;
    let first = registry
        .next_event_after(0.0, &[])?
        .ok_or("Missing first knot")?;
    assert_eq!(first.time_s.to_bits(), 1.0f64.to_bits());
    assert_eq!(first.origins.len(), 1);
    assert_eq!(first.origins[0].source_id, "a");
    let second = registry
        .next_event_after(1.0, &[])?
        .ok_or("Missing adjacent knot")?;
    assert_eq!(second.time_s.to_bits(), adjacent.to_bits());
    assert_eq!(second.origins.len(), 1);
    assert_eq!(second.origins[0].source_id, "b");
    Ok(())
}

#[test]
fn fresh_actuator_arrival_flags_join_original_knots_without_advancing_sources() -> Result {
    let context = StructureBatch::new(vec![gate("actual")?, gate("path")?])?;
    let actual = actuator(&context, "actual", 0.0, 0.5, (0.5, 0.5), 0.0, 3.0)?;
    let path = trajectory(&context, "path", &[(0.0, 0.25), (1.0, 0.5), (2.0, 0.75)])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Actuated {
            source_id: "actual".into(),
            actuator: &actual,
        },
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &path,
        },
    ])?;
    let before = actual.snapshot_data()?;
    let event = registry
        .next_event_after(
            0.0,
            &[ActuatorSourceFrame {
                source_id: "actual",
                actuator: &actual,
            }],
        )?
        .ok_or("Missing joined event")?;
    assert_eq!(event.time_s, 1.0);
    assert_eq!(event.origins.len(), 2);
    assert_eq!(event.origins[0].source_id, "actual");
    match &event.origins[0].kind {
        OperatingSourceEventKind::Actuator {
            target_index,
            arrival_lower_boundary,
            arrival_upper_boundary,
            terminal,
        } => {
            assert_eq!(*target_index, None);
            assert!(*arrival_lower_boundary);
            assert!(*arrival_upper_boundary);
            assert!(!*terminal);
        }
        _ => return Err("Actuator arrival lost its typed flags".into()),
    }
    assert_eq!(actual.snapshot_data()?, before);
    let actual = advance(actual, 1.0)?;
    let terminal = registry
        .next_event_after(
            1.0,
            &[ActuatorSourceFrame {
                source_id: "actual",
                actuator: &actual,
            }],
        )?
        .ok_or("Missing common terminal")?;
    assert_eq!(terminal.time_s, 2.0);
    assert!(terminal.terminal);
    assert_eq!(terminal.origins.len(), 1);
    assert_eq!(terminal.origins[0].source_id, "path");
    Ok(())
}

#[test]
fn empty_common_support_rejects_and_single_actuator_prescribed_intersection_remains_a_point()
-> Result {
    let context = StructureBatch::new(vec![gate("a")?, gate("b")?])?;
    let a = trajectory(&context, "a", &[(0.0, 0.25), (1.0, 0.5)])?;
    let disjoint = trajectory(&context, "b", &[(2.0, 0.25), (3.0, 0.5)])?;
    kind(
        context.with_operating_sources(vec![
            OperatingSourceBinding::Prescribed {
                source_id: "a".into(),
                trajectory: &a,
            },
            OperatingSourceBinding::Prescribed {
                source_id: "b".into(),
                trajectory: &disjoint,
            },
        ]),
        StructureInputErrorKind::Definition,
    )?;
    let actual = actuator(&context, "b", 0.75, 0.75, (0.0, 0.0), 1.0, 2.0)?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "a".into(),
            trajectory: &a,
        },
        OperatingSourceBinding::Actuated {
            source_id: "b".into(),
            actuator: &actual,
        },
    ])?;
    let support = registry.support().ok_or("Missing closed point support")?;
    assert_eq!(support.start_s, 1.0);
    assert_eq!(support.end_s, 1.0);
    let frames = [ActuatorSourceFrame {
        source_id: "b",
        actuator: &actual,
    }];
    let proposal = registry.propose_at(1.0, &heads(&context), &[], &frames)?;
    explicit_match(&context, &heads(&context), &proposal)?;
    assert!(registry.next_event_after(1.0, &frames)?.is_none());
    Ok(())
}
