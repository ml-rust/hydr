use super::fixture::*;
use hydr::structures::*;

#[test]
fn interleaved_source_groups_match_explicit_center_laws_and_keep_typed_uncertainty() -> Result {
    let fixture = Mixed::new()?;
    let heads = heads(&fixture.context);
    let frames = fixture.frames();
    let proposal = fixture
        .registry
        .propose_at(0.0, &heads, &fixture.manual(), &frames)?;
    explicit_match(&fixture.context, &heads, &proposal)?;
    let operations = proposal.operations()?;
    let expected = [0.125, 0.125, 0.0, 0.25, 0.0, 0.75, 0.75, 0.875];
    for (slot, operation) in operations.iter().enumerate() {
        if let StructureOperatingPoint::GateOpening { opening_m } = operation.operating_point {
            assert_eq!(opening_m, expected[slot], "slot={slot}");
        }
    }
    let outputs = proposal.outputs()?;
    let sources = [
        "manual", "path-a", "manual", "actual-a", "manual", "path-b", "actual-b", "manual",
    ];
    for (slot, output) in outputs.iter().enumerate() {
        assert_eq!(output.source.element, slot);
        assert_eq!(output.source.source_id, sources[slot]);
        assert_eq!(
            output.source.structure_id,
            fixture.context.elements()[slot].id
        );
        assert_eq!(output.hydraulic.element, slot);
    }
    for (index, slot) in [(0, 3), (1, 6)] {
        let state = fixture.actuators[index].outputs()?.remove(0);
        match &outputs[slot].source.detail {
            OperatingSourceDetail::ActuatedGate {
                actuator_id,
                accepted_steps,
                accepted_time_s,
                position,
                phase,
                support,
            } => {
                assert_eq!(actuator_id, &state.actuator_id);
                assert_eq!(*accepted_steps, fixture.actuators[index].accepted_steps());
                assert_eq!(
                    accepted_time_s.to_bits(),
                    fixture.actuators[index].time().to_bits()
                );
                assert_eq!(*position, state.position);
                assert_eq!(*phase, state.phase);
                assert_eq!(*support, state.support);
            }
            _ => return Err("Actuated source lost its typed position receipt".into()),
        }
    }
    for (index, slot) in [(0, 1), (1, 5)] {
        let expected = fixture.prescribed[index]
            .evaluate(0.0)?
            .outputs()?
            .remove(0);
        match &outputs[slot].source.detail {
            OperatingSourceDetail::PrescribedGate {
                absolute_opening_error_m,
                segment_index,
                exact_knot_index,
                closed_by_definition,
            } => {
                assert_eq!(
                    absolute_opening_error_m.to_bits(),
                    expected.absolute_error_m.to_bits()
                );
                assert_eq!(*segment_index, expected.segment_index);
                assert_eq!(*exact_knot_index, expected.exact_knot_index);
                assert_eq!(*closed_by_definition, expected.closed_by_definition);
            }
            _ => return Err("Prescribed source lost its interpolation receipt".into()),
        }
    }
    Ok(())
}

#[test]
fn repeated_projections_and_queries_preserve_sources_and_earlier_point_proposals() -> Result {
    let mut fixture = Mixed::new()?;
    let heads = heads(&fixture.context);
    let before = [
        fixture.actuators[0].snapshot_data()?,
        fixture.actuators[1].snapshot_data()?,
    ];
    let proposal =
        fixture
            .registry
            .propose_at(0.0, &heads, &fixture.manual(), &fixture.frames())?;
    let saved_outputs = proposal.outputs()?;
    let saved_operations = proposal.operations()?;
    for _ in 0..3 {
        assert_eq!(proposal.outputs()?, saved_outputs);
        assert_eq!(proposal.operations()?, saved_operations);
        let repeated =
            fixture
                .registry
                .propose_at(0.0, &heads, &fixture.manual(), &fixture.frames())?;
        assert_eq!(repeated.outputs()?, saved_outputs);
        assert_eq!(fixture.actuators[0].snapshot_data()?, before[0]);
        assert_eq!(fixture.actuators[1].snapshot_data()?, before[1]);
    }
    fixture.actuators[0] = advance(fixture.actuators[0].clone(), 1.0)?;
    fixture.actuators[1] = advance(fixture.actuators[1].clone(), 1.0)?;
    let later = fixture
        .registry
        .propose_at(1.0, &heads, &fixture.manual(), &fixture.frames())?;
    explicit_match(&fixture.context, &heads, &later)?;
    assert_ne!(later.operations()?, saved_operations);
    assert_eq!(proposal.outputs()?, saved_outputs);
    assert_eq!(proposal.operations()?, saved_operations);
    Ok(())
}

#[test]
fn nonzero_source_error_receipts_remain_separate_from_center_discharge() -> Result {
    let context = StructureBatch::new(vec![gate("path")?, gate("actual")?])?;
    let path = trajectory(&context, "path", &[(0.1, 0.2), (0.9, 0.8)])?;
    let actual = advance(
        actuator(&context, "actual", 0.2, 0.9, (0.3, 0.7), 0.0, 2.0)?,
        0.3,
    )?;
    let path_receipt = path.evaluate(0.3)?.outputs()?.remove(0);
    let actual_receipt = actual.outputs()?.remove(0);
    assert!(path_receipt.absolute_error_m > 0.0);
    assert!(actual_receipt.position.absolute_error_m > 0.0);
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &path,
        },
        OperatingSourceBinding::Actuated {
            source_id: "actual".into(),
            actuator: &actual,
        },
    ])?;
    let heads = heads(&context);
    let proposal = registry.propose_at(
        0.3,
        &heads,
        &[],
        &[ActuatorSourceFrame {
            source_id: "actual",
            actuator: &actual,
        }],
    )?;
    explicit_match(&context, &heads, &proposal)?;
    let outputs = proposal.outputs()?;
    match &outputs[0].source.detail {
        OperatingSourceDetail::PrescribedGate {
            absolute_opening_error_m,
            ..
        } => assert_eq!(
            absolute_opening_error_m.to_bits(),
            path_receipt.absolute_error_m.to_bits()
        ),
        _ => return Err("Prescribed error receipt is missing".into()),
    }
    match &outputs[1].source.detail {
        OperatingSourceDetail::ActuatedGate { position, .. } => {
            assert_eq!(*position, actual_receipt.position)
        }
        _ => return Err("Actual position enclosure is missing".into()),
    }
    Ok(())
}
