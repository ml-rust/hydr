use super::fixture::*;
use hydr::structures::*;

#[test]
fn actuated_frames_require_registered_order_identity_and_exact_accepted_time() -> Result {
    let fixture = Mixed::new()?;
    let heads = heads(&fixture.context);
    let manual = fixture.manual();
    kind(
        fixture.registry.propose_at(0.0, &heads, &manual, &[]),
        StructureInputErrorKind::Identity,
    )?;
    kind(
        fixture.registry.propose_at(
            0.0,
            &heads,
            &manual,
            &[OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &fixture.actuators[1],
            }],
        ),
        StructureInputErrorKind::Identity,
    )?;
    for frames in [
        vec![
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-a",
                actuator: &fixture.actuators[0],
            },
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &fixture.actuators[1],
            },
        ],
        vec![
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &fixture.actuators[1],
            },
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &fixture.actuators[0],
            },
        ],
        vec![
            OperatingSourceFrame::ActuatedGate {
                source_id: "foreign",
                actuator: &fixture.actuators[1],
            },
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-a",
                actuator: &fixture.actuators[0],
            },
        ],
    ] {
        kind(
            fixture.registry.propose_at(0.0, &heads, &manual, &frames),
            StructureInputErrorKind::Identity,
        )?;
        kind(
            fixture.registry.next_event_after(0.0, &frames),
            StructureInputErrorKind::Identity,
        )?;
    }
    let mut extra = fixture.frames();
    extra.push(OperatingSourceFrame::ActuatedGate {
        source_id: "foreign",
        actuator: &fixture.actuators[0],
    });
    kind(
        fixture.registry.propose_at(0.0, &heads, &manual, &extra),
        StructureInputErrorKind::Identity,
    )?;
    let later = advance(fixture.actuators[0].clone(), 1.0)?;
    let frames = [
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-b",
            actuator: &fixture.actuators[1],
        },
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-a",
            actuator: &later,
        },
    ];
    kind(
        fixture.registry.propose_at(0.0, &heads, &manual, &frames),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    kind(
        fixture.registry.next_event_after(0.0, &frames),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    kind(
        fixture
            .registry
            .propose_at(1.0, &heads, &manual, &fixture.frames()),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    Ok(())
}

#[test]
fn separately_reconstructed_and_cold_restored_frames_preserve_full_immutable_identity() -> Result {
    let fixture = Mixed::new()?;
    let source = &fixture.actuators[0];
    let reconstructed_context = StructureBatch::new(fixture.context.elements().to_vec())?;
    let rebuilt = reconstructed_context.with_gate_actuators(
        source.definitions().to_vec(),
        source.initial_states().to_vec(),
        source.schedule().clone(),
    )?;
    let frames = [
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-b",
            actuator: &fixture.actuators[1],
        },
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-a",
            actuator: &rebuilt,
        },
    ];
    let heads = heads(&fixture.context);
    let expected = fixture
        .registry
        .propose_at(0.0, &heads, &fixture.manual(), &fixture.frames())?
        .outputs()?;
    assert_eq!(
        fixture
            .registry
            .propose_at(0.0, &heads, &fixture.manual(), &frames)?
            .outputs()?,
        expected
    );
    let advanced_a = advance(source.clone(), 1.0)?;
    let advanced_b = advance(fixture.actuators[1].clone(), 1.0)?;
    let mut restored = rebuilt;
    restored.restore_data(&advanced_a.snapshot_data()?)?;
    let frames = [
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-b",
            actuator: &advanced_b,
        },
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-a",
            actuator: &restored,
        },
    ];
    let proposal = fixture
        .registry
        .propose_at(1.0, &heads, &fixture.manual(), &frames)?;
    explicit_match(&fixture.context, &heads, &proposal)?;
    Ok(())
}

#[test]
fn matching_current_centers_do_not_waive_original_frame_definition_or_context_bits() -> Result {
    let fixture = Mixed::new()?;
    let source = &fixture.actuators[0];
    let heads = heads(&fixture.context);
    for variant in 0..4 {
        let mut definitions = source.definitions().to_vec();
        let mut initial = source.initial_states().to_vec();
        let mut schedule = source.schedule().clone();
        let mut elements = fixture.context.elements().to_vec();
        match variant {
            0 => definitions[0].accuracy.maximum_arrival_window_s = 1e-9,
            1 => initial[0].target_opening_m = 0.75,
            2 => schedule.end_s = 5.0,
            3 => elements[0].endpoint_b.push_str(" altered"),
            _ => return Err("Unknown frame-context mutation".into()),
        }
        let context = StructureBatch::new(elements)?;
        let foreign = context.with_gate_actuators(definitions, initial, schedule)?;
        assert_eq!(
            foreign.outputs()?[0].position.opening_m,
            source.outputs()?[0].position.opening_m
        );
        let frames = [
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-b",
                actuator: &fixture.actuators[1],
            },
            OperatingSourceFrame::ActuatedGate {
                source_id: "actual-a",
                actuator: &foreign,
            },
        ];
        kind(
            fixture
                .registry
                .propose_at(0.0, &heads, &fixture.manual(), &frames),
            StructureInputErrorKind::Definition,
        )?;
    }
    Ok(())
}

#[test]
fn numerically_equal_signed_zero_frame_and_query_times_keep_separate_original_receipts() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    for (frame_time, query_time) in [(-0.0, 0.0), (0.0, -0.0)] {
        let actual = actuator(&context, "gate", 0.0, 0.0, (0.0, 0.0), frame_time, 1.0)?;
        let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
            source_id: "actual".into(),
            actuator: &actual,
        }])?;
        let frames = [OperatingSourceFrame::ActuatedGate {
            source_id: "actual",
            actuator: &actual,
        }];
        let proposal = registry.propose_at(query_time, &heads(&context), &[], &frames)?;
        assert_eq!(proposal.time_s().to_bits(), query_time.to_bits());
        let output = proposal.outputs()?.remove(0);
        assert_eq!(output.source.time_s.to_bits(), query_time.to_bits());
        match output.source.detail {
            OperatingSourceDetail::ActuatedGate {
                accepted_time_s, ..
            } => assert_eq!(accepted_time_s.to_bits(), frame_time.to_bits()),
            _ => return Err("Actuator point lost its original accepted timestamp".into()),
        }
    }
    Ok(())
}

#[test]
fn raw_initial_clock_bits_remain_part_of_immutable_actuator_frame_identity() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    let registered = actuator(&context, "gate", 0.0, 0.0, (0.0, 0.0), -0.0, 1.0)?;
    let foreign = actuator(&context, "gate", 0.0, 0.0, (0.0, 0.0), 0.0, 1.0)?;
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
        source_id: "actual".into(),
        actuator: &registered,
    }])?;
    assert_eq!(registered.time(), foreign.time());
    assert_ne!(
        registered.schedule().initial_time_s.to_bits(),
        foreign.schedule().initial_time_s.to_bits()
    );
    kind(
        registry.propose_at(
            0.0,
            &heads(&context),
            &[],
            &[OperatingSourceFrame::ActuatedGate {
                source_id: "actual",
                actuator: &foreign,
            }],
        ),
        StructureInputErrorKind::Definition,
    )
}
