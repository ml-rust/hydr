use super::fixture::*;
use hydr::structures::*;

fn unknown(context: &StructureBatch) -> Vec<StructureHeads> {
    let mut heads = heads(context);
    for packet in &mut heads {
        packet.endpoint_a.support = HeadSupport::Unknown;
        packet.endpoint_b.support = HeadSupport::Unknown;
    }
    heads
}

#[test]
fn canonical_closed_sources_waive_only_unknown_support_and_keep_original_packets() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    let path = trajectory(&context, "gate", &[(0.0, 0.0), (1.0, 1.0)])?;
    let actual = actuator(&context, "gate", 0.0, 1.0, (0.3, 0.3), 0.0, 1.0)?;
    let heads = unknown(&context);
    for source in 0..3 {
        let binding = match source {
            0 => OperatingSourceBinding::Manual {
                source_id: "source".into(),
                structure_ids: vec!["gate".into()],
            },
            1 => OperatingSourceBinding::Prescribed {
                source_id: "source".into(),
                trajectory: &path,
            },
            2 => OperatingSourceBinding::Actuated {
                source_id: "source".into(),
                actuator: &actual,
            },
            _ => return Err("Unknown source fixture".into()),
        };
        let registry = context.with_operating_sources(vec![binding])?;
        let manual = if source == 0 {
            vec![opening("gate", 0.0)]
        } else {
            vec![]
        };
        let frames = if source == 2 {
            vec![OperatingSourceFrame::ActuatedGate {
                source_id: "source",
                actuator: &actual,
            }]
        } else {
            vec![]
        };
        let proposal = registry.propose_at(0.0, &heads, &manual, &frames)?;
        let output = proposal.outputs()?.remove(0);
        assert_eq!(output.hydraulic.regime, StructureRegime::GateClosed);
        assert_eq!(output.hydraulic.discharge_m3_per_s, 0.0);
        assert!(output.hydraulic.hydraulic_operands.is_none());
        assert_eq!(output.hydraulic.input, heads[0]);
    }
    let stalled = actuator(&context, "gate", 0.0, 1.0, (0.0, 1.0), 0.0, 1.0)?;
    let stalled = advance(stalled, 0.5)?;
    assert_eq!(
        stalled.outputs()?[0].support,
        GateActuatorSupport::ExactClosed
    );
    assert_eq!(stalled.outputs()?[0].phase, GateActuatorPhase::Stalled);
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
        source_id: "stalled".into(),
        actuator: &stalled,
    }])?;
    let proposal = registry.propose_at(
        0.5,
        &heads,
        &[],
        &[OperatingSourceFrame::ActuatedGate {
            source_id: "stalled",
            actuator: &stalled,
        }],
    )?;
    assert_eq!(
        proposal.outputs()?[0].hydraulic.regime,
        StructureRegime::GateClosed
    );
    Ok(())
}

#[test]
fn malformed_known_heads_reject_before_every_closed_source_waiver() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    let path = trajectory(&context, "gate", &[(0.0, 0.0), (1.0, 1.0)])?;
    let actual = actuator(&context, "gate", 0.0, 1.0, (0.3, 0.3), 0.0, 1.0)?;
    for source in 0..3 {
        let binding = match source {
            0 => OperatingSourceBinding::Manual {
                source_id: "source".into(),
                structure_ids: vec!["gate".into()],
            },
            1 => OperatingSourceBinding::Prescribed {
                source_id: "source".into(),
                trajectory: &path,
            },
            2 => OperatingSourceBinding::Actuated {
                source_id: "source".into(),
                actuator: &actual,
            },
            _ => return Err("Unknown source fixture".into()),
        };
        let registry = context.with_operating_sources(vec![binding])?;
        let manual = if source == 0 {
            vec![opening("gate", 0.0)]
        } else {
            vec![]
        };
        let frames = if source == 2 {
            vec![OperatingSourceFrame::ActuatedGate {
                source_id: "source",
                actuator: &actual,
            }]
        } else {
            vec![]
        };
        for variant in 0..5 {
            let mut heads = unknown(&context);
            heads[0].endpoint_a.support = HeadSupport::Wet { stage_m: 1.0 };
            let expected = match variant {
                0 => {
                    heads[0].endpoint_a.support = HeadSupport::Wet { stage_m: f64::NAN };
                    StructureInputErrorKind::NonfiniteInput
                }
                1 => {
                    heads[0].datum = "Different datum".into();
                    StructureInputErrorKind::Datum
                }
                2 => {
                    heads[0].convention = HeadConvention::EnergyHead;
                    StructureInputErrorKind::HeadConvention
                }
                3 => {
                    heads[0].endpoint_a.id = "foreign endpoint".into();
                    StructureInputErrorKind::Identity
                }
                4 => {
                    heads[0].endpoint_a.support = HeadSupport::Dry {
                        bed_elevation_m: f64::INFINITY,
                    };
                    StructureInputErrorKind::NonfiniteInput
                }
                _ => return Err("Unknown head fixture".into()),
            };
            let error = rejected(registry.propose_at(0.0, &heads, &manual, &frames))?;
            assert_eq!(error.input_kind(), Some(expected), "{error}");
            assert_eq!(error.element(), Some(0));
        }
    }
    Ok(())
}

#[test]
fn positive_openings_and_zero_targets_never_isolate_unknown_heads() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    let heads = unknown(&context);
    let manual = context.with_operating_sources(vec![OperatingSourceBinding::Manual {
        source_id: "manual".into(),
        structure_ids: vec!["gate".into()],
    }])?;
    for opening_m in [f64::from_bits(1), 0.25] {
        kind(
            manual.propose_at(0.0, &heads, &[opening("gate", opening_m)], &[]),
            StructureInputErrorKind::UnknownSupport,
        )?;
    }
    let path = trajectory(
        &context,
        "gate",
        &[(0.0, f64::from_bits(1)), (1.0, f64::from_bits(1))],
    )?;
    let prescribed = context.with_operating_sources(vec![OperatingSourceBinding::Prescribed {
        source_id: "path".into(),
        trajectory: &path,
    }])?;
    kind(
        prescribed.propose_at(0.0, &heads, &[], &[]),
        StructureInputErrorKind::UnknownSupport,
    )?;
    for rates in [(1.0, 0.25), (1.0, 0.0)] {
        let actual = actuator(&context, "gate", 1.0, 0.0, rates, 0.0, 1.0)?;
        let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
            source_id: "actual".into(),
            actuator: &actual,
        }])?;
        kind(
            registry.propose_at(
                0.0,
                &heads,
                &[],
                &[OperatingSourceFrame::ActuatedGate {
                    source_id: "actual",
                    actuator: &actual,
                }],
            ),
            StructureInputErrorKind::UnknownSupport,
        )?;
    }
    Ok(())
}

#[test]
fn unresolved_actual_closure_keeps_positive_center_and_unknown_heads_influential() -> Result {
    let context = StructureBatch::new(vec![gate("gate")?])?;
    let original = actuator(&context, "gate", 0.0, 1.0, (0.3, 0.3), 0.0, 1.0)?;
    let mut definitions = original.definitions().to_vec();
    definitions[0].accuracy.maximum_absolute_opening_error_m = f64::from_bits(2);
    definitions[0].accuracy.maximum_relative_opening_error = 0.0;
    let actual = context.with_gate_actuators(
        definitions,
        original.initial_states().to_vec(),
        original.schedule().clone(),
    )?;
    let actual = advance(actual, f64::from_bits(2))?;
    let output = actual.outputs()?.remove(0);
    assert_eq!(output.position.opening_m.to_bits(), 1);
    assert_eq!(output.position.lower_opening_m.to_bits(), 0);
    assert_eq!(output.position.upper_opening_m.to_bits(), 2);
    assert_eq!(output.position.absolute_error_m.to_bits(), 1);
    assert_eq!(output.support, GateActuatorSupport::ClosureUnresolved);
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
        source_id: "actual".into(),
        actuator: &actual,
    }])?;
    kind(
        registry.propose_at(
            actual.time(),
            &unknown(&context),
            &[],
            &[OperatingSourceFrame::ActuatedGate {
                source_id: "actual",
                actuator: &actual,
            }],
        ),
        StructureInputErrorKind::UnknownSupport,
    )
}

#[test]
fn prescribed_padding_cannot_isolate_an_unassigned_positive_manual_gate() -> Result {
    let context = StructureBatch::new(vec![gate("path")?, gate("manual")?])?;
    let path = trajectory(&context, "path", &[(0.0, 0.0), (1.0, 1.0)])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &path,
        },
        OperatingSourceBinding::Manual {
            source_id: "manual".into(),
            structure_ids: vec!["manual".into()],
        },
    ])?;
    let error = rejected(registry.propose_at(
        0.0,
        &unknown(&context),
        &[opening("manual", f64::from_bits(1))],
        &[],
    ))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::UnknownSupport)
    );
    assert_eq!(error.element(), Some(1));
    Ok(())
}

#[test]
fn explicit_stopped_pump_isolation_preserves_unknown_support_while_running_rejects() -> Result {
    let context = StructureBatch::new(vec![pump("pump")?])?;
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Manual {
        source_id: "manual".into(),
        structure_ids: vec!["pump".into()],
    }])?;
    let heads = unknown(&context);
    let stopped = registry.propose_at(
        0.0,
        &heads,
        &[operation(
            "pump",
            StructureOperatingPoint::PumpStoppedIsolated,
        )],
        &[],
    )?;
    assert_eq!(
        stopped.outputs()?[0].hydraulic.regime,
        StructureRegime::PumpIsolated
    );
    assert_eq!(stopped.outputs()?[0].hydraulic.input, heads[0]);
    kind(
        registry.propose_at(
            0.0,
            &heads,
            &[operation(
                "pump",
                StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            )],
            &[],
        ),
        StructureInputErrorKind::UnknownSupport,
    )
}
