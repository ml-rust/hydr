use super::fixture::*;
use hydr::structures::*;

fn prescribed(
    context: &StructureBatch<crate::runtime::Runtime>,
) -> Result<GateTrajectoryBatch<crate::runtime::Runtime>> {
    Ok(
        context.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
            structure_id: "prescribed".into(),
            trajectory: GateOpeningTrajectory::new(
                vec![
                    GateOpeningKnot {
                        time_s: 0.0,
                        opening_m: 0.25,
                    },
                    GateOpeningKnot {
                        time_s: 4.0,
                        opening_m: 0.75,
                    },
                ],
                GateMotionLimits {
                    maximum_opening_speed_m_per_s: 1.0,
                    maximum_closing_speed_m_per_s: 1.0,
                },
                GateTrajectoryAccuracy {
                    maximum_absolute_opening_error_m: 1e-12,
                    maximum_relative_opening_error: 1e-12,
                },
                "Original analytical aperture schedule".into(),
            )?,
        }])?,
    )
}

#[test]
fn mixed_sources_keep_manual_prescribed_actual_gate_and_controlled_pump_ownership() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![
            pump("manual")?,
            gate("prescribed")?,
            gate("actual")?,
            pump("a")?,
            pump("b")?,
        ],
    )?;
    let config = Config::switching();
    let a = config.build(&context, &["a"])?;
    let b = config.build(&context, &["b"])?;
    let actual = actual_gate(&context, "actual", 4.0)?;
    let prescribed = prescribed(&context)?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::ControlledPump {
            source_id: "pump-b".into(),
            pump: &b,
        },
        OperatingSourceBinding::Manual {
            source_id: "manual-source".into(),
            structure_ids: vec!["manual".into()],
        },
        OperatingSourceBinding::Prescribed {
            source_id: "path".into(),
            trajectory: &prescribed,
        },
        OperatingSourceBinding::Actuated {
            source_id: "actual-source".into(),
            actuator: &actual,
        },
        OperatingSourceBinding::ControlledPump {
            source_id: "pump-a".into(),
            pump: &a,
        },
    ])?;
    let a = accept(a, 1.0, Some(2.0))?;
    let b = accept(b, 1.0, Some(1.5))?;
    let actual = actual.trial_to(1.0, &[])?.into_next();
    let frames = [
        OperatingSourceFrame::ControlledPump {
            source_id: "pump-b",
            pump: &b,
        },
        OperatingSourceFrame::ActuatedGate {
            source_id: "actual-source",
            actuator: &actual,
        },
        OperatingSourceFrame::ControlledPump {
            source_id: "pump-a",
            pump: &a,
        },
    ];
    let packets = context
        .elements()
        .iter()
        .map(|element| wet(&element.id, 0.0))
        .collect::<Vec<_>>();
    let manual = [StructureOperation {
        structure_id: "manual".into(),
        operating_point: StructureOperatingPoint::PumpStoppedIsolated,
    }];
    let proposal = registry.propose_at(1.0, &packets, &manual, &frames)?;
    let operations = proposal.operations()?;
    assert_eq!(
        operations[0].operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    assert_eq!(
        operations[1].operating_point,
        StructureOperatingPoint::GateOpening { opening_m: 0.375 }
    );
    assert_eq!(
        operations[2].operating_point,
        StructureOperatingPoint::GateOpening { opening_m: 0.25 }
    );
    assert_eq!(
        operations[3].operating_point,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
    );
    assert_eq!(
        operations[4].operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    let expected = context
        .propose_with_operations(&packets, &operations)?
        .outputs()?;
    let outputs = proposal.outputs()?;
    for (output, expected) in outputs.iter().zip(expected) {
        assert_eq!(output.hydraulic, expected);
    }
    assert_eq!(
        outputs
            .iter()
            .map(|output| output.source.source_id.as_str())
            .collect::<Vec<_>>(),
        ["manual-source", "path", "actual-source", "pump-a", "pump-b"]
    );
    let event = registry
        .next_event_after(1.0, &frames)?
        .ok_or("Missing next decision")?;
    assert_eq!(event.time_s, 2.0);
    assert_eq!(
        event
            .origins
            .iter()
            .map(|origin| origin.source_id.as_str())
            .collect::<Vec<_>>(),
        ["pump-b", "pump-a"]
    );
    kind(
        registry.propose_at(1.0, &packets, &[], &frames),
        StructureInputErrorKind::OperatingPoint,
    )?;
    let mut override_packet = manual.to_vec();
    override_packet.push(StructureOperation {
        structure_id: "a".into(),
        operating_point: StructureOperatingPoint::PumpStoppedIsolated,
    });
    kind(
        registry.propose_at(1.0, &packets, &override_packet, &frames),
        StructureInputErrorKind::OperatingPoint,
    )?;
    Ok(())
}

#[test]
fn controlled_pump_registration_rejects_duplicate_or_missing_actual_element_writers() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let batch = Config::switching().build(&context, &["pump"])?;
    kind(
        context.with_operating_sources(vec![
            OperatingSourceBinding::Manual {
                source_id: "manual".into(),
                structure_ids: vec!["pump".into()],
            },
            OperatingSourceBinding::ControlledPump {
                source_id: "controlled".into(),
                pump: &batch,
            },
        ]),
        StructureInputErrorKind::Identity,
    )?;
    kind(
        context.with_operating_sources(Vec::new()),
        StructureInputErrorKind::Identity,
    )
}

#[test]
fn many_reversed_pump_groups_preserve_original_slots_operations_and_response_provenance() -> Result
{
    let names = (0..129)
        .map(|index| format!("pump-{index}"))
        .collect::<Vec<_>>();
    let sources = (0..names.len())
        .map(|index| format!("source-{index}"))
        .collect::<Vec<_>>();
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        names
            .iter()
            .map(|id| pump(id))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let mut pumps = Vec::with_capacity(names.len());
    for (index, name) in names.iter().enumerate() {
        let mut config = Config::switching();
        config.commands[0] = match index % 4 {
            0 => ControlCommand::PumpStoppedIsolated,
            1 => ControlCommand::PumpRunning { speed_ratio: 0.5 },
            2 => ControlCommand::PumpRunning { speed_ratio: 0.75 },
            _ => ControlCommand::PumpRunning { speed_ratio: 1.0 },
        };
        pumps.push(config.build(&context, &[name.as_str()])?);
    }
    let bindings = pumps
        .iter()
        .zip(&sources)
        .rev()
        .map(|(pump, source_id)| OperatingSourceBinding::ControlledPump {
            source_id: source_id.clone(),
            pump,
        })
        .collect();
    let registry = context.with_operating_sources(bindings)?;
    let frames = pumps
        .iter()
        .zip(&sources)
        .rev()
        .map(|(pump, source_id)| OperatingSourceFrame::ControlledPump { source_id, pump })
        .collect::<Vec<_>>();
    let packets = names.iter().map(|name| wet(name, 0.0)).collect::<Vec<_>>();
    let proposal = registry.propose_at(0.0, &packets, &[], &frames)?;
    let operations = proposal.operations()?;
    let direct = context
        .propose_with_operations(&packets, &operations)?
        .outputs()?;
    for (index, (output, expected)) in proposal.outputs()?.iter().zip(direct).enumerate() {
        assert_eq!(output.hydraulic, expected);
        assert_eq!(output.hydraulic.element, index);
        assert_eq!(output.source.source_id, sources[index]);
        assert_eq!(operations[index].structure_id, names[index]);
        match &output.source.detail {
            OperatingSourceDetail::ControlledPump {
                controller_id,
                accepted_steps,
                accepted_time_s,
                response_model,
                operating_point,
                ..
            } => {
                assert_eq!(controller_id, &controller(&names[index]));
                assert_eq!(*accepted_steps, 0);
                assert_eq!(accepted_time_s.to_bits(), 0.0_f64.to_bits());
                assert_eq!(*response_model, PumpResponseModel::IdealInstantaneous);
                assert_eq!(*operating_point, operations[index].operating_point);
            }
            _ => return Err("Pump source lost its per-group response receipt".into()),
        }
    }
    Ok(())
}
