use super::fixture::*;
use hydr::structures::*;

fn value(operation: &StructureOperation) -> Result<f64> {
    match operation.operating_point {
        StructureOperatingPoint::GateOpening { opening_m } => Ok(opening_m),
        _ => Err("Automatic gate routing produced a non-gate operation".into()),
    }
}

#[test]
fn many_registered_groups_preserve_original_slot_values_and_source_receipts() -> Result {
    const WIDTH: usize = 129;
    let time = 0.3;
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        (0..WIDTH)
            .map(|lane| gate(&format!("gate-{lane}")))
            .collect::<Result<Vec<_>>>()?,
    )?;
    let source_ids = (0..WIDTH)
        .map(|lane| format!("source-{lane}"))
        .collect::<Vec<_>>();
    let mut prescribed = Vec::with_capacity(WIDTH);
    let mut actual = Vec::with_capacity(WIDTH);
    let mut manual = Vec::new();
    for lane in 0..WIDTH {
        let id = format!("gate-{lane}");
        match lane % 3 {
            0 => {
                let opening_m = if lane == 0 {
                    -0.0
                } else {
                    (lane + 1) as f64 / 260.0
                };
                manual.push(opening(&id, opening_m));
                prescribed.push(None);
                actual.push(None);
            }
            1 => {
                let base = 0.1 + (lane + 1) as f64 / 400.0;
                let knots = if lane == 1 {
                    [(0.1, -0.0), (0.9, -0.0)]
                } else {
                    [(0.1, base), (0.9, base + 0.125 + (lane % 5) as f64 / 100.0)]
                };
                prescribed.push(Some(trajectory(&context, &id, &knots)?));
                actual.push(None);
            }
            2 => {
                let (initial, target, rates) = if lane == 2 {
                    (-0.0, -0.0, (0.0, 0.0))
                } else {
                    (
                        0.1 + (lane + 1) as f64 / 400.0,
                        0.9,
                        (0.2 + (lane % 7) as f64 / 100.0, 0.5),
                    )
                };
                actual.push(Some(advance(
                    actuator(&context, &id, initial, target, rates, 0.0, 1.0)?,
                    time,
                )?));
                prescribed.push(None);
            }
            _ => return Err("Unknown registered gate ownership".into()),
        }
    }
    let bindings = (0..WIDTH)
        .rev()
        .map(|lane| {
            Ok(match lane % 3 {
                0 => OperatingSourceBinding::Manual {
                    source_id: source_ids[lane].clone(),
                    structure_ids: vec![format!("gate-{lane}")],
                },
                1 => OperatingSourceBinding::Prescribed {
                    source_id: source_ids[lane].clone(),
                    trajectory: prescribed[lane]
                        .as_ref()
                        .ok_or("Missing prescribed source")?,
                },
                2 => OperatingSourceBinding::Actuated {
                    source_id: source_ids[lane].clone(),
                    actuator: actual[lane].as_ref().ok_or("Missing actual source")?,
                },
                _ => return Err("Unknown registered source group".into()),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let registry = context.with_operating_sources(bindings)?;
    let frames = (0..WIDTH)
        .rev()
        .filter_map(|lane| {
            actual[lane]
                .as_ref()
                .map(|actuator| OperatingSourceFrame::ActuatedGate {
                    source_id: &source_ids[lane],
                    actuator,
                })
        })
        .collect::<Vec<_>>();
    assert_eq!(frames.len(), WIDTH / 3);
    assert!(matches!(
        frames[0],
        OperatingSourceFrame::ActuatedGate {
            source_id: "source-128",
            ..
        }
    ));
    let mut heads = heads(&context);
    for (lane, packet) in heads.iter_mut().enumerate() {
        packet.endpoint_a.support = HeadSupport::Wet {
            stage_m: 2.0 + (lane % 4) as f64 / 8.0,
        };
        packet.endpoint_b.support = HeadSupport::Wet {
            stage_m: 1.0 + (lane % 3) as f64 / 8.0,
        };
    }
    let proposal = registry.propose_at(time, &heads, &manual, &frames)?;
    let operations = proposal.operations()?;
    let outputs = proposal.outputs()?;
    assert_eq!(operations.len(), WIDTH);
    assert_eq!(outputs.len(), WIDTH);
    let mut manual_index = 0;
    let mut nonzero_prescribed_errors = 0;
    let mut nonzero_actual_errors = 0;
    for (lane, (operation, output)) in operations.iter().zip(&outputs).enumerate() {
        assert_eq!(operation.structure_id, format!("gate-{lane}"));
        assert_eq!(output.source.source_id, source_ids[lane]);
        assert_eq!(output.source.structure_id, operation.structure_id);
        assert_eq!(output.source.element, lane);
        assert_eq!(output.source.time_s.to_bits(), time.to_bits());
        assert_eq!(output.hydraulic.element, lane);
        assert_eq!(output.hydraulic.input, heads[lane]);
        match lane % 3 {
            0 => {
                assert!(matches!(
                    output.source.detail,
                    OperatingSourceDetail::Manual
                ));
                assert_eq!(
                    value(operation)?.to_bits(),
                    value(&manual[manual_index])?.to_bits()
                );
                manual_index += 1;
            }
            1 => {
                let expected = prescribed[lane]
                    .as_ref()
                    .ok_or("Missing prescribed source")?
                    .evaluate(time)?
                    .outputs()?
                    .remove(0);
                assert_eq!(expected.element, lane);
                assert_eq!(value(operation)?.to_bits(), expected.opening_m.to_bits());
                match &output.source.detail {
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
                        if *absolute_opening_error_m > 0.0 {
                            nonzero_prescribed_errors += 1;
                        }
                    }
                    _ => return Err("Prescribed routing lost its source receipt".into()),
                }
            }
            2 => {
                let frame = actual[lane].as_ref().ok_or("Missing actual source")?;
                let expected = frame.outputs()?.remove(0);
                assert_eq!(expected.element, lane);
                assert_eq!(
                    value(operation)?.to_bits(),
                    expected.position.opening_m.to_bits()
                );
                match &output.source.detail {
                    OperatingSourceDetail::ActuatedGate {
                        actuator_id,
                        accepted_steps,
                        accepted_time_s,
                        position,
                        phase,
                        support,
                    } => {
                        assert_eq!(actuator_id, &expected.actuator_id);
                        assert_eq!(*accepted_steps, frame.accepted_steps());
                        assert_eq!(accepted_time_s.to_bits(), frame.time().to_bits());
                        assert_eq!(*position, expected.position);
                        assert_eq!(*phase, expected.phase);
                        assert_eq!(*support, expected.support);
                        if position.absolute_error_m > 0.0 {
                            nonzero_actual_errors += 1;
                        }
                    }
                    _ => return Err("Actual routing lost its source receipt".into()),
                }
            }
            _ => return Err("Unknown projected gate ownership".into()),
        }
    }
    assert_eq!(manual_index, manual.len());
    assert!(nonzero_prescribed_errors > 0);
    assert!(nonzero_actual_errors > 0);
    assert_eq!(value(&operations[0])?.to_bits(), (-0.0f64).to_bits());
    assert_eq!(value(&operations[2])?.to_bits(), (-0.0f64).to_bits());
    explicit_match(&context, &heads, &proposal)
}
