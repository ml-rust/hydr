use super::fixture::*;
use hydr::structures::*;

#[test]
fn distinct_interleaved_lane_values_survive_compact_gather_scatter_at_vector_and_tail_sizes()
-> Result {
    for width in [1, 31, 32, 33, 64, 65] {
        let context = StructureBatch::new(
            (0..width)
                .map(|lane| gate(&format!("gate-{lane}")))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let manual_slots = (0..width).filter(|lane| lane % 3 == 0).collect::<Vec<_>>();
        let prescribed_slots = (0..width).filter(|lane| lane % 3 == 1).collect::<Vec<_>>();
        let actual_slots = (0..width).filter(|lane| lane % 3 == 2).collect::<Vec<_>>();
        let manual = manual_slots
            .iter()
            .map(|&lane| {
                opening(
                    &format!("gate-{lane}"),
                    (lane + 1) as f64 / (width + 2) as f64,
                )
            })
            .collect::<Vec<_>>();
        let prescribed = if prescribed_slots.is_empty() {
            None
        } else {
            let assignments = prescribed_slots
                .iter()
                .map(|&lane| {
                    let base = (lane + 1) as f64 / 128.0;
                    let path = GateOpeningTrajectory::new(
                        vec![
                            GateOpeningKnot {
                                time_s: 0.0,
                                opening_m: base,
                            },
                            GateOpeningKnot {
                                time_s: 1.0,
                                opening_m: base + 1.0 / 128.0,
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
                        "Distinct dyadic lane trajectory".into(),
                    )?;
                    Ok(GateTrajectoryAssignment {
                        structure_id: format!("gate-{lane}"),
                        trajectory: path,
                    })
                })
                .collect::<std::result::Result<Vec<_>, StructureError>>()?;
            Some(context.with_gate_trajectory_subset(assignments)?)
        };
        let actual = if actual_slots.is_empty() {
            None
        } else {
            let definitions = actual_slots
                .iter()
                .map(|&lane| GateActuatorDefinition {
                    actuator_id: format!("actuator-{lane}"),
                    structure_id: format!("gate-{lane}"),
                    opening_rate_m_per_s: 1.0 / 64.0,
                    closing_rate_m_per_s: 1.0 / 32.0,
                    accuracy: GateActuatorAccuracy {
                        maximum_absolute_opening_error_m: 1e-12,
                        maximum_relative_opening_error: 1e-12,
                        maximum_arrival_window_s: 1e-10,
                    },
                    rate_provenance: "Explicit lane rates".into(),
                    ideal_response_applicability:
                        "Ideal rate-limited response without inertia or backlash".into(),
                })
                .collect();
            let initial = actual_slots
                .iter()
                .map(|&lane| GateActuatorInitialState {
                    actuator_id: format!("actuator-{lane}"),
                    opening_m: (lane + 1) as f64 / 128.0,
                    target_opening_m: 1.0,
                })
                .collect();
            Some(advance(
                context.with_gate_actuators(
                    definitions,
                    initial,
                    GateActuatorSchedule {
                        initial_time_s: 0.0,
                        target_times_s: vec![],
                        end_s: 1.0,
                    },
                )?,
                0.5,
            )?)
        };
        let mut bindings = Vec::new();
        if let Some(actual) = &actual {
            bindings.push(OperatingSourceBinding::Actuated {
                source_id: "actual".into(),
                actuator: actual,
            });
        }
        bindings.push(OperatingSourceBinding::Manual {
            source_id: "manual".into(),
            structure_ids: manual_slots
                .iter()
                .map(|lane| format!("gate-{lane}"))
                .collect(),
        });
        if let Some(prescribed) = &prescribed {
            bindings.push(OperatingSourceBinding::Prescribed {
                source_id: "path".into(),
                trajectory: prescribed,
            });
        }
        let registry = context.with_operating_sources(bindings)?;
        let frames = actual
            .as_ref()
            .map(|actual| {
                vec![ActuatorSourceFrame {
                    source_id: "actual",
                    actuator: actual,
                }]
            })
            .unwrap_or_default();
        let heads = heads(&context);
        let proposal = registry.propose_at(0.5, &heads, &manual, &frames)?;
        let operations = proposal.operations()?;
        let outputs = proposal.outputs()?;
        assert_eq!(operations.len(), width);
        assert_eq!(outputs.len(), width);
        for (lane, (operation, output)) in operations.iter().zip(&outputs).enumerate() {
            let (source, expected) = match lane % 3 {
                0 => ("manual", (lane + 1) as f64 / (width + 2) as f64),
                1 => ("path", (lane + 1) as f64 / 128.0 + 1.0 / 256.0),
                2 => ("actual", (lane + 1) as f64 / 128.0 + 1.0 / 128.0),
                _ => return Err("Unknown lane ownership".into()),
            };
            assert_eq!(operation.structure_id, format!("gate-{lane}"));
            assert_eq!(
                operation.operating_point,
                StructureOperatingPoint::GateOpening {
                    opening_m: expected
                }
            );
            assert_eq!(output.source.source_id, source);
            assert_eq!(output.source.element, lane);
        }
        explicit_match(&context, &heads, &proposal)?;
    }
    Ok(())
}
