use super::{reference_case::*, reference_inputs::cases};
use hydr::structures::*;

#[test]
fn frozen_physical_and_policy_cases_cover_every_grouped_layout() -> Result {
    assert_eq!(cases().count(), 31);
    for case in cases() {
        for width in LAYOUTS {
            let definition = match trajectory(case) {
                Ok(value) => {
                    assert!(!case.static_invalid, "{}", case.id);
                    value
                }
                Err(error) => {
                    assert!(case.static_invalid, "{}", case.id);
                    check_error(&error, case.binding, None)?;
                    continue;
                }
            };
            let batch = match structure_batch(case, width)?
                .with_gate_trajectories(assignments(&definition, width))
            {
                Ok(value) => {
                    assert_eq!(case.binding, "success", "{}", case.id);
                    value
                }
                Err(error) => {
                    check_error(&error, case.binding, Some(0))?;
                    continue;
                }
            };
            let support = batch.support();
            assert_eq!(
                support.start_s.to_bits(),
                case.knots
                    .first()
                    .ok_or("Missing support start")?
                    .0
                    .to_bits()
            );
            assert_eq!(
                support.end_s.to_bits(),
                case.knots.last().ok_or("Missing support end")?.0.to_bits()
            );
            for index in 0..case.queries.len() {
                let query = &case.queries[(index + width) % case.queries.len()];
                match batch.evaluate(query.time) {
                    Ok(evaluation) => {
                        assert_eq!(query.expected, "success", "{}", case.id);
                        assert_eq!(evaluation.time_s().to_bits(), query.time.to_bits());
                        let outputs = evaluation.outputs()?;
                        let operations = evaluation.operations()?;
                        assert_eq!(outputs.len(), width);
                        assert_eq!(operations.len(), width);
                        for (lane, (value, operation)) in
                            outputs.iter().zip(&operations).enumerate()
                        {
                            check_output(case, query, value, lane)?;
                            assert_eq!(operation.structure_id, element_id(lane));
                            assert_eq!(
                                operation.operating_point,
                                StructureOperatingPoint::GateOpening {
                                    opening_m: value.opening_m
                                }
                            );
                        }
                    }
                    Err(error) => check_error(
                        &error,
                        query.expected,
                        if matches!(query.expected, "Numerical" | "Runtime") {
                            Some(0)
                        } else {
                            None
                        },
                    )?,
                }
            }
        }
    }
    Ok(())
}

#[test]
fn mixed_law_motion_and_accuracy_errors_keep_original_element_indices() -> Result {
    use super::fixture;
    for width in LAYOUTS {
        for (offset, id) in [
            "nonzero_duration_low_exceeds_limit",
            "ordinary_near_closure_exact_only",
        ]
        .into_iter()
        .enumerate()
        {
            let reference = fixture::case(id)?;
            let slot = (width / 2 + offset) % width;
            let original = trajectory(reference)?;
            let alternate = if offset == 0 {
                fixture::stationary(
                    0.0,
                    reference.knots[0].0,
                    reference.knots[reference.knots.len() - 1].0,
                )?
            } else {
                GateOpeningTrajectory::new(
                    original.knots().to_vec(),
                    original.motion_limits(),
                    fixture::permissive(),
                    original.provenance().into(),
                )?
            };
            let mut elements = Vec::new();
            let mut assigned = Vec::new();
            for lane in 0..width {
                if lane != slot && lane % 3 == 0 {
                    elements.push(fixture::passive(&format!("passive-{lane}"))?);
                } else if lane != slot && lane % 3 == 1 {
                    elements.push(fixture::pump(&format!("pump-{lane}"))?);
                } else {
                    let id = element_id(lane);
                    elements.push(gate(&id, reference.maximum_opening)?);
                    assigned.push(GateTrajectoryAssignment {
                        structure_id: id,
                        trajectory: if lane == slot {
                            original.clone()
                        } else {
                            alternate.clone()
                        },
                    });
                }
            }
            let result =
                StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?
                    .with_gate_trajectories(assigned);
            if offset == 0 {
                fixture::expect_error(result, "Numerical", Some(slot))?;
            } else {
                let batch = result?;
                fixture::expect_error(
                    batch.evaluate(reference.queries[0].time),
                    "Numerical",
                    Some(slot),
                )?;
            }
        }
    }
    Ok(())
}

#[test]
fn later_runtime_error_groups_report_their_first_original_element() -> Result {
    use super::fixture;
    let reference = fixture::case("subnormal_opening_ramp")?;
    let invalid = trajectory(reference)?;
    let valid = fixture::stationary(0.5, 0.0, 1.0)?;
    assert_eq!(reference.queries[0].expected, "Runtime");
    for width in [31, 32, 33, 64, 65] {
        let slot = width / 2;
        let invalid_slots = [slot, slot + 2];
        let mut elements = Vec::new();
        let mut assigned = Vec::new();
        for lane in 0..width {
            if invalid_slots.contains(&lane) {
                let id = element_id(lane);
                elements.push(gate(&id, reference.maximum_opening)?);
                assigned.push(GateTrajectoryAssignment {
                    structure_id: id,
                    trajectory: invalid.clone(),
                });
            } else if lane % 3 == 0 {
                elements.push(fixture::passive(&format!("passive-{lane}"))?);
            } else if lane % 3 == 1 {
                elements.push(fixture::pump(&format!("pump-{lane}"))?);
            } else {
                let id = element_id(lane);
                elements.push(gate(&id, 1.0)?);
                assigned.push(GateTrajectoryAssignment {
                    structure_id: id,
                    trajectory: valid.clone(),
                });
            }
        }
        let batch =
            StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?
                .with_gate_trajectories(assigned)?;
        fixture::expect_error(
            batch.evaluate(reference.queries[0].time),
            "Runtime",
            Some(slot),
        )?;
    }
    Ok(())
}
