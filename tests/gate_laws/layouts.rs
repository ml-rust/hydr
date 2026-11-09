use super::{comparison, fixture, reference_case, reference_inputs::*, references};
use hydr::structures::*;
use reference_case::Result;

#[test]
fn every_reference_rotates_through_mixed_laws_across_cpu_layouts() -> Result {
    let successful = cases()
        .filter(|case| matches!(case.expected, ReferenceExpectation::Success))
        .collect::<Vec<_>>();
    for width in [1, 31, 32, 33, 64, 65] {
        for reverse in [false, true] {
            for (index, selected) in cases().enumerate() {
                let slot = index % width;
                let mut definitions = Vec::with_capacity(width);
                let mut packets = Vec::with_capacity(width);
                let mut operations = Vec::with_capacity(width);
                let mut lane_cases = Vec::with_capacity(width);
                for lane in 0..width {
                    let id = format!("lane-{lane}");
                    if lane != slot && lane % 3 == 0 {
                        definitions.push(fixture::element(
                            &id,
                            fixture::passive_law()?,
                            DirectionPolicy::Bidirectional,
                        ));
                        packets.push(fixture::heads(&id, 0.0, 0.0));
                        operations.push(fixture::operation(&id, StructureOperatingPoint::Passive));
                        lane_cases.push(None);
                    } else if lane != slot && lane % 3 == 1 {
                        definitions.push(fixture::element(
                            &id,
                            fixture::pump_law()?,
                            DirectionPolicy::AtoBOnly,
                        ));
                        packets.push(fixture::packet(
                            &id,
                            HeadSupport::Unknown,
                            HeadSupport::Unknown,
                        ));
                        operations.push(fixture::operation(
                            &id,
                            StructureOperatingPoint::PumpStoppedIsolated,
                        ));
                        lane_cases.push(None);
                    } else {
                        let case = if lane == slot {
                            selected
                        } else {
                            successful[(lane + index) % successful.len()]
                        };
                        definitions.push(reference_case::definition(case, &id)?);
                        packets.push(reference_case::packet(case, &id, reverse));
                        operations.push(reference_case::operation(case, &id));
                        lane_cases.push(Some(case));
                    }
                }
                let batch = StructureBatch::new(definitions)?;
                match batch.propose_with_operations(&packets, &operations) {
                    Ok(proposal) => {
                        assert!(
                            matches!(selected.expected, ReferenceExpectation::Success),
                            "{}",
                            selected.id
                        );
                        let outputs = proposal.outputs()?;
                        assert_eq!(outputs.len(), width);
                        for ((case, output), packet) in
                            lane_cases.iter().zip(&outputs).zip(&packets)
                        {
                            if let Some(case) = case {
                                references::check_output(case, reverse, packet, output)?;
                            } else {
                                assert_eq!(output.discharge_m3_per_s, 0.0);
                                assert!(output.hydraulic_operands.is_none());
                                assert_eq!(output.input, *packet);
                            }
                        }
                    }
                    Err(error) => {
                        comparison::error(selected, &error)?;
                        assert_eq!(error.element(), Some(slot), "{} width={width}", selected.id);
                    }
                }
            }
        }
    }
    Ok(())
}
