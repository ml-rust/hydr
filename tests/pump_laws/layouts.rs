use super::{
    fixture::*,
    references::{cases, check_output},
};
use hydr::structures::*;

fn passive(id: &str) -> Result<StructureElement> {
    Ok(element(
        id,
        StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            2.0,
            10.0,
            1.7,
            WeirApplicability {
                coefficient_provenance: "Independent analytical SI fixture".into(),
                free_nappe_and_approach_assertion: "Suppressed sharp-crested analytical fixture"
                    .into(),
                submergence_correction_assertion: "Adopted analytical correction".into(),
                symmetric_reversal: true,
            },
        )?),
        DirectionPolicy::Bidirectional,
    ))
}

#[test]
fn mixed_passive_and_pump_layouts_preserve_all_admitted_references() -> Result {
    let cases: Vec<_> = cases()
        .into_iter()
        .filter(|case| matches!(case.expected, "active" | "shutoff"))
        .collect();
    assert_eq!(cases.len(), 17);
    for count in [1, 31, 32, 33, 64, 65] {
        for rotation in 0..cases.len() + 2 {
            let mut definitions = Vec::new();
            let mut packets = Vec::new();
            let mut operations = Vec::new();
            for lane in 0..count {
                let slot = (lane + rotation) % (cases.len() + 2);
                let id = format!("structure-{lane}");
                if let Some(case) = cases.get(slot) {
                    let reverse = lane % 2 == 1;
                    definitions.push(definition(case, &id, reverse)?);
                    packets.push(case_heads(case, &id, reverse));
                    operations.push(running(&id, case.speed));
                } else if slot == cases.len() {
                    definitions.push(passive(&id)?);
                    packets.push(heads(&id, 11.0, 9.0));
                    operations.push(operation(&id, StructureOperatingPoint::Passive));
                } else {
                    definitions.push(standard(&id)?);
                    packets.push(packet(
                        &id,
                        HeadSupport::Unknown,
                        HeadSupport::Dry {
                            bed_elevation_m: 50.0,
                        },
                    ));
                    operations.push(operation(&id, StructureOperatingPoint::PumpStoppedIsolated));
                }
            }
            let batch = StructureBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                definitions,
            )?;
            let outputs = batch
                .propose_with_operations(&packets, &operations)?
                .outputs()?;
            assert_eq!(outputs.len(), count);
            for (lane, output) in outputs.into_iter().enumerate() {
                assert_eq!(output.element, lane);
                assert_eq!(output.input, packets[lane]);
                let slot = (lane + rotation) % (cases.len() + 2);
                if let Some(case) = cases.get(slot) {
                    check_output(case, &output, lane % 2 == 1)?;
                } else if slot == cases.len() {
                    let id = format!("structure-{lane}");
                    let single = StructureBatch::<crate::runtime::Runtime>::new(
                        crate::runtime::client(),
                        vec![passive(&id)?],
                    )?;
                    let expected = first(&single.propose(&[heads(&id, 11.0, 9.0)])?)?;
                    assert_eq!(
                        output.discharge_m3_per_s.to_bits(),
                        expected.discharge_m3_per_s.to_bits()
                    );
                    assert_eq!(output.status, StructureStatus::Active);
                    assert_eq!(output.operation, StructureOperatingPoint::Passive);
                } else {
                    assert_eq!(output.status, StructureStatus::PumpStoppedIsolated);
                    assert_eq!(output.discharge_m3_per_s, 0.0);
                    assert!(output.hydraulic_operands.is_none());
                }
            }
        }
    }
    Ok(())
}
