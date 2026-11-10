use super::{fixture::*, reference_case, reference_inputs::*};
use hydr::structures::*;

#[test]
fn adjacent_original_stages_select_the_physical_transition_regime() -> Result {
    for (id, expected) in [
        ("donor_sill_below", StructureRegime::GateBelowSill),
        ("donor_sill_equal", StructureRegime::GateBelowSill),
        ("donor_crown_below", StructureRegime::FreeGateAperture),
        ("donor_crown_equal", StructureRegime::FreeGateAperture),
        ("donor_crown_above", StructureRegime::FreeGateAperture),
        ("receiver_sill_below", StructureRegime::FreeGateAperture),
        ("receiver_sill_equal", StructureRegime::FreeGateAperture),
        ("receiver_sill_above", StructureRegime::MixedGateAperture),
        ("receiver_crown_below", StructureRegime::MixedGateAperture),
        (
            "receiver_crown_equal",
            StructureRegime::SubmergedGateAperture,
        ),
        (
            "receiver_crown_above",
            StructureRegime::SubmergedGateAperture,
        ),
    ] {
        let case = cases()
            .find(|case| case.id == id)
            .ok_or("Missing gate transition input")?;
        for reverse in [false, true] {
            let batch = StructureBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                vec![reference_case::definition(case, id)?],
            )?;
            let output = first(&batch.propose_with_operations(
                &[reference_case::packet(case, id, reverse)],
                &[reference_case::operation(case, id)],
            )?)?;
            assert_eq!(output.regime, expected, "{id} reverse={reverse}");
        }
    }
    Ok(())
}

#[test]
fn opening_sweeps_preserve_monotonicity_and_have_no_traversal_history() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("gate", DirectionPolicy::Bidirectional)?],
    )?;
    for packets in [
        heads("gate", 2.0, 0.5),
        packet(
            "gate",
            HeadSupport::Wet { stage_m: 0.25 },
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
        ),
    ] {
        let mut forward = Vec::new();
        let mut previous = 0.0_f64;
        for index in 0..=32 {
            let value = f64::from(index) / 32.0;
            let proposal = batch.propose_with_operations(
                std::slice::from_ref(&packets),
                &[opening("gate", value)],
            )?;
            let output = first(&proposal)?;
            let tolerance =
                128.0 * f64::EPSILON * previous.abs().max(output.discharge_m3_per_s.abs());
            assert!(previous <= output.discharge_m3_per_s + tolerance);
            previous = output.discharge_m3_per_s;
            forward.push(output);
        }
        for (index, expected) in forward.iter().enumerate().rev() {
            let value = index as f64 / 32.0;
            let output = first(&batch.propose_with_operations(
                std::slice::from_ref(&packets),
                &[opening("gate", value)],
            )?)?;
            assert_eq!(&output, expected);
        }
    }
    Ok(())
}

#[test]
fn submerged_apertures_match_the_existing_orifice_law() -> Result {
    for (sill, opening_m, width, donor, receiver) in
        [(0.0, 0.75, 2.0, 2.0, 1.0), (-1e16, 2.0, 1.0, 1.0, 0.5)]
    {
        let gate = element(
            "gate",
            StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                width,
                sill,
                opening_m,
                0.6,
                0.5,
                applicability(),
            )?),
            DirectionPolicy::Bidirectional,
        );
        let orifice = element(
            "orifice",
            StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
                width * opening_m,
                sill,
                sill + opening_m,
                0.6,
                0.5,
                OrificeApplicability {
                    coefficient_provenance: "Analytical overlap fixture".into(),
                    fully_submerged_applicability: "Both original stages exceed the aperture crown"
                        .into(),
                    symmetric_reversal: true,
                },
            )?),
            DirectionPolicy::Bidirectional,
        );
        let batch = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![gate, orifice],
        )?;
        for (a, b) in [(donor, receiver), (receiver, donor)] {
            let output = batch
                .propose_with_operations(
                    &[heads("gate", a, b), heads("orifice", a, b)],
                    &[
                        opening("gate", opening_m),
                        operation("orifice", StructureOperatingPoint::Passive),
                    ],
                )?
                .outputs()?;
            let gate_q = output
                .first()
                .ok_or("Missing gate overlap output")?
                .discharge_m3_per_s;
            let orifice_q = output
                .get(1)
                .ok_or("Missing orifice overlap output")?
                .discharge_m3_per_s;
            assert!((gate_q - orifice_q).abs() <= 128.0 * f64::EPSILON * orifice_q.abs());
        }
    }
    Ok(())
}
