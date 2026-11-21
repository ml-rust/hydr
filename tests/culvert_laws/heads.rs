//! Direction, reversal, equal heads, dry ends and the zero-discharge headwater.

use super::support::{Culvert, TestResult, client, dry, elements, heads, propose, wet};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, FlowDirection, StructureBatch, StructureError,
    StructureInputErrorKind, StructureRegime, StructureStatus,
};
use numr::runtime::cpu::CpuRuntime;

fn circle() -> TestResult<CulvertBarrel> {
    Ok(CulvertBarrel::circular(1.2)?)
}

fn level(code_a: u8, code_b: u8) -> TestResult<Culvert> {
    let mut culvert = Culvert::both(circle()?, [0.0, 0.0], CulvertInlet::hds5(code_a)?, 0.5);
    culvert.end_b = Some((CulvertInlet::hds5(code_b)?, 0.2));
    Ok(culvert)
}

#[test]
fn reversed_heads_negate_the_discharge_with_the_b_end_inlet() -> TestResult {
    let forward = Culvert::both(circle()?, [0.0, 0.0], CulvertInlet::hds5(1)?, 0.5);
    let mut tilted = level(1, 3)?;
    tilted.invert_a_m = 0.1;
    // The same barrel entered at A through the B end of `tilted`, with the inverts swapped.
    let mut mirror = level(3, 1)?;
    mirror.invert_b_m = 0.1;
    mirror.end_a = tilted.end_b;
    mirror.end_b = tilted.end_a;
    let culverts = [forward, forward, tilted, mirror];
    let supports = [
        (wet(1.0), dry(0.0)),
        (dry(0.0), wet(1.0)),
        (wet(0.05), wet(1.3)),
        (wet(1.3), wet(0.05)),
    ];
    let outputs = propose(&elements(&culverts)?, &supports)?;
    let q: Vec<f64> = outputs.iter().map(|o| o.discharge_m3_per_s).collect();
    assert!(q[0] > 0.0, "{outputs:?}");
    assert_eq!(q[1].to_bits(), (-q[0]).to_bits());
    assert_eq!(outputs[1].proposed_direction, Some(FlowDirection::BtoA));
    assert!(q[2] < 0.0 && q[3] > 0.0, "{q:?}");
    assert_eq!(q[2].to_bits(), (-q[3]).to_bits());
    Ok(())
}

#[test]
fn one_end_culvert_blocks_reversed_heads_and_zero_classes_carry_no_water() -> TestResult {
    let mut flap = level(1, 1)?;
    flap.end_b = None;
    let both = level(1, 1)?;
    let culverts = [flap, both, both, both, both];
    let supports = [
        (wet(0.2), wet(1.0)),
        (wet(0.8), wet(0.8)),
        (wet(0.0), dry(0.0)),
        (dry(-0.5), dry(-0.5)),
        (wet(-0.3), wet(-0.1)),
    ];
    let outputs = propose(&elements(&culverts)?, &supports)?;
    let expected = [
        (
            StructureStatus::FlapBlocked,
            StructureRegime::CulvertUndeclaredEnd,
            Some(FlowDirection::BtoA),
        ),
        (
            StructureStatus::PhysicalZero,
            StructureRegime::EqualHeads,
            None,
        ),
        (
            StructureStatus::PhysicalZero,
            StructureRegime::CulvertBelowInvert,
            None,
        ),
        (
            StructureStatus::PhysicalZero,
            StructureRegime::CulvertBelowInvert,
            None,
        ),
        (
            StructureStatus::PhysicalZero,
            StructureRegime::CulvertBelowInvert,
            None,
        ),
    ];
    for (index, (output, (status, regime, direction))) in outputs.iter().zip(expected).enumerate() {
        assert_eq!(
            output.discharge_m3_per_s.to_bits(),
            0.0f64.to_bits(),
            "{index}"
        );
        assert_eq!(
            (output.status, output.regime, output.proposed_direction),
            (status, regime, direction),
            "lane {index}: {output:?}"
        );
        assert!(output.hydraulic_operands.is_none(), "{index}");
    }
    Ok(())
}

#[test]
fn a_dry_receiver_is_a_free_outfall_at_its_invert() -> TestResult {
    let culvert = level(1, 1)?;
    let culverts = [culvert; 4];
    let supports = [
        (wet(0.9), dry(0.0)),
        (wet(0.9), dry(-2.0)),
        (wet(0.9), wet(0.0)),
        (wet(0.9), wet(-0.4)),
    ];
    let outputs = propose(&elements(&culverts)?, &supports)?;
    let first = outputs[0].discharge_m3_per_s;
    assert!(first > 0.0);
    for output in &outputs {
        assert_eq!(
            output.discharge_m3_per_s.to_bits(),
            first.to_bits(),
            "{output:?}"
        );
    }
    Ok(())
}

#[test]
fn a_dry_end_above_its_invert_rejects() -> TestResult {
    let culvert = level(1, 1)?;
    let elements = elements(&[culvert])?;
    let batch = StructureBatch::<CpuRuntime>::with_culverts(client(), elements.clone())?;
    let packets = [heads(&elements[0], wet(1.0), dry(0.2))];
    match batch.propose(&packets) {
        Err(StructureError::InvalidInput {
            kind: StructureInputErrorKind::UnsupportedRegime,
            context,
            element: Some(0),
        }) => assert!(context.contains("bed at or below its invert"), "{context}"),
        other => return Err(format!("expected a rejection, found {other:?}").into()),
    }
    Ok(())
}

#[test]
fn inlet_admission_names_the_culvert_element_among_other_laws() -> TestResult {
    use hydr::structures::CulvertInletForm;
    let good = level(1, 1)?;
    let mut falling = good;
    let inlet = CulvertInlet::explicit(CulvertInletForm::Two, [1.0, 1.0, 0.01, 0.0], false)?;
    falling.end_a = Some((inlet, 0.5));
    falling.end_b = None;
    let elements = elements(&[good, falling])?;
    match StructureBatch::<CpuRuntime>::with_culverts(client(), elements) {
        Err(StructureError::InvalidInput {
            kind: StructureInputErrorKind::Definition,
            context,
            element: Some(1),
        }) => assert!(context.contains("falls across the transition"), "{context}"),
        other => return Err(format!("expected element 1 to reject, found {other:?}").into()),
    }
    Ok(())
}

#[test]
fn heads_at_or_below_the_zero_discharge_headwater_give_exact_zero() -> TestResult {
    // An adverse form-1 barrel starts its inlet curve at D Ks S = 1.2 x 0.005 m.
    let mut adverse = Culvert::both(circle()?, [0.0, 0.3], CulvertInlet::hds5(1)?, 0.5);
    adverse.end_b = None;
    let culverts = [adverse; 4];
    // Water below the outlet invert, 0.3 m above the inlet invert, cannot leave the barrel.
    let supports = [
        (wet(0.003), dry(0.3)),
        (wet(0.005), dry(0.3)),
        (wet(0.25), dry(0.3)),
        (wet(0.5), dry(0.3)),
    ];
    let outputs = propose(&elements(&culverts)?, &supports)?;
    let zero = [
        StructureRegime::CulvertInletControl,
        StructureRegime::CulvertInletControl,
        StructureRegime::CulvertOutletControl,
    ];
    for (output, regime) in outputs.iter().zip(zero) {
        assert_eq!(output.discharge_m3_per_s.to_bits(), 0.0f64.to_bits());
        assert_eq!(output.status, StructureStatus::PhysicalZero);
        assert_eq!(output.regime, regime, "{output:?}");
    }
    assert!(outputs[3].discharge_m3_per_s > 0.0, "{:?}", outputs[3]);
    Ok(())
}
