use super::{comparison, reference_case, reference_inputs::*};
use hydr::structures::*;
use reference_case::Result;

pub(super) fn check_output(
    case: &GateReferenceCase,
    reverse: bool,
    packet: &StructureHeads,
    output: &StructureOutput,
) -> Result {
    comparison::discharge(case, reverse, output.discharge_m3_per_s)?;
    assert_eq!(output.input, *packet);
    assert_eq!(
        output.operation,
        StructureOperatingPoint::GateOpening {
            opening_m: case.opening_m
        }
    );
    assert_eq!(output.law, StructureLawKind::HydrostaticRectangularGate);
    if output.discharge_m3_per_s == 0.0 {
        assert_eq!(output.status, StructureStatus::PhysicalZero);
        assert!(output.hydraulic_operands.is_none());
        assert!(output.proposed_direction.is_none());
        return Ok(());
    }
    assert_eq!(output.status, StructureStatus::Active);
    assert_eq!(
        output.proposed_direction,
        Some(if output.discharge_m3_per_s > 0.0 {
            FlowDirection::AtoB
        } else {
            FlowDirection::BtoA
        })
    );
    let Some(StructureHydraulicOperands::HydrostaticRectangularGate(operands)) =
        &output.hydraulic_operands
    else {
        return Err("Missing active gate geometry".into());
    };
    assert_eq!(operands.opening_m().to_bits(), case.opening_m.to_bits());
    for value in [
        operands.submerged_height_m(),
        operands.submerged_height_error_bound_m(),
        operands.free_height_m(),
        operands.free_height_error_bound_m(),
        operands.closure_uncertainty_bound_m(),
        operands.closure_physical_allowance_m(),
    ] {
        assert!(
            value.is_finite() && value >= 0.0,
            "{} geometry={value:e}",
            case.id
        );
    }
    assert!(operands.closure_residual_m().is_finite());
    assert!(operands.log_discharge_magnitude().is_finite());
    if operands.submerged_height_m() == 0.0 {
        assert!(operands.submerged_head_gap_m().is_none());
        assert!(operands.submerged_head_gap_error_bound_m().is_none());
    } else {
        assert!(
            operands
                .submerged_head_gap_m()
                .is_some_and(|value| value.is_finite() && value > 0.0)
        );
        assert!(
            operands
                .submerged_head_gap_error_bound_m()
                .is_some_and(|value| value.is_finite() && value >= 0.0)
        );
    }
    let free_values = [
        operands.free_lower_head_m(),
        operands.free_lower_head_error_bound_m(),
        operands.free_upper_head_m(),
        operands.free_upper_head_error_bound_m(),
        operands.free_head_ratio(),
        operands.free_integral_factor_k(),
    ];
    if operands.free_height_m() == 0.0 {
        assert!(free_values.iter().all(Option::is_none));
    } else {
        assert!(
            free_values
                .iter()
                .all(|value| value.is_some_and(f64::is_finite))
        );
        assert!(
            operands
                .free_lower_head_m()
                .is_some_and(|value| value > 0.0)
        );
        assert!(
            operands
                .free_upper_head_m()
                .is_some_and(|value| value >= 0.0)
        );
        assert!(
            operands
                .free_head_ratio()
                .is_some_and(|value| (0.0..=1.0).contains(&value))
        );
        assert!(
            operands
                .free_integral_factor_k()
                .is_some_and(|value| (1.0..=1.5).contains(&value))
        );
    }
    match output.regime {
        StructureRegime::FreeGateAperture => assert_eq!(operands.submerged_height_m(), 0.0),
        StructureRegime::MixedGateAperture => {
            assert!(operands.submerged_height_m() > 0.0 && operands.free_height_m() > 0.0);
        }
        StructureRegime::SubmergedGateAperture => assert_eq!(operands.free_height_m(), 0.0),
        _ => {
            return Err(format!("{} unexpected active regime {:?}", case.id, output.regime).into());
        }
    }
    Ok(())
}

#[test]
fn original_inputs_match_every_independent_discharge_and_rejection_category() -> Result {
    assert_eq!(cases().count(), 40);
    for case in cases() {
        for reverse in [false, true] {
            let batch = StructureBatch::new(vec![reference_case::definition(case, case.id)?])?;
            let packet = reference_case::packet(case, case.id, reverse);
            match batch.propose_with_operations(
                std::slice::from_ref(&packet),
                &[reference_case::operation(case, case.id)],
            ) {
                Ok(proposal) => {
                    assert!(
                        matches!(case.expected, ReferenceExpectation::Success),
                        "{}",
                        case.id
                    );
                    let output = super::fixture::first(&proposal)?;
                    check_output(case, reverse, &packet, &output)?;
                }
                Err(error) => comparison::error(case, &error)?,
            }
        }
    }
    Ok(())
}
