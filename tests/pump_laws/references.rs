use super::fixture::*;
use hydr::structures::*;

pub struct Case {
    pub id: &'static str,
    pub donor: f64,
    pub receiver: f64,
    pub speed: f64,
    pub heads: Vec<f64>,
    pub flows: Vec<f64>,
    pub expected: &'static str,
    pub exact: bool,
    pub reference: Option<f64>,
    pub high: f64,
    pub low: f64,
    pub reference_only: bool,
}

pub fn cases() -> Vec<Case> {
    let mut rows = super::reference_inputs::rows();
    rows.extend(super::reference_limits::rows());
    rows
}

pub fn check_output(case: &Case, output: &StructureOutput, reverse: bool) -> Result {
    let q = if reverse {
        -output.discharge_m3_per_s
    } else {
        output.discharge_m3_per_s
    };
    assert_eq!(
        output.operation,
        StructureOperatingPoint::PumpRunning {
            speed_ratio: case.speed
        }
    );
    assert_eq!(output.regime, StructureRegime::PumpCurve);
    let reference = case.reference.ok_or("Missing admitted reference")?;
    if case.expected == "shutoff" {
        assert_eq!(q, 0.0);
        assert_eq!(output.status, StructureStatus::PumpShutoff);
        assert_eq!(output.proposed_direction, None);
    } else {
        assert!(q.is_finite() && q > 0.0, "{}: {q}", case.id);
        assert_eq!(output.status, StructureStatus::PumpRunning);
        assert_eq!(
            output.proposed_direction,
            Some(if reverse {
                FlowDirection::BtoA
            } else {
                FlowDirection::AtoB
            })
        );
        if case.exact {
            assert_eq!(q.to_bits(), reference.to_bits(), "{}", case.id);
        } else {
            let tolerance = 64.0 * f64::EPSILON * reference.abs().max(q.abs());
            assert!(
                (q - reference).abs() <= tolerance,
                "{}: {q:e}, reference {reference:e}, tolerance {tolerance:e}",
                case.id
            );
        }
    }
    match &output.hydraulic_operands {
        Some(StructureHydraulicOperands::Pump(operands)) => {
            assert_eq!(
                operands.receiving_minus_donor_head_m().to_bits(),
                case.high.to_bits(),
                "{}",
                case.id
            );
            assert_eq!(
                operands.receiving_minus_donor_head_low_m().to_bits(),
                case.low.to_bits(),
                "{}",
                case.id
            );
            assert_eq!(operands.speed_ratio().to_bits(), case.speed.to_bits());
            assert!(operands.normalization_error_bound_m().is_finite());
            assert!(operands.reference_head_lower_m() <= operands.reference_head_m());
            assert!(operands.reference_head_upper_m() >= operands.reference_head_m());
            if case.low != 0.0 {
                assert!(operands.normalization_error_bound_m() >= case.low.abs());
                assert!(operands.reference_head_upper_m() > operands.reference_head_m());
                assert!(operands.reference_head_lower_m() < operands.reference_head_m());
            }
        }
        other => return Err(format!("Missing pump operands: {other:?}").into()),
    }
    Ok(())
}

#[test]
fn exact_rational_corpus_preserves_discharge_domain_and_subtraction_witnesses() -> Result {
    let rows = cases();
    assert_eq!(rows.len(), 26);
    for case in rows {
        for reverse in [false, true] {
            let definition = definition(&case, case.id, reverse)?;
            let batch = StructureBatch::new(vec![definition])?;
            let isolated = batch.propose_with_operations(
                &[packet(case.id, HeadSupport::Unknown, HeadSupport::Unknown)],
                &[operation(
                    case.id,
                    StructureOperatingPoint::PumpStoppedIsolated,
                )],
            )?;
            let isolated_before = isolated.outputs()?;
            let packet = case_heads(&case, case.id, reverse);
            let result = batch.propose_with_operations(&[packet], &[running(case.id, case.speed)]);
            match case.expected {
                "active" | "shutoff" => check_output(&case, &first(&result?)?, reverse)?,
                "numerical" => {
                    let error = rejection(result)?;
                    assert!(
                        matches!(error, StructureError::Numerical { .. }),
                        "{}: {error}",
                        case.id
                    );
                    assert_eq!(error.element(), Some(0));
                    let expected_context = match case.id {
                        "positive_discharge_underflow" | "positive_discharge_overflow" => {
                            "active final pump discharge overflows or underflows to zero"
                        }
                        "nonbinary_shutoff_ambiguity" => {
                            "pump normalization interval spans positive and zero-flow regions"
                        }
                        _ => "pump normalized-head interval overlaps curve endpoint",
                    };
                    assert!(
                        error.to_string().contains(expected_context),
                        "{}: {error}",
                        case.id
                    );
                }
                "unsupported_regime" => {
                    let error = rejection(result)?;
                    assert_eq!(
                        error.input_kind(),
                        Some(StructureInputErrorKind::UnsupportedRegime),
                        "{}: {error}",
                        case.id
                    );
                    assert_eq!(error.element(), Some(0));
                }
                other => return Err(format!("Unknown frozen expectation: {other}").into()),
            }
            assert_eq!(isolated.outputs()?, isolated_before);
        }
    }
    Ok(())
}
