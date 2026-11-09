use super::reference_case::{Result, error_label, expected_input_kind, expected_label};
use super::reference_inputs::{GateReferenceCase, ReferenceExpectation};
use hydr::structures::StructureError;

pub fn discharge(case: &GateReferenceCase, reverse: bool, actual: f64) -> Result {
    let reference = case
        .reference_discharge
        .ok_or("Missing successful gate reference")?;
    let expected = if reverse { -reference } else { reference };
    if !actual.is_finite() {
        return Err(format!(
            "{} reverse={reverse}: nonfinite discharge {actual}",
            case.id
        )
        .into());
    }
    if case.requires_positive
        && (actual == 0.0 || actual.is_sign_negative() != expected.is_sign_negative())
    {
        return Err(format!(
            "{} reverse={reverse}: discharge {actual:e} loses nonzero signed support",
            case.id
        )
        .into());
    }
    if expected == 0.0 {
        if actual != 0.0 {
            return Err(format!(
                "{} reverse={reverse}: physical zero becomes {actual:e}",
                case.id
            )
            .into());
        }
        return Ok(());
    }
    let tolerance = (case.relative_tolerance * expected.abs())
        .max(f64::from(case.subnormal_ulps) * f64::from_bits(1));
    if (actual - expected).abs() > tolerance {
        return Err(format!("{} reverse={reverse}: discharge={actual:e},reference={expected:e},tolerance={tolerance:e}", case.id).into());
    }
    Ok(())
}

pub fn error(case: &GateReferenceCase, error: &StructureError) -> Result {
    if error_label(error) != expected_label(case.expected) {
        return Err(format!("{} expected {:?}, got {error}", case.id, case.expected).into());
    }
    if matches!(case.expected, ReferenceExpectation::Unsupported)
        && error.input_kind() != expected_input_kind(case)
    {
        return Err(format!(
            "{} expected {:?}, got {error}",
            case.id,
            expected_input_kind(case)
        )
        .into());
    }
    Ok(())
}
