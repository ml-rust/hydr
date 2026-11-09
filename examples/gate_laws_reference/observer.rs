use super::{comparison, reference_case::*, reference_inputs::*};
use hydr::structures::*;
use std::io::{BufWriter, Write};

const HEADER: &str = "case_id,reverse,expected,result,input_kind,discharge_m3_per_s,opening_m,crown_high_m,crown_low_m,submerged_height_m,submerged_height_error_bound_m,free_height_m,free_height_error_bound_m,submerged_head_gap_m,submerged_head_gap_error_bound_m,free_lower_head_m,free_lower_head_error_bound_m,free_upper_head_m,free_upper_head_error_bound_m,free_head_ratio,free_integral_factor_k,closure_residual_m,closure_uncertainty_bound_m,closure_physical_allowance_m,log_discharge_magnitude,held_volume_m3,input_width_bits,input_sill_bits,input_maximum_opening_bits,input_opening_bits,input_coefficient_bits,input_gravity_bits,input_a_kind,input_a_bits,input_b_kind,input_b_bits";

fn number(value: f64) -> String {
    format!("{value:.17e}")
}

fn optional(value: Option<f64>) -> String {
    value.map_or_else(String::new, number)
}

fn geometry(operands: Option<&StructureHydraulicOperands>) -> Result<Vec<String>> {
    let Some(operands) = operands else {
        return Ok(vec![String::new(); 19]);
    };
    let StructureHydraulicOperands::HydrostaticRectangularGate(value) = operands else {
        return Err("Gate observer receives a different hydraulic operand model".into());
    };
    let crown = value.crown_expansion_m();
    Ok(vec![
        number(value.opening_m()),
        optional(crown.map(|value| value.0)),
        optional(crown.map(|value| value.1)),
        number(value.submerged_height_m()),
        number(value.submerged_height_error_bound_m()),
        number(value.free_height_m()),
        number(value.free_height_error_bound_m()),
        optional(value.submerged_head_gap_m()),
        optional(value.submerged_head_gap_error_bound_m()),
        optional(value.free_lower_head_m()),
        optional(value.free_lower_head_error_bound_m()),
        optional(value.free_upper_head_m()),
        optional(value.free_upper_head_error_bound_m()),
        optional(value.free_head_ratio()),
        optional(value.free_integral_factor_k()),
        number(value.closure_residual_m()),
        number(value.closure_uncertainty_bound_m()),
        number(value.closure_physical_allowance_m()),
        number(value.log_discharge_magnitude()),
    ])
}

fn head_fields(support: &HeadSupport) -> [String; 2] {
    match support {
        HeadSupport::Wet { stage_m } => ["wet".into(), format!("{:016x}", stage_m.to_bits())],
        HeadSupport::Dry { bed_elevation_m } => {
            ["dry".into(), format!("{:016x}", bed_elevation_m.to_bits())]
        }
        HeadSupport::Unknown => ["unknown".into(), String::new()],
    }
}

fn input_fields(
    batch: &StructureBatch,
    input: &StructureHeads,
    operation: StructureOperatingPoint,
) -> Result<Vec<String>> {
    let element = batch
        .elements()
        .first()
        .ok_or("Missing gate observer definition")?;
    let StructureLaw::HydrostaticRectangularGate(parameters) = &element.law else {
        return Err("Gate observer receives a different structure law".into());
    };
    let StructureOperatingPoint::GateOpening { opening_m } = operation else {
        return Err("Gate observer receives a different operation".into());
    };
    let mut fields = [
        parameters.width_m(),
        parameters.sill_m(),
        parameters.maximum_opening_m(),
        opening_m,
        parameters.discharge_coefficient(),
        parameters.gravity_m_per_s2(),
    ]
    .map(|value| format!("{:016x}", value.to_bits()))
    .to_vec();
    fields.extend(head_fields(&input.endpoint_a.support));
    fields.extend(head_fields(&input.endpoint_b.support));
    Ok(fields)
}

pub fn run() -> Result {
    let mut writer = BufWriter::new(std::io::stdout().lock());
    writeln!(writer, "{HEADER}")?;
    let mut errors = Vec::new();
    for case in cases() {
        for reverse in [false, true] {
            let batch = StructureBatch::new(vec![definition(case, case.id)?])?;
            let packet = packet(case, case.id, reverse);
            let operation = operation(case, case.id);
            let mut row = vec![
                case.id.into(),
                reverse.to_string(),
                expected_label(case.expected).into(),
            ];
            match batch.propose_with_operations(
                std::slice::from_ref(&packet),
                std::slice::from_ref(&operation),
            ) {
                Ok(proposal) => {
                    let output = proposal
                        .outputs()?
                        .into_iter()
                        .next()
                        .ok_or("Missing gate observer output")?;
                    if !matches!(case.expected, ReferenceExpectation::Success) {
                        errors.push(format!(
                            "{} expects {:?}, received success",
                            case.id, case.expected
                        ));
                    }
                    if let Err(error) =
                        comparison::discharge(case, reverse, output.discharge_m3_per_s)
                    {
                        errors.push(error.to_string());
                    }
                    let held = proposal.held_volumes(HeldHeadInterval {
                        start: 0.0,
                        end: 1.0,
                    })?;
                    let volume = held
                        .outputs()?
                        .into_iter()
                        .next()
                        .ok_or("Missing held gate observer output")?
                        .volume_m3;
                    row.extend([
                        "success".into(),
                        String::new(),
                        number(output.discharge_m3_per_s),
                    ]);
                    row.extend(geometry(output.hydraulic_operands.as_ref())?);
                    row.push(number(volume));
                    row.extend(input_fields(&batch, &output.input, output.operation)?);
                }
                Err(error) => {
                    if let Err(comparison_error) = comparison::error(case, &error) {
                        errors.push(comparison_error.to_string());
                    }
                    row.extend([
                        error_label(&error).into(),
                        error
                            .input_kind()
                            .map_or_else(String::new, |kind| format!("{kind:?}")),
                    ]);
                    row.extend(vec![String::new(); 21]);
                    row.extend(input_fields(&batch, &packet, operation.operating_point)?);
                }
            }
            writeln!(writer, "{}", row.join(","))?;
        }
    }
    writer.flush()?;
    if !errors.is_empty() {
        return Err(errors.join("\n").into());
    }
    Ok(())
}
