use super::{
    receipt::{Observation, ReceiptObserver},
    reference_case::*,
    reference_inputs::{Case, Query, cases},
};
use hydr::structures::*;
use std::io::{BufWriter, Write};

const HEADER: &str = "family,case_id,layout,lane,query_index,phase,status,element,time_bits,opening_bits,error_bits,segment_index,exact_knot_index,closed_by_definition,operation_opening_bits,support_start_bits,support_end_bits,knots_time_bits,knots_opening_bits,opening_limit_bits,closing_limit_bits,maximum_opening_bits,absolute_tolerance_bits,relative_tolerance_bits,solvr_status,solvr_opening_bits,solvr_error_bits";

struct Row<'a> {
    case: &'a Case,
    width: usize,
    lane: usize,
    query: Option<(usize, &'a Query)>,
    phase: &'static str,
    status: String,
    element: Option<usize>,
    output: Option<&'a GateTrajectoryOutput>,
    operation: Option<f64>,
    support: Option<GateTrajectorySupport>,
    solvr: Option<&'a Observation>,
}

struct Failure<'a> {
    phase: &'static str,
    query: Option<(usize, &'a Query)>,
    support: Option<GateTrajectorySupport>,
    solvr: Option<&'a Observation>,
}

fn bits(value: f64) -> String {
    format!("{:016x}", value.to_bits())
}
fn optional_bits(value: Option<f64>) -> String {
    value.map_or_else(String::new, bits)
}
fn optional_index(value: Option<usize>) -> String {
    value.map_or_else(String::new, |value| value.to_string())
}

fn write_row(writer: &mut impl Write, row: Row<'_>) -> Result {
    let case = row.case;
    let output = row.output;
    let fields = [
        case.family.into(),
        case.id.into(),
        row.width.to_string(),
        row.lane.to_string(),
        optional_index(row.query.map(|query| query.0)),
        row.phase.into(),
        row.status,
        optional_index(row.element),
        optional_bits(row.query.map(|query| query.1.time)),
        optional_bits(output.map(|output| output.opening_m)),
        optional_bits(output.map(|output| output.absolute_error_m)),
        optional_index(output.map(|output| output.segment_index)),
        optional_index(output.and_then(|output| output.exact_knot_index)),
        output.map_or_else(String::new, |output| {
            output.closed_by_definition.to_string()
        }),
        optional_bits(row.operation),
        optional_bits(row.support.map(|support| support.start_s)),
        optional_bits(row.support.map(|support| support.end_s)),
        case.knots
            .iter()
            .map(|knot| bits(knot.0))
            .collect::<Vec<_>>()
            .join("|"),
        case.knots
            .iter()
            .map(|knot| bits(knot.1))
            .collect::<Vec<_>>()
            .join("|"),
        bits(case.opening_speed),
        bits(case.closing_speed),
        bits(case.maximum_opening),
        bits(case.absolute),
        bits(case.relative),
        row.solvr
            .map_or_else(String::new, |solvr| solvr.status.into()),
        optional_bits(
            row.solvr
                .and_then(|solvr| solvr.values.get(row.lane))
                .copied(),
        ),
        optional_bits(
            row.solvr
                .and_then(|solvr| solvr.errors.get(row.lane))
                .copied(),
        ),
    ];
    writeln!(writer, "{}", fields.join(","))?;
    Ok(())
}

fn failure(
    writer: &mut impl Write,
    case: &Case,
    width: usize,
    error: &StructureError,
    context: Failure<'_>,
) -> Result {
    for lane in 0..width {
        write_row(
            writer,
            Row {
                case,
                width,
                lane,
                query: context.query,
                phase: context.phase,
                status: category(error),
                element: error.element(),
                output: None,
                operation: None,
                support: context.support,
                solvr: context.solvr,
            },
        )?;
    }
    Ok(())
}

pub fn run() -> Result {
    let mut writer = BufWriter::new(std::io::stdout().lock());
    writeln!(writer, "{HEADER}")?;
    let receipts = ReceiptObserver::new();
    let mut errors = Vec::new();
    for case in cases() {
        for width in LAYOUTS {
            let definition = match trajectory(case) {
                Ok(value) => {
                    if case.static_invalid {
                        errors.push(format!(
                            "{}: static definition unexpectedly admitted",
                            case.id
                        ));
                    }
                    value
                }
                Err(error) => {
                    if !case.static_invalid {
                        errors.push(format!("{}: unexpected static rejection: {error}", case.id));
                    }
                    if let Err(error) = check_error(&error, case.binding, None) {
                        errors.push(error.to_string());
                    }
                    failure(
                        &mut writer,
                        case,
                        width,
                        &error,
                        Failure {
                            phase: "construct",
                            query: None,
                            support: None,
                            solvr: None,
                        },
                    )?;
                    continue;
                }
            };
            let batch = match structure_batch(case, width)?
                .with_gate_trajectories(assignments(&definition, width))
            {
                Ok(value) => {
                    if case.binding != "success" {
                        errors.push(format!(
                            "{}: expected binding {}, received success",
                            case.id, case.binding
                        ));
                    }
                    value
                }
                Err(error) => {
                    if let Err(error) = check_error(&error, case.binding, Some(0)) {
                        errors.push(error.to_string());
                    }
                    failure(
                        &mut writer,
                        case,
                        width,
                        &error,
                        Failure {
                            phase: "bind",
                            query: None,
                            support: None,
                            solvr: None,
                        },
                    )?;
                    continue;
                }
            };
            let support = batch.support();
            if case.queries.is_empty() {
                for lane in 0..width {
                    write_row(
                        &mut writer,
                        Row {
                            case,
                            width,
                            lane,
                            query: None,
                            phase: "bind",
                            status: "success".into(),
                            element: None,
                            output: None,
                            operation: None,
                            support: Some(support),
                            solvr: None,
                        },
                    )?;
                }
            }
            for index in 0..case.queries.len() {
                let query_index = (index + width) % case.queries.len();
                let query = &case.queries[query_index];
                let solvr = receipts.observe(case, query.time, width)?;
                match batch.evaluate(query.time) {
                    Ok(evaluation) => {
                        if query.expected != "success" {
                            errors.push(format!(
                                "{} query={query_index}: expected {}, received success",
                                case.id, query.expected
                            ));
                        }
                        let outputs = evaluation.outputs()?;
                        let operations = evaluation.operations()?;
                        if outputs.len() != width || operations.len() != width {
                            return Err("Trajectory observer projection shape differs".into());
                        }
                        for (lane, (output, operation)) in
                            outputs.iter().zip(&operations).enumerate()
                        {
                            if let Err(error) = check_output(case, query, output, lane) {
                                errors.push(error.to_string());
                            }
                            let StructureOperatingPoint::GateOpening { opening_m } =
                                operation.operating_point
                            else {
                                return Err(
                                    "Trajectory observer receives a different operation".into()
                                );
                            };
                            if operation.structure_id != output.structure_id
                                || opening_m.to_bits() != output.opening_m.to_bits()
                            {
                                errors.push(format!(
                                    "{} lane={lane}: projected operation differs",
                                    case.id
                                ));
                            }
                            write_row(
                                &mut writer,
                                Row {
                                    case,
                                    width,
                                    lane,
                                    query: Some((query_index, query)),
                                    phase: "evaluate",
                                    status: "success".into(),
                                    element: Some(output.element),
                                    output: Some(output),
                                    operation: Some(opening_m),
                                    support: Some(support),
                                    solvr: Some(&solvr),
                                },
                            )?;
                        }
                    }
                    Err(error) => {
                        let element = if matches!(query.expected, "Numerical" | "Runtime") {
                            Some(0)
                        } else {
                            None
                        };
                        if let Err(error) = check_error(&error, query.expected, element) {
                            errors.push(error.to_string());
                        }
                        failure(
                            &mut writer,
                            case,
                            width,
                            &error,
                            Failure {
                                phase: "evaluate",
                                query: Some((query_index, query)),
                                support: Some(support),
                                solvr: Some(&solvr),
                            },
                        )?;
                    }
                }
            }
        }
    }
    writer.flush()?;
    if !errors.is_empty() {
        return Err(errors.join("\n").into());
    }
    Ok(())
}
