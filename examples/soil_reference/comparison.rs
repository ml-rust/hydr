//! Frozen source-conformance tolerances applied through the public component.

use super::records::Record;
use hydr::soil::{
    ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilInput, NativeSoilState,
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn close(case: &str, field: &str, actual: f64, expected: f64, absolute: f64) -> Result<()> {
    let tolerance = absolute + 1e-11 * actual.abs().max(expected.abs());
    if !actual.is_finite() || (actual - expected).abs() > tolerance {
        return Err(format!("case={case} field={field} actual={actual:.17e} expected={expected:.17e} tolerance={tolerance:.17e}").into());
    }
    Ok(())
}

fn state(case: &str, actual: NativeSoilState, expected: NativeSoilState) -> Result<()> {
    close(case, "IMD", actual.imd, expected.imd, 1e-12)?;
    close(case, "F", actual.f_ft, expected.f_ft, 1e-12)?;
    close(case, "Fu", actual.fu_ft, expected.fu_ft, 1e-12)?;
    close(case, "T", actual.t_s, expected.t_s, 1e-9)?;
    if actual.saturated != expected.saturated {
        return Err(format!("case={case}: native Sat differs").into());
    }
    Ok(())
}

fn one_state(batch: &ModifiedGreenAmptBatch) -> Result<NativeSoilState> {
    batch
        .native_states()?
        .first()
        .copied()
        .ok_or_else(|| "missing soil cell".into())
}

fn direct_root(row: &Record) -> Result<()> {
    let f1 = row.value(21)?;
    let c1 = row.value(22)?;
    let ks = row.value(23)?;
    let dt = row.value(24)?;
    let parameters = ModifiedGreenAmptParameters::from_native(c1, ks, 1.0)?;
    let initial = NativeSoilState {
        imd: 1.0,
        f_ft: f1,
        fu_ft: 0.0,
        saturated: true,
        t_s: 0.0,
    };
    let batch = ModifiedGreenAmptBatch::from_native_states(&[parameters], &[initial])?;
    let trial = batch.trial_native(
        dt,
        &[NativeSoilInput {
            incoming_rate_ft_per_s: 1e6,
            ponded_depth_ft: 0.0,
        }],
    )?;
    let expected_f2 = row.value(28)?;
    let expected_df = expected_f2 - f1;
    let output = trial
        .outputs()?
        .into_iter()
        .next()
        .ok_or("missing root interval output")?;
    close(
        &row.case,
        "F2 through saturated state",
        output.native_rate_ft_per_s,
        expected_df / dt,
        1e-14,
    )?;
    let next = trial.into_next();
    let expected = NativeSoilState {
        f_ft: f1 + expected_df,
        fu_ft: expected_df.min(parameters.capacity_ft()),
        t_s: 5400.0 / parameters.upper_zone_depth_ft(),
        ..initial
    };
    state(&row.case, one_state(&next)?, expected)
}

pub fn compare(records: &[Record]) -> Result<()> {
    let mut batch = None;
    let mut checked_initializations = 0;
    let mut checked_sequences = 0;
    let mut checked_steps = 0;
    let mut checked_roots = 0;
    let mut accumulated_depth = 0.0;
    let mut case_has_step = false;
    let mut unsupported = Vec::new();
    for row in records {
        if row.model == 2 {
            if row.index == 1 {
                unsupported.push(format!("{}: original model 2", row.case));
            }
            continue;
        }
        if row.index == 1 {
            let parameters = ModifiedGreenAmptParameters::from_native(
                row.value(5)?,
                row.value(6)?,
                row.value(7)?,
            )?;
            let initialized = if row.initial_mode == "INIT" {
                ModifiedGreenAmptBatch::initialized(&[parameters])?
            } else {
                ModifiedGreenAmptBatch::from_native_states(&[parameters], &[row.state(8)?])?
            };
            let actual = one_state(&initialized)?;
            state(&row.case, actual, row.state(8)?)?;
            state(&row.case, actual, row.state(30)?)?;
            batch = Some(initialized);
            accumulated_depth = 0.0;
            case_has_step = false;
            checked_initializations += 1;
        } else {
            let current = batch
                .as_ref()
                .ok_or("numerical command lacks initialized batch")?;
            state(&row.case, one_state(current)?, row.state(13)?)?;
            match row.operation.as_str() {
                "STEP" => {
                    let trial = current.trial_native(
                        row.value(18)?,
                        &[NativeSoilInput {
                            incoming_rate_ft_per_s: row.value(19)?,
                            ponded_depth_ft: row.value(20)?,
                        }],
                    )?;
                    let output = trial
                        .outputs()?
                        .into_iter()
                        .next()
                        .ok_or("missing interval output")?;
                    close(
                        &row.case,
                        "rate",
                        output.native_rate_ft_per_s,
                        row.value(25)?,
                        1e-14,
                    )?;
                    close(
                        &row.case,
                        "interval depth",
                        output.native_depth_ft,
                        row.value(26)?,
                        1e-12,
                    )?;
                    accumulated_depth += output.native_depth_ft;
                    let next = trial.into_next();
                    let actual = one_state(&next)?;
                    state(&row.case, actual, row.state(30)?)?;
                    // Names label exact witnesses only. Operation metadata controls numerical coverage.
                    if row.case.starts_with("dry_timer_")
                        && (actual.f_ft == 0.0) != (row.state(30)?.f_ft == 0.0)
                    {
                        return Err(format!("case={}: timer reset branch differs", row.case).into());
                    }
                    if (row.case.starts_with("zero_") || row.case.starts_with("ponding_zero_"))
                        && (output.native_rate_ft_per_s == 0.0) != (row.value(25)? == 0.0)
                    {
                        return Err(
                            format!("case={}: effective-ZERO branch differs", row.case).into()
                        );
                    }
                    batch = Some(next);
                    checked_steps += 1;
                    if !case_has_step {
                        checked_sequences += 1;
                        case_has_step = true;
                    }
                }
                "F2" => {
                    if row.value(23)? == 0.0 {
                        unsupported.push(format!(
                            "{} index={}: zero-Ks F2 helper",
                            row.case, row.index
                        ));
                    } else {
                        direct_root(row)?;
                        checked_roots += 1;
                    }
                    // A direct F2 probe never advances the case's initialized soil state.
                    state(&row.case, one_state(current)?, row.state(30)?)?;
                }
                _ => {
                    return Err(format!(
                        "case={} index={}: unexpected continuation operation",
                        row.case, row.index
                    )
                    .into());
                }
            }
        }
        let current = batch.as_ref().ok_or("missing checked soil context")?;
        close(
            &row.case,
            "Lu",
            current.parameters()[0].upper_zone_depth_ft(),
            row.value(29)?,
            1e-12,
        )?;
        close(
            &row.case,
            "accumulated interval infiltration",
            accumulated_depth,
            row.value(27)?,
            1e-12,
        )?;
    }
    if checked_steps + checked_roots == 0 {
        return Err("native CSV contains no supported STEP or F2 numerical comparison".into());
    }
    println!(
        "Checked {checked_initializations} modified initializations, {checked_sequences} STEP sequences, {checked_steps} STEP calls, and {checked_roots} F2 probes through saturated public states."
    );
    println!(
        "Unsupported original-model cases/zero-Ks helper calls ({}): {}",
        unsupported.len(),
        unsupported.join(", ")
    );
    compare_batches(records)?;
    Ok(())
}

struct Probe {
    case: String,
    parameters: ModifiedGreenAmptParameters,
    initial: NativeSoilState,
    input: NativeSoilInput,
    expected: NativeSoilState,
    rate: f64,
    dt: f64,
}

fn compare_batches(records: &[Record]) -> Result<()> {
    let mut groups = std::collections::BTreeMap::<u64, Vec<Probe>>::new();
    for row in records
        .iter()
        .filter(|r| r.model == 3 && r.operation == "STEP")
    {
        let dt = row.value(18)?;
        groups.entry(dt.to_bits()).or_default().push(Probe {
            case: row.case.clone(),
            parameters: ModifiedGreenAmptParameters::from_native(
                row.value(5)?,
                row.value(6)?,
                row.value(7)?,
            )?,
            initial: row.state(13)?,
            expected: row.state(30)?,
            dt,
            rate: row.value(25)?,
            input: NativeSoilInput {
                incoming_rate_ft_per_s: row.value(19)?,
                ponded_depth_ft: row.value(20)?,
            },
        });
    }
    for row in records
        .iter()
        .filter(|r| r.model == 3 && r.operation == "F2")
    {
        let (f1, c1, ks, dt) = (
            row.value(21)?,
            row.value(22)?,
            row.value(23)?,
            row.value(24)?,
        );
        if ks == 0.0 {
            continue;
        }
        let parameters = ModifiedGreenAmptParameters::from_native(c1, ks, 1.0)?;
        let initial = NativeSoilState {
            imd: 1.0,
            f_ft: f1,
            fu_ft: 0.0,
            saturated: true,
            t_s: 0.0,
        };
        let df = row.value(28)? - f1;
        groups.entry(dt.to_bits()).or_default().push(Probe {
            case: row.case.clone(),
            parameters,
            initial,
            dt,
            input: NativeSoilInput {
                incoming_rate_ft_per_s: 1e6,
                ponded_depth_ft: 0.0,
            },
            rate: df / dt,
            expected: NativeSoilState {
                f_ft: f1 + df,
                fu_ft: df.min(parameters.capacity_ft()),
                t_s: 5400.0 / parameters.upper_zone_depth_ft(),
                ..initial
            },
        });
    }
    let mut checked = 0;
    for probes in groups.values() {
        for length in [31, 32, 33, 64, 65] {
            // Rotations put every branch probe in vectorized and scalar tail positions.
            for offset in 0..probes.len() {
                let lanes: Vec<_> = (0..length)
                    .map(|i| &probes[(i + offset) % probes.len()])
                    .collect();
                let parameters: Vec<_> = lanes.iter().map(|p| p.parameters).collect();
                let initial: Vec<_> = lanes.iter().map(|p| p.initial).collect();
                let inputs: Vec<_> = lanes.iter().map(|p| p.input).collect();
                let batch = ModifiedGreenAmptBatch::from_native_states(&parameters, &initial)?;
                let trial = batch.trial_native(probes[0].dt, &inputs)?;
                let outputs = trial.outputs()?;
                let states = trial.into_next().native_states()?;
                for ((probe, actual), output) in lanes.into_iter().zip(states).zip(outputs) {
                    state(&probe.case, actual, probe.expected)?;
                    close(
                        &probe.case,
                        "batched rate",
                        output.native_rate_ft_per_s,
                        probe.rate,
                        1e-14,
                    )?;
                    if (output.native_rate_ft_per_s == 0.0) != (probe.rate == 0.0) {
                        return Err(format!(
                            "case={} batch length={length}: zero-rate branch differs",
                            probe.case
                        )
                        .into());
                    }
                    if probe.case.starts_with("dry_timer_")
                        && (actual.f_ft == 0.0) != (probe.expected.f_ft == 0.0)
                    {
                        return Err(format!(
                            "case={} batch length={length}: timer-reset branch differs",
                            probe.case
                        )
                        .into());
                    }
                    checked += 1;
                }
            }
        }
    }
    println!("Checked {checked} rotated oracle lanes at lengths 31, 32, 33, 64, 65.");
    Ok(())
}
