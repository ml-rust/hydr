//! Self-contained differential-consumer coverage independent of ignored evidence.

use super::{comparison, records};
type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn numbers(row: &mut [String], start: usize, values: &[f64]) {
    for (field, value) in row[start..].iter_mut().zip(values) {
        *field = format!("{value:.17}");
    }
}

fn row(
    case: &str,
    index: usize,
    operation: &str,
    model: usize,
    imd: f64,
    timer: f64,
) -> Vec<String> {
    let mut row = vec![String::new(); 35];
    row[0] = case.into();
    row[1] = index.to_string();
    row[2] = operation.into();
    row[3] = model.to_string();
    row[4] = "INIT".into();
    numbers(&mut row, 5, &[0.4, 1e-5, 0.3]);
    numbers(&mut row, 8, &[imd, 0.0, 0.0, 0.0, 0.0]);
    row[27] = "0".into();
    row[29] = "0.21908902300206645".into();
    numbers(&mut row, 30, &[imd, 0.0, 0.0, 0.0, timer]);
    if index > 1 {
        numbers(&mut row, 13, &[imd, 0.0, 0.0, 0.0, timer]);
    }
    row
}

fn initial(case: &str, model: usize, imd: f64) -> Vec<String> {
    row(case, 1, "INIT", model, imd, 0.0)
}

fn dry_step(case: &str, index: usize, model: usize, imd: f64, before_t: f64) -> Vec<String> {
    let mut row = row(case, index, "STEP", model, imd, before_t - 60.0);
    numbers(&mut row, 13, &[imd, 0.0, 0.0, 0.0, before_t]);
    numbers(&mut row, 18, &[60.0, 0.0, 0.0]);
    numbers(&mut row, 25, &[0.0, 0.0]);
    row
}

fn root(case: &str, index: usize, ks: f64, expected: f64, timer: f64) -> Vec<String> {
    let mut row = row(case, index, "F2", 3, 0.3, timer);
    // Source c1=0 returns the analytic minimum f1+ks*ts.
    numbers(&mut row, 21, &[0.1, 0.0, ks, 60.0]);
    row[28] = format!("{expected:.17}");
    row
}

fn csv(rows: Vec<Vec<String>>) -> String {
    let mut text = records::HEADER.to_owned();
    for row in rows {
        text.push('\n');
        text.push_str(&row.join(","));
    }
    text.push('\n');
    text
}

#[test]
fn renamed_step_sequence_receives_numerical_comparison() -> Result {
    let case = "F2_renamed_sequence";
    let input = csv(vec![initial(case, 3, 0.3), dry_step(case, 2, 3, 0.3, 0.0)]);
    comparison::compare(&records::parse(&input)?)
}

#[test]
fn renamed_step_sequence_checks_initialization_before_batch_probes() -> Result {
    let case = "F2_renamed_sequence";
    // All CSV state links agree, but INIT incorrectly claims IMD=0.2 instead of IMDmax=0.3.
    let input = csv(vec![initial(case, 3, 0.2), dry_step(case, 2, 3, 0.2, 0.0)]);
    let error = comparison::compare(&records::parse(&input)?)
        .err()
        .ok_or("incorrect initialization returned success")?;
    assert!(error.to_string().contains("field=IMD"));
    Ok(())
}

#[test]
fn positive_conductivity_root_is_checked_despite_zero_ks_label() -> Result {
    let case = "F2_zero_ks";
    let input = csv(vec![
        initial(case, 3, 0.3),
        root(case, 2, 1e-5, 0.1009, 0.0),
    ]);
    let error = comparison::compare(&records::parse(&input)?)
        .err()
        .ok_or("incorrect positive-Ks root returned success")?;
    assert!(error.to_string().contains("F2 through saturated state"));
    Ok(())
}

#[test]
fn accumulated_interval_infiltration_is_checked_independently() -> Result {
    let case = "dry_sequence";
    let mut step = dry_step(case, 2, 3, 0.3, 0.0);
    step[27] = "0.01".into();
    let input = csv(vec![initial(case, 3, 0.3), step]);
    let error = comparison::compare(&records::parse(&input)?)
        .err()
        .ok_or("incorrect accumulated depth returned success")?;
    assert!(
        error
            .to_string()
            .contains("accumulated interval infiltration")
    );
    Ok(())
}

#[test]
fn unsupported_only_and_initialization_only_inputs_reject() -> Result {
    for input in [
        csv(vec![
            initial("original", 2, 0.3),
            dry_step("original", 2, 2, 0.3, 0.0),
        ]),
        csv(vec![
            initial("renamed_zero_conductivity", 3, 0.3),
            root("renamed_zero_conductivity", 2, 0.0, 0.1, 0.0),
        ]),
        csv(vec![initial("initialization_only", 3, 0.3)]),
    ] {
        let error = comparison::compare(&records::parse(&input)?)
            .err()
            .ok_or("input without a supported numerical comparison returned success")?;
        assert!(
            error
                .to_string()
                .contains("no supported STEP or F2 numerical comparison")
        );
    }
    Ok(())
}

#[test]
fn direct_root_preserves_the_sequential_soil_context() -> Result {
    let case = "mixed_operations";
    let input = csv(vec![
        initial(case, 3, 0.3),
        dry_step(case, 2, 3, 0.3, 0.0),
        root(case, 3, 1e-5, 0.1 + 1e-5 * 60.0, -60.0),
        dry_step(case, 4, 3, 0.3, -60.0),
    ]);
    comparison::compare(&records::parse(&input)?)
}
