use super::fixtures::{binding, exact, forcing, inflow, moving_cell, rate, withdrawal};
use crate::common::{TestResult, cell, definition, reach};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};

#[test]
fn incomplete_or_direction_mismatched_inputs_preserve_the_checkpoint() -> TestResult {
    let initial = moving_cell(vec![
        binding("in", "cell-0", ChannelLateralDirection::Inflow),
        binding("out", "cell-0", ChannelLateralDirection::Withdrawal),
    ])?;
    let before = initial.snapshot_data()?;
    let valid = [inflow("in", exact(0.0), 0.0), withdrawal("out", exact(0.0))];
    let mut cases = vec![
        vec![],
        vec![valid[0].clone()],
        vec![valid[0].clone(), valid[0].clone(), valid[1].clone()],
        vec![inflow("unknown", exact(0.0), 0.0), valid[1].clone()],
        vec![withdrawal("in", exact(0.0)), valid[1].clone()],
        vec![valid[0].clone(), inflow("out", exact(0.0), 0.0)],
    ];
    for amount in [
        exact(-1.0),
        rate(-1.0),
        exact(f64::NAN),
        rate(f64::INFINITY),
    ] {
        cases.push(vec![inflow("in", amount, 0.0), valid[1].clone()]);
    }
    for velocity in [f64::NAN, f64::INFINITY] {
        cases.push(vec![inflow("in", exact(1.0), velocity), valid[1].clone()]);
    }
    for inputs in cases {
        assert!(
            initial
                .step(0.01, &forcing(0.0, 0.01, inputs, true))
                .is_err()
        );
        assert_eq!(initial.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn lateral_bindings_require_unique_names_existing_cells_and_ownership_evidence() -> TestResult {
    let valid = binding("in", "cell-0", ChannelLateralDirection::Inflow);
    let mut cases = vec![vec![valid.clone(), valid.clone()]];
    let mut unknown_cell = valid.clone();
    unknown_cell.cell_id = "absent".into();
    cases.push(vec![unknown_cell]);
    let mut unnamed = valid.clone();
    unnamed.boundary_id.clear();
    cases.push(vec![unnamed]);
    let mut unsupported = valid;
    unsupported.ownership_assertion.clear();
    cases.push(vec![unsupported]);
    for bindings in cases {
        assert!(moving_cell(bindings).is_err());
    }
    Ok(())
}

#[test]
fn zero_withdrawals_report_only_defined_original_donor_velocity() -> TestResult {
    let input = withdrawal("out", rate(0.0));
    assert!(matches!(input, ChannelLateralInput::Withdrawal { .. }));
    let initial = moving_cell(vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )])?;
    let next = initial
        .step(0.01, &forcing(0.0, 0.01, vec![input], true))?
        .next;
    let receipt = next.last_step()?.ok_or("missing zero-withdrawal receipt")?;
    assert_eq!(receipt.lateral_receipts.len(), 1);
    assert_eq!(
        receipt.lateral_receipts[0].applied_longitudinal_velocity_m_per_s,
        Some(3.0)
    );
    let mut layout = definition(vec![cell(0, 4.0, 2.0, 0.0)], 1.0);
    layout.lateral_bindings = vec![binding(
        "out",
        "cell-0",
        ChannelLateralDirection::Withdrawal,
    )];
    let dry = reach(layout, &[0.0], &[0.0])?;
    let dry = dry
        .step(
            0.01,
            &forcing(0.0, 0.01, vec![withdrawal("out", rate(0.0))], false),
        )?
        .next;
    let receipt = dry
        .last_step()?
        .ok_or("missing dry zero-withdrawal receipt")?;
    assert_eq!(
        receipt.lateral_receipts[0].applied_longitudinal_velocity_m_per_s,
        None
    );
    Ok(())
}
