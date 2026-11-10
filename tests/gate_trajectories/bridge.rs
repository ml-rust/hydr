use super::fixture::*;
use hydr::structures::*;

#[test]
fn mixed_proposals_preserve_ordered_operations_and_complete_bound_context() -> Result {
    let original = mixed()?;
    let batch = original.with_gate_trajectories(mixed_assignment(
        stationary(0.0, 0.0, 2.0)?,
        stationary(0.5, 0.0, 2.0)?,
    ))?;
    let heads = mixed_heads();
    let others = other_operations();
    let evaluation = batch.evaluate(0.5)?;
    let gates = evaluation.operations()?;
    assert_eq!(
        evaluation
            .outputs()?
            .iter()
            .map(|output| output.element)
            .collect::<Vec<_>>(),
        vec![1, 3]
    );
    let complete = vec![
        others[0].clone(),
        gates[0].clone(),
        others[1].clone(),
        gates[1].clone(),
    ];
    let proposal = batch.propose_at(0.5, &heads, &others)?;
    assert_eq!(proposal.operations()?, complete);
    assert_eq!(
        proposal.outputs()?,
        original
            .propose_with_operations(&heads, &complete)?
            .outputs()?
    );
    let before = proposal.outputs()?;
    for invalid in [vec![], vec![others[0].clone()]] {
        expect_kind(
            batch.propose_at(0.5, &heads, &invalid),
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    for invalid in [
        vec![others[1].clone(), others[0].clone()],
        vec![gates[0].clone(), others[1].clone()],
    ] {
        expect_kind(
            batch.propose_at(0.5, &heads, &invalid),
            StructureInputErrorKind::Identity,
        )?;
    }
    expect_error(
        batch.propose_at(
            0.5,
            &heads,
            &[
                operation("passive", StructureOperatingPoint::PumpStoppedIsolated),
                others[1].clone(),
            ],
        ),
        "OperatingPoint",
        Some(0),
    )?;
    let mut unknown_positive = heads.clone();
    unknown_positive[3] = unknown("gate-1");
    expect_error(
        batch.propose_at(0.5, &unknown_positive, &others),
        "UnknownSupport",
        Some(3),
    )?;
    let mut bad_closed = heads.clone();
    bad_closed[1].endpoint_a.support = HeadSupport::Wet { stage_m: f64::NAN };
    expect_error(
        batch.propose_at(0.5, &bad_closed, &others),
        "NonfiniteInput",
        Some(1),
    )?;
    assert_eq!(proposal.outputs()?, before);
    Ok(())
}

#[test]
fn original_zero_support_alone_grants_unknown_head_isolation() -> Result {
    let batch = bind("binary_triangle", 1)?;
    for time in [0.0, 4.0] {
        let proposal = batch.propose_at(time, &[unknown("gate-0")], &[])?;
        assert_eq!(proposal.outputs()?[0].discharge_m3_per_s, 0.0);
    }
    expect_error(
        batch.propose_at(0.5, &[unknown("gate-0")], &[]),
        "UnknownSupport",
        Some(0),
    )?;
    let cancellation = case("cancellation_huge_tolerances")?;
    let batch = bind(cancellation.id, 1)?;
    expect_error(
        batch.propose_at(cancellation.queries[0].time, &[unknown("gate-0")], &[]),
        "Numerical",
        Some(0),
    )?;
    let all_gate = bind("constant_zero", 1)?;
    expect_kind(
        all_gate.propose_at(
            0.0,
            &[unknown("gate-0")],
            &[operation("foreign", StructureOperatingPoint::Passive)],
        ),
        StructureInputErrorKind::OperatingPoint,
    )?;
    Ok(())
}
