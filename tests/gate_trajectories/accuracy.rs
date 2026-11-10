use super::{fixture::*, receipt::ReceiptObserver, reference_inputs::cases};
use hydr::structures::*;

#[test]
fn explicit_accuracy_policies_preserve_the_unmodified_solvr_receipt() -> Result {
    let observer = ReceiptObserver::new();
    for case in cases().filter(|case| case.family == "public") {
        let query = case.queries.first().ok_or("Missing accuracy query")?;
        let raw = observer.observe(case, query.time, 1)?;
        assert_eq!(raw.status, "success");
        assert!(raw.errors[0] > 0.0);
        let batch = bind(case.id, 1)?;
        match batch.evaluate(query.time) {
            Ok(evaluation) => {
                assert_eq!(query.expected, "success");
                let value = evaluation
                    .outputs()?
                    .into_iter()
                    .next()
                    .ok_or("Missing accuracy output")?;
                assert_eq!(value.opening_m.to_bits(), raw.values[0].to_bits());
                assert_eq!(value.absolute_error_m.to_bits(), raw.errors[0].to_bits());
                assert!(value.opening_m > 0.0 && !value.closed_by_definition);
            }
            Err(error) => super::reference_case::check_error(&error, query.expected, Some(0))?,
        }
    }
    Ok(())
}

#[test]
fn exact_only_policy_admits_original_holds_and_knots_without_claiming_exact_interiors() -> Result {
    let base = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![super::reference_case::gate("gate-0", 1.0)?],
    )?;
    let hold = base.with_gate_trajectories(vec![GateTrajectoryAssignment {
        structure_id: "gate-0".into(),
        trajectory: stationary(0.5, 0.0, 1.0)?,
    }])?;
    assert_eq!(hold.evaluate(0.5)?.outputs()?[0].absolute_error_m, 0.0);
    let ramp = base.with_gate_trajectories(vec![GateTrajectoryAssignment {
        structure_id: "gate-0".into(),
        trajectory: prescribed(&[(0.0, 0.0), (1.0, 1.0)], exact())?,
    }])?;
    expect_error(ramp.evaluate(0.5), "Numerical", Some(0))?;
    assert_eq!(ramp.evaluate(0.0)?.outputs()?[0].absolute_error_m, 0.0);
    assert_eq!(ramp.evaluate(1.0)?.outputs()?[0].absolute_error_m, 0.0);
    Ok(())
}
