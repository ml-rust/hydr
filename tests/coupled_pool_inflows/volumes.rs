use super::fixture::{DATUM, Result, assemble, near, orifice_case};
use super::inflow_fixture::{binding, rate, span, volume};
use super::laws::{pump, running};
use super::source_fixture::{Batch, inactive_case};
use hydr::coupling::{CoupledLevelPoolError, CoupledPoolInflowValue};
use hydr::storage::{LevelPoolCurve, LevelPoolKnot};

#[test]
fn exact_and_mixed_inputs_keep_original_volume_operands() -> Result {
    for mixed in [false, true] {
        let (mut setup, initial) = inactive_case([1.0, 1.0])?;
        setup.config.maximum_step_s = 2.0;
        setup.external_inflows = vec![binding("integrated", "a")];
        let mut inputs = vec![volume("integrated", 0.75)];
        if mixed {
            setup.external_inflows.push(binding("rate", "a"));
            inputs.push(rate("rate", 0.25));
        }
        let forcing = span(0.0, 2.0, inputs);
        let batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let next = batch.trial_with_forcing_to(2.0, &forcing)?.into_next();
        assert_eq!(next.time(), 2.0);
        near(
            next.pools().states()?[0].volume_m3,
            if mixed { 2.25 } else { 1.75 },
        );
        let step = next.last_step()?.ok_or("Missing exact-volume receipt")?;
        assert_eq!(step.retries, 0);
        assert_eq!(step.forcing_interval, Some(forcing));
        assert_eq!(
            step.external_inflows[0].input,
            CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.75 }
        );
        assert_eq!(
            step.external_inflows[0].volume_m3.to_bits(),
            0.75_f64.to_bits()
        );
        if mixed {
            assert_eq!(
                step.external_inflows[1].volume_m3.to_bits(),
                0.5_f64.to_bits()
            );
        }
    }
    Ok(())
}

#[test]
fn represented_integrated_film_preserves_bits_and_unresolved_midpoint_rejects_atomically() -> Result
{
    let smallest = f64::from_bits(1);
    assert_eq!((smallest / 2.0).to_bits(), 0.0_f64.to_bits());
    let represented = f64::from_bits(3);
    assert_eq!(
        ((represented / 2.0) * 2.0).to_bits(),
        f64::from_bits(4).to_bits()
    );
    let (mut setup, initial) = inactive_case([0.0, 1.0])?;
    setup.config.maximum_step_s = 2.0;
    setup.pools[0].curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 0.0,
            },
            LevelPoolKnot {
                volume_m3: 1.0,
                stage_m: 1.0,
            },
        ],
        DATUM.into(),
        "Exact unit-area dry receiver over the linear zero-to-one interval".into(),
    )?;
    setup.external_inflows = vec![binding("smallest", "a")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_with_forcing_to(2.0, &span(0.0, 2.0, vec![volume("smallest", smallest)]))
        .err()
        .ok_or("Unrepresented midpoint was accepted")?;
    assert!(error.to_string().contains("midpoint"), "{error}");
    assert_eq!(batch.snapshot_data()?, before);
    let forcing = span(0.0, 2.0, vec![volume("smallest", represented)]);
    let next = batch.trial_with_forcing_to(2.0, &forcing)?.into_next();
    let output = &next.pools().outputs()?[0];
    assert_eq!(output.volume_m3.to_bits(), represented.to_bits());
    assert_eq!(
        output
            .stage_m
            .ok_or("Represented film lost its stage")?
            .to_bits(),
        represented.to_bits()
    );
    let step = next
        .last_step()?
        .ok_or("Missing represented film receipt")?;
    assert_eq!(
        step.transfers[0]
            .midpoint_stage_a_m
            .ok_or("Missing represented film midpoint")?
            .to_bits(),
        2
    );
    let receipt = &step.external_inflows[0];
    match receipt.input {
        CoupledPoolInflowValue::IntervalVolume { volume_m3 } => {
            assert_eq!(volume_m3.to_bits(), represented.to_bits())
        }
        other => return Err(format!("Integrated operand changed kind: {other:?}").into()),
    }
    assert_eq!(receipt.volume_m3.to_bits(), represented.to_bits());
    assert_eq!(
        next.external_inflows()?[0].cumulative_volume_m3.to_bits(),
        represented.to_bits()
    );
    Ok(())
}

#[test]
fn smallest_integrated_input_survives_when_the_receiving_film_has_a_represented_midpoint() -> Result
{
    let smallest = f64::from_bits(1);
    assert_eq!((smallest / 2.0).to_bits(), 0.0_f64.to_bits());
    let (mut setup, initial) = inactive_case([smallest, 1.0])?;
    setup.config.maximum_step_s = 2.0;
    setup.pools[0].curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 0.0,
            },
            LevelPoolKnot {
                volume_m3: 1.0,
                stage_m: 1.0,
            },
        ],
        DATUM.into(),
        "Exact unit-area positive-film receiver over the linear zero-to-one interval".into(),
    )?;
    setup.external_inflows = vec![binding("smallest", "a")];
    let head_budget = setup.config.head_absolute_m;
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(2.0, &span(0.0, 2.0, vec![volume("smallest", smallest)]))?
        .into_next();
    let output = &next.pools().outputs()?[0];
    assert_eq!(output.volume_m3.to_bits(), 2);
    assert_eq!(
        output
            .stage_m
            .ok_or("Positive film lost its stage")?
            .to_bits(),
        2
    );
    let step = next.last_step()?.ok_or("Missing minimum-input receipt")?;
    // The retained midpoint uses the converged estimate within its declared head budget.
    let midpoint = step.transfers[0]
        .midpoint_stage_a_m
        .ok_or("Missing positive-film midpoint")?;
    assert!(midpoint.is_finite() && midpoint > 0.0);
    assert!(step.head_residual_m.is_finite() && step.head_residual_m <= head_budget);
    let receipt = &step.external_inflows[0];
    assert_eq!(receipt.volume_m3.to_bits(), smallest.to_bits());
    match receipt.input {
        CoupledPoolInflowValue::IntervalVolume { volume_m3 } => {
            assert_eq!(volume_m3.to_bits(), smallest.to_bits())
        }
        other => return Err(format!("Smallest integrated input changed kind: {other:?}").into()),
    }
    assert_eq!(
        next.external_inflows()?[0].cumulative_volume_m3.to_bits(),
        smallest.to_bits()
    );
    Ok(())
}

#[test]
fn integrated_zero_and_positive_volumes_require_exact_span_bits() -> Result {
    for amount in [0.0, 0.75] {
        let (mut setup, initial) = inactive_case([1.0, 1.0])?;
        setup.external_inflows = vec![binding("integrated", "a")];
        let batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let before = batch.snapshot_data()?;
        for (start, end, requested) in [(-0.0, 1.0, 1.0), (-0.25, 1.0, 1.0), (0.0, 2.0, 1.0)] {
            assert!(
                batch
                    .trial_with_forcing_to(
                        requested,
                        &span(start, end, vec![volume("integrated", amount)])
                    )
                    .is_err()
            );
            assert_eq!(batch.snapshot_data()?, before);
        }
        let next = batch
            .trial_with_forcing_to(0.5, &span(0.0, 0.5, vec![volume("integrated", amount)]))?
            .into_next();
        let before = next.snapshot_data()?;
        assert!(
            next.trial_with_forcing_to(1.0, &span(0.0, 1.0, vec![volume("integrated", amount)]))
                .is_err()
        );
        assert_eq!(next.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn maximum_step_requests_fresh_forcing_without_accepting_or_scaling_volume() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.config.maximum_step_s = 0.5;
    setup.external_inflows = vec![binding("integrated", "a")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_with_forcing_to(1.0, &span(0.0, 1.0, vec![volume("integrated", 0.75)]))
        .err()
        .ok_or("Exact volume was automatically shortened")?;
    match error {
        CoupledLevelPoolError::RecomputeInterval {
            start_s,
            requested_end_s,
            suggested_end_s,
            reason,
        } => {
            assert_eq!(start_s, 0.0);
            assert_eq!(requested_end_s, 1.0);
            assert_eq!(suggested_end_s, 0.5);
            assert!(!reason.is_empty());
        }
        other => return Err(format!("Unexpected exact-span limit error: {other}").into()),
    }
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn unconverged_exact_span_requests_fresh_operands_from_original_accepted_state() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.config.iteration_limit = 8;
    let minimum = setup.config.minimum_step_s;
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    let mut end = 1.0;
    let mut requests = 0;
    loop {
        let forcing = span(
            0.0,
            end,
            vec![volume("left", 0.25 * end), volume("right", 0.25 * end)],
        );
        match batch.trial_with_forcing_to(end, &forcing) {
            Ok(trial) => {
                assert!(requests > 0);
                let next = trial.into_next();
                assert_eq!(next.time(), end);
                let transfer = end * 0.5 * ((end * end + 8.0).sqrt() - end);
                let step = next
                    .last_step()?
                    .ok_or("Missing recomputed exact interval")?;
                assert_eq!(step.retries, 0);
                near(step.transfers[0].signed_volume_m3, transfer);
                near(
                    next.pools().states()?[0].volume_m3,
                    4.0 + 0.25 * end - transfer,
                );
                near(
                    next.pools().states()?[1].volume_m3,
                    2.0 + 0.25 * end + transfer,
                );
                break;
            }
            Err(CoupledLevelPoolError::RecomputeInterval {
                start_s,
                requested_end_s,
                suggested_end_s,
                reason,
            }) => {
                assert_eq!(start_s, 0.0);
                assert_eq!(requested_end_s, end);
                assert!(
                    suggested_end_s > 0.0 && suggested_end_s < end && suggested_end_s >= minimum
                );
                assert!(!reason.is_empty());
                requests += 1;
                assert!(
                    requests <= 32,
                    "Exact-input requests did not reach an admitted interval"
                );
                end = suggested_end_s;
            }
            Err(other) => {
                return Err(format!("Unexpected exact-input convergence error: {other}").into());
            }
        }
        assert_eq!(batch.snapshot_data()?, before);
    }
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn integrated_inflow_cannot_finance_same_interval_structure_withdrawals() -> Result {
    let (mut setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[1.0, 1.0],
        vec![pump("pump", "a", "b", &[(-16.0, 2.0), (16.0, 2.0)])?],
        vec![running("pump")],
    )?;
    setup.external_inflows = vec![binding("donor", "a")];
    let next = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &span(0.0, 1.0, vec![volume("donor", 1.0)]))?
        .into_next();
    let step = next.last_step()?.ok_or("Missing integrated donor limit")?;
    assert_eq!(step.retries, 0);
    assert!(step.transfers[0].signed_volume_m3 <= 1.0);
    near(step.transfers[0].signed_volume_m3, 1.0);
    near(next.pools().states()?[0].volume_m3, 1.0);
    near(next.pools().states()?[1].volume_m3, 2.0);
    Ok(())
}
