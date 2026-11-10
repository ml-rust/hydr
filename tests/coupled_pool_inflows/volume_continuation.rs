use super::fixture::Result;
use super::inflow_fixture::{binding, rate, span, volume};
use super::source_fixture::{Batch, inactive_case};
use hydr::coupling::{CoupledLevelPoolError, CoupledPoolInflowValue};

#[test]
fn invalid_integrated_operands_and_absorbed_assembly_preserve_accepted_state() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.config.minimum_step_s = 1.0;
    setup.external_inflows = vec![binding("integrated", "a"), binding("other", "a")];
    let batch = Batch::new(crate::runtime::client(), setup, initial)?;
    let before = batch.snapshot_data()?;
    for bad in [-1.0, f64::NAN, f64::INFINITY] {
        let error = batch
            .trial_with_forcing_to(
                1.0,
                &span(
                    0.0,
                    1.0,
                    vec![volume("integrated", bad), volume("other", 0.0)],
                ),
            )
            .err()
            .ok_or("Invalid integrated operand was accepted")?;
        assert!(!matches!(
            error,
            CoupledLevelPoolError::RecomputeInterval { .. }
        ));
        assert_eq!(batch.snapshot_data()?, before);
    }
    assert!(
        batch
            .trial_with_forcing_to(
                1.0,
                &span(
                    0.0,
                    1.0,
                    vec![volume("missing", 0.75), volume("other", 0.0)]
                )
            )
            .is_err()
    );
    assert!(
        batch
            .trial_with_forcing_to(
                1.0,
                &span(
                    0.0,
                    1.0,
                    vec![
                        volume("integrated", 1.0),
                        volume("other", f64::EPSILON / 4.0)
                    ]
                )
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn exact_and_mixed_intervals_reproduce_trusted_and_owned_continuation() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.external_inflows = vec![binding("integrated", "a"), binding("rate", "a")];
    let first = span(
        0.0,
        1.0,
        vec![volume("integrated", 0.75), volume("rate", 0.0)],
    );
    let second = span(
        1.0,
        2.0,
        vec![volume("integrated", 0.375), rate("rate", 0.125)],
    );
    let mut batch = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?
        .trial_with_forcing_to(1.0, &first)?
        .into_next();
    let trusted = batch.snapshot();
    let owned = batch.snapshot_data()?;
    let expected = batch
        .trial_with_forcing_to(2.0, &second)?
        .into_next()
        .snapshot_data()?;
    batch = batch.trial_with_forcing_to(2.0, &second)?.into_next();
    batch.restore(&trusted)?;
    assert_eq!(batch.snapshot_data()?, owned);
    assert_eq!(
        batch
            .trial_with_forcing_to(2.0, &second)?
            .into_next()
            .snapshot_data()?,
        expected
    );
    let mut cold = Batch::new(crate::runtime::client(), setup, initial)?;
    cold.restore_data(&owned)?;
    assert_eq!(cold.snapshot_data()?, owned);
    let mut continued = cold.trial_with_forcing_to(2.0, &second)?.into_next();
    assert_eq!(continued.snapshot_data()?, expected);
    continued.restore_data(&expected)?;
    assert_eq!(continued.snapshot_data()?, expected);
    Ok(())
}

#[test]
fn original_kind_operand_duration_and_retry_forgeries_reject_atomically() -> Result {
    let (mut setup, initial) = inactive_case([1.0, 1.0])?;
    setup.external_inflows = vec![binding("integrated", "a")];
    let forcing = span(0.0, 1.0, vec![volume("integrated", 0.75)]);
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?
        .trial_with_forcing_to(1.0, &forcing)?
        .into_next();
    let original = batch.snapshot_data()?;
    for mode in 0..8 {
        let mut altered = original.clone();
        let step = altered
            .last_step
            .as_mut()
            .ok_or("Missing exact-volume receipt")?;
        match mode {
            0 => {
                step.external_inflows[0].input = CoupledPoolInflowValue::Rate {
                    rate_m3_per_s: 0.75,
                }
            }
            1 => {
                step.external_inflows[0].input =
                    CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.5 }
            }
            2 => step.external_inflows[0].dt_s = 0.5,
            3 => step.retries = 1,
            4 => step.requested_end_s = 2.0,
            5..=7 => {
                let forcing = step
                    .forcing_interval
                    .as_mut()
                    .ok_or("Missing original integrated forcing")?;
                match mode {
                    5 => {
                        forcing.inflows[0].value = CoupledPoolInflowValue::Rate {
                            rate_m3_per_s: 0.75,
                        }
                    }
                    6 => {
                        forcing.inflows[0].value =
                            CoupledPoolInflowValue::IntervalVolume { volume_m3: 0.5 }
                    }
                    7 => forcing.end_s = 2.0,
                    _ => return Err("Unknown original forcing mutation".into()),
                }
            }
            _ => return Err("Unknown integrated checkpoint mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted integrated checkpoint mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}
