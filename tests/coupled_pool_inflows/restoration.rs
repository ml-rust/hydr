use super::fixture::Result;
use super::inflow_fixture::{binding, forcing, rate};
use super::source_fixture::{Batch, policies, pump_case, rates};
use hydr::coupling::CoupledPoolInflowValue;
use hydr::coupling::{
    CoupledLevelPoolForcingInterval, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
};

fn case() -> Result<(
    CoupledLevelPoolSetup,
    CoupledLevelPoolInitialState,
    CoupledLevelPoolForcingInterval,
)> {
    let (mut setup, initial) = pump_case([4.0, 2.0])?;
    setup.source_policies = Some(policies(2));
    setup.external_inflows = vec![binding("left", "a"), binding("right", "b")];
    let mut forcing = forcing(vec![rate("left", 0.5), rate("right", 0.25)]);
    forcing.sources = Some(vec![rates(1.0, 0.25, 0.5), rates(0.0, 0.0, 0.0)]);
    Ok((setup, initial, forcing))
}

#[test]
fn trusted_and_cold_restart_preserve_external_history_and_repeat_continuation() -> Result {
    let (setup, initial, forcing) = case()?;
    let mut batch = Batch::new(crate::runtime::client(), setup.clone(), initial.clone())?;
    for end in [0.25, 0.5] {
        batch = batch.trial_with_forcing_to(end, &forcing)?.into_next();
    }
    let trusted = batch.snapshot();
    let owned = batch.snapshot_data()?;
    assert_eq!(trusted.data()?, owned);
    let expected = batch
        .trial_with_forcing_to(0.75, &forcing)?
        .into_next()
        .snapshot_data()?;
    batch = batch.trial_with_forcing_to(0.75, &forcing)?.into_next();
    batch.restore(&trusted)?;
    assert_eq!(batch.snapshot_data()?, owned);
    assert_eq!(
        batch
            .trial_with_forcing_to(0.75, &forcing)?
            .into_next()
            .snapshot_data()?,
        expected
    );
    let mut cold = Batch::new(crate::runtime::client(), setup, initial)?;
    cold.restore_data(&owned)?;
    assert_eq!(cold.snapshot_data()?, owned);
    assert_eq!(
        cold.trial_with_forcing_to(0.75, &forcing)?
            .into_next()
            .snapshot_data()?,
        expected
    );
    Ok(())
}

#[test]
fn altered_current_start_and_original_external_operands_reject_atomically() -> Result {
    let (setup, initial, forcing) = case()?;
    let mut batch = Batch::new(crate::runtime::client(), setup, initial)?;
    for end in [0.25, 0.5] {
        batch = batch.trial_with_forcing_to(end, &forcing)?.into_next();
    }
    let original = batch.snapshot_data()?;
    for mode in 0..15 {
        let mut altered = original.clone();
        match mode {
            0 => altered.external_inflows[0].cumulative_volume_m3 += 0.25,
            1 => altered.external_inflows[0].roundoff_bound_m3 = -1.0,
            2 => altered.external_inflows[0].input_id = "other-input".into(),
            3..=5 => {
                let start = altered.start.as_mut().ok_or("Missing bounded start")?;
                match mode {
                    3 => start.external_inflows[0].cumulative_volume_m3 += 0.25,
                    4 => start.external_inflows[0].roundoff_bound_m3 = -1.0,
                    5 => start.external_inflows[0].boundary_id = "other-boundary".into(),
                    _ => return Err("Unknown start ledger mutation".into()),
                }
            }
            6..=12 => {
                let receipt = &mut altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing coupled receipt")?
                    .external_inflows[0];
                match mode {
                    6 => receipt.volume_m3 += 0.125,
                    7 => receipt.addition_witness_m3 = 1.0,
                    8 => {
                        receipt.input = CoupledPoolInflowValue::Rate {
                            rate_m3_per_s: 0.25,
                        }
                    }
                    9 => receipt.dt_s *= 0.5,
                    10 => receipt.cumulative_before_m3 += 0.25,
                    11 => receipt.cumulative_after_m3 += 0.25,
                    12 => receipt.bound_after_m3 = -1.0,
                    _ => return Err("Unknown external receipt mutation".into()),
                }
            }
            13 => {
                altered
                    .last_step
                    .as_mut()
                    .ok_or("Missing coupled receipt")?
                    .forcing_interval
                    .as_mut()
                    .ok_or("Missing original forcing")?
                    .inflows[0]
                    .value = CoupledPoolInflowValue::Rate {
                    rate_m3_per_s: 0.75,
                }
            }
            14 => {
                altered.external_inflows.pop();
            }
            _ => return Err("Unknown external checkpoint mutation".into()),
        }
        assert!(
            batch.restore_data(&altered).is_err(),
            "Accepted external checkpoint mutation={mode}"
        );
        assert_eq!(batch.snapshot_data()?, original);
    }
    Ok(())
}
