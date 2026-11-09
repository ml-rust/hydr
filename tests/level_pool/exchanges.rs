use super::fixture::{Result, constant, input, rejection, single, tolerance, unit};

#[test]
fn ordinary_decimal_exchange_retains_original_transfers_and_exact_twosum_witnesses() -> Result {
    let batch = single(constant()?, 1.0)?;
    let trial = batch.trial_to(1.0, &[input(0.2, 0.1)])?;
    let diagnostics = trial.diagnostics()?;
    let receipt = diagnostics
        .receipts
        .first()
        .ok_or("missing exchange receipt")?;
    assert_eq!(receipt.incoming_volume_m3.to_bits(), 0.2_f64.to_bits());
    assert_eq!(receipt.outgoing_volume_m3.to_bits(), 0.1_f64.to_bits());
    assert_eq!(receipt.remaining_volume_m3.to_bits(), 0.9_f64.to_bits());
    assert_eq!(receipt.volume_m3.to_bits(), 1.1_f64.to_bits());
    assert_eq!(
        receipt.outgoing_witness_m3.to_bits(),
        (-2.0_f64.powi(-55)).to_bits()
    );
    assert_eq!(
        receipt.incoming_witness_m3.to_bits(),
        (-2.0_f64.powi(-54)).to_bits()
    );
    assert_ne!(
        (receipt.volume_m3 - receipt.incoming_volume_m3).to_bits(),
        receipt.remaining_volume_m3.to_bits()
    );
    assert_eq!(
        receipt.water_residual_m3.to_bits(),
        (3.0_f64 / 36028797018963968.0).to_bits()
    );
    assert!(receipt.water_residual_m3.abs() <= tolerance(receipt.physical_scale_m3));
    assert!(receipt.witness_consistency_m3.abs() <= receipt.witness_consistency_bound_m3);
    let next = trial.into_next();
    assert_eq!(next.states()?[0].volume_m3.to_bits(), 1.1_f64.to_bits());
    Ok(())
}

#[test]
fn incoming_water_cannot_finance_the_same_interval_donor_budget() -> Result {
    let batch = single(unit(16.0)?, 1.0)?;
    let before = batch.snapshot_data()?;
    let error = rejection(batch.trial_to(1.0, &[input(10.0, 1.1)]))?;
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("without incoming credit"));
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn exact_drain_fill_and_signed_zero_operations_preserve_authoritative_volume() -> Result {
    let batch = single(unit(4.0)?, 1.0)?;
    let empty = batch.trial_to(1.0, &[input(0.0, 1.0)])?.into_next();
    assert_eq!(empty.outputs()?[0].volume_m3, 0.0);
    assert_eq!(empty.outputs()?[0].stage_m, None);
    let full = empty.trial_to(2.0, &[input(4.0, 0.0)])?.into_next();
    assert_eq!(full.outputs()?[0].volume_m3, 4.0);
    assert_eq!(full.outputs()?[0].stage_m, Some(4.0));
    let negative_zero = single(unit(1.0)?, -0.0)?;
    let trial = negative_zero.trial_to(1.0, &[input(0.0, 0.0)])?;
    assert_eq!(
        trial.outputs()?[0].volume_m3.to_bits(),
        (-0.0_f64).to_bits()
    );
    let receipt = trial
        .diagnostics()?
        .receipts
        .into_iter()
        .next()
        .ok_or("missing zero-operation receipt")?;
    for witness in [
        receipt.outgoing_witness_m3,
        receipt.incoming_witness_m3,
        receipt.cumulative_incoming_witness_m3,
        receipt.cumulative_outgoing_witness_m3,
    ] {
        assert_eq!(witness.to_bits(), 0.0_f64.to_bits());
    }
    Ok(())
}

#[test]
fn absorbed_exchanges_and_nonfinite_arithmetic_reject_without_changing_state() -> Result {
    for (maximum, volume, exchange) in [
        (2.0_f64.powi(54), 2.0_f64.powi(53), input(1.0, 0.0)),
        (2.0_f64.powi(55), 2.0_f64.powi(54), input(0.0, 1.0)),
        (f64::MAX, 0.75 * f64::MAX, input(0.75 * f64::MAX, 0.0)),
    ] {
        let batch = single(unit(maximum)?, volume)?;
        let before = batch.snapshot_data()?;
        assert!(batch.trial_to(1.0, &[exchange]).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    let batch = single(constant()?, 1.0)?;
    let before = batch.snapshot_data()?;
    for inputs in [
        vec![],
        vec![input(-1.0, 0.0)],
        vec![input(0.0, f64::INFINITY)],
        vec![input(f64::NAN, 0.0)],
    ] {
        assert!(batch.trial_to(1.0, &inputs).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
