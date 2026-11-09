use super::fixture::{Result, parameters, single};
use hydr::runoff::{ReservoirConfig, ReservoirInput, ReservoirParameters};

#[test]
fn depression_filling_retains_supplied_storage_without_export() -> Result {
    let p = parameters()?;
    let rain = 0.3048 / 1024.0;
    for dt in [64.0, 128.0] {
        let batch = single(p, 0.0, ReservoirConfig::default())?;
        let trial = batch.trial_to(
            dt,
            &[ReservoirInput {
                incoming_rate_m_per_s: rain,
            }],
        )?;
        let output = &trial.outputs()?[0];
        assert_eq!(output.depth_m.to_bits(), (rain * dt).to_bits());
        assert_eq!(output.export_volume_m3, 0.0);
        assert_eq!(output.endpoint_discharge_m3_per_s, Some(0.0));
        assert_eq!(
            output.storage_m3.to_bits(),
            (p.area_m2() * rain * dt).to_bits()
        );
        assert_eq!(trial.diagnostics()?.root_iterations, 0);
    }
    Ok(())
}

#[test]
fn storage_threshold_neighbors_preserve_films_and_reject_unresolved_drainage() -> Result {
    let p = parameters()?;
    let storage = p.depression_depth_m();
    for supply in [storage.next_down(), storage] {
        let batch = single(p, 0.0, ReservoirConfig::default())?;
        let trial = batch.trial_to(
            128.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: supply / 128.0,
            }],
        )?;
        assert_eq!(trial.outputs()?[0].depth_m.to_bits(), supply.to_bits());
        assert_eq!(trial.outputs()?[0].export_volume_m3, 0.0);
    }
    let batch = single(p, 0.0, ReservoirConfig::default())?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_to(
                128.0,
                &[ReservoirInput {
                    incoming_rate_m_per_s: storage.next_up() / 128.0
                }]
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn instantaneous_overflow_exports_the_native_established_available_volume_once() -> Result {
    let p = ReservoirParameters::from_native(1024.0, 1024.0, 32.0, 1.0 / 64.0, 0.0, 0.125)?;
    assert!(p.is_instantaneous());
    assert_eq!(p.coefficient_si(), None);
    let batch = single(p, 0.0, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        256.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.3048 / 1024.0,
        }],
    )?;
    let output = &trial.outputs()?[0];
    let native_volume_si = 128.0 * 0.3048_f64.powi(3);
    assert!((output.export_volume_m3 - native_volume_si).abs() <= 1e-13);
    assert_eq!(output.depth_m.to_bits(), p.depression_depth_m().to_bits());
    assert_eq!(output.endpoint_discharge_m3_per_s, None);
    assert_eq!(trial.diagnostics()?.root_iterations, 0);
    let next = trial.into_next();
    let dry_trial = next.trial_to(
        512.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.0,
        }],
    )?;
    assert_eq!(dry_trial.outputs()?[0].export_volume_m3, 0.0);
    assert_eq!(
        dry_trial.into_next().ledger().export_volume_m3.to_bits(),
        output.export_volume_m3.to_bits()
    );
    Ok(())
}

#[test]
fn zero_width_or_slope_keeps_all_supplied_water_in_closed_storage() -> Result {
    for (width, slope) in [(0.0, 1.0 / 64.0), (32.0, 0.0)] {
        let p = ReservoirParameters::from_native(1024.0, 1024.0, width, slope, 1.0 / 32.0, 0.125)?;
        assert!(p.is_closed());
        let batch = single(p, 0.0, ReservoirConfig::default())?;
        let trial = batch.trial_to(
            256.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.3048 / 1024.0,
            }],
        )?;
        let output = &trial.outputs()?[0];
        assert_eq!(output.depth_m.to_bits(), (0.3048_f64 / 4.0).to_bits());
        assert_eq!(output.export_volume_m3, 0.0);
        assert_eq!(output.endpoint_discharge_m3_per_s, Some(0.0));
        assert_eq!(trial.diagnostics()?.root_iterations, 0);
    }
    Ok(())
}

#[test]
fn finite_overflow_reports_storage_export_instead_of_full_interval_endpoint_rate() -> Result {
    let p = parameters()?;
    let batch = single(p, 0.0, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        256.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.3048 / 1024.0,
        }],
    )?;
    let diagnostics = trial.diagnostics()?;
    let receipt = &diagnostics.receipts[0];
    assert_eq!(receipt.active_duration_s, 128.0);
    let endpoint = receipt
        .endpoint_discharge_m3_per_s
        .ok_or("missing endpoint discharge")?;
    assert!((receipt.export_volume_m3 - endpoint * receipt.active_duration_s).abs() <= 1e-9);
    assert!((receipt.export_volume_m3 - endpoint * diagnostics.dt).abs() > 1.0);
    assert_eq!(
        receipt.export_depth_m.to_bits(),
        (receipt.available_depth_m - receipt.depth_m).to_bits()
    );
    Ok(())
}
