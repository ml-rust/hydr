use super::fixture::{Result, parameters, single};
use hydr::runoff::{ReservoirConfig, ReservoirInput, ReservoirStepDiagnostics};

fn check_gates(step: &ReservoirStepDiagnostics, config: &ReservoirConfig) {
    assert!(
        step.constitutive_residual_m3.abs()
            <= config.volume_absolute + config.volume_relative * step.constitutive_scale_m3
    );
    assert!(
        step.water_residual_m3.abs()
            <= config.volume_absolute + config.volume_relative * step.water_scale_m3
    );
    for receipt in &step.receipts {
        assert!(receipt.depth_m >= 0.0 && receipt.depth_m <= receipt.available_depth_m);
        assert!(
            receipt.constitutive_residual_m3.abs()
                <= config.volume_absolute + config.volume_relative * receipt.constitutive_scale_m3
        );
        assert!(
            receipt.water_residual_m3.abs()
                <= config.volume_absolute + config.volume_relative * receipt.water_scale_m3
        );
        let root_tolerance = config.root_absolute_depth
            + config.root_relative_depth
                * receipt.root_lower_m.abs().max(receipt.root_upper_m.abs());
        assert!(receipt.root_lower_m >= 0.0 && receipt.root_upper_m >= receipt.root_lower_m);
        assert!(receipt.root_upper_m - receipt.root_lower_m <= root_tolerance);
    }
}

#[test]
fn equilibrium_retains_depth_under_the_stored_si_coefficient() -> Result {
    let p = parameters()?;
    let depth = 0.25 * 0.3048;
    let rate = p.coefficient_si().ok_or("missing finite coefficient")?
        * (depth - p.depression_depth_m()).powf(1.6666667);
    let batch = single(p, depth, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        64.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: rate,
        }],
    )?;
    check_gates(&trial.diagnostics()?, batch.config());
    let next = trial.into_next();
    assert!((next.states()?[0].depth_m - depth).abs() <= 1e-10);
    assert!((next.ledger().incoming_volume_m3 - next.ledger().export_volume_m3).abs() <= 1e-8);
    Ok(())
}

#[test]
fn recession_refinement_reduces_depth_and_integrated_export_errors() -> Result {
    let p = parameters()?;
    let initial_depth = 0.25 * 0.3048;
    let exponent = 1.6666667;
    let alpha = p.coefficient_si().ok_or("missing finite coefficient")?;
    let initial_excess = initial_depth - p.depression_depth_m();
    let exact_excess = (initial_excess.powf(1.0 - exponent) + (exponent - 1.0) * alpha * 64.0)
        .powf(1.0 / (1.0 - exponent));
    let exact_depth = p.depression_depth_m() + exact_excess;
    let exact_export = p.area_m2() * (initial_depth - exact_depth);
    let exact_endpoint_rate = alpha * exact_excess.powf(exponent);
    // EPA SWMM 5.2.4 recession: eight 8-second steps with the frozen native geometry.
    let native_depth = f64::from_bits(0x3fc3_198f_5648_00ad) * 0.3048;
    let native_endpoint_rate = f64::from_bits(0x3f38_be4e_8dd4_bb9b) * 0.3048;
    assert!((native_depth - exact_depth).abs() <= 1e-8);
    assert!((native_endpoint_rate - exact_endpoint_rate).abs() <= 1e-10);
    let depth_target = 1e-9 + 0.002 * exact_excess.abs();
    let export_target = 1e-9 + 0.002 * exact_export.abs();
    let mut previous = [f64::INFINITY; 2];
    let mut previous_native = [f64::INFINITY; 2];
    for dt in [1.0, 0.5, 0.25, 0.125, 0.0625] {
        let mut batch = single(p, initial_depth, ReservoirConfig::default())?;
        let count = (64.0 / dt) as usize;
        for index in 1..=count {
            let trial = batch.trial_to(
                index as f64 * dt,
                &[ReservoirInput {
                    incoming_rate_m_per_s: 0.0,
                }],
            )?;
            check_gates(&trial.diagnostics()?, batch.config());
            batch = trial.into_next();
        }
        let depth = batch.states()?[0].depth_m;
        let endpoint_rate = batch
            .last_step()?
            .ok_or("missing accepted recession step")?
            .receipts[0]
            .endpoint_rate_m_per_s
            .ok_or("missing finite endpoint rate")?;
        let errors = [
            (depth - exact_depth).abs(),
            (batch.ledger().export_volume_m3 - exact_export).abs(),
        ];
        let native_errors = [
            (depth - native_depth).abs(),
            (endpoint_rate - native_endpoint_rate).abs(),
        ];
        assert!(
            errors[0] < previous[0],
            "dt={dt}, errors={errors:?}, previous={previous:?}"
        );
        assert!(
            errors[1] < previous[1],
            "dt={dt}, errors={errors:?}, previous={previous:?}"
        );
        assert!(
            native_errors[0] < previous_native[0],
            "dt={dt}, native_errors={native_errors:?}, previous={previous_native:?}"
        );
        assert!(
            native_errors[1] < previous_native[1],
            "dt={dt}, native_errors={native_errors:?}, previous={previous_native:?}"
        );
        eprintln!(
            "dt_s={dt:.6} analytical_depth_error_m={:.17e} depth_target_m={depth_target:.17e} analytical_export_error_m3={:.17e} export_target_m3={export_target:.17e} native_depth_difference_m={:.17e} native_endpoint_rate_difference_m_per_s={:.17e}",
            errors[0], errors[1], native_errors[0], native_errors[1]
        );
        previous = errors;
        previous_native = native_errors;
    }
    assert!(
        previous[0] <= depth_target,
        "finest depth error={}",
        previous[0]
    );
    assert!(
        previous[1] <= export_target,
        "finest export error={}",
        previous[1]
    );
    Ok(())
}

#[test]
fn root_width_and_constitutive_volume_have_independent_acceptance_gates() -> Result {
    let p = parameters()?;
    let initial_depth = 0.25 * 0.3048;
    let loose = ReservoirConfig {
        root_absolute_depth: 0.01,
        root_relative_depth: 0.1,
        ..ReservoirConfig::default()
    };
    let batch = single(p, initial_depth, loose)?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_to(
                1.0,
                &[ReservoirInput {
                    incoming_rate_m_per_s: 0.0
                }]
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    let batch = single(p, initial_depth, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        1.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.0,
        }],
    )?;
    check_gates(&trial.diagnostics()?, batch.config());
    let exhausted = ReservoirConfig {
        root_iterations: 1,
        volume_absolute: 1e6,
        volume_relative: 0.1,
        ..ReservoirConfig::default()
    };
    let batch = single(p, initial_depth, exhausted)?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_to(
                1.0,
                &[ReservoirInput {
                    incoming_rate_m_per_s: 0.0
                }]
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn stiff_drainage_keeps_bracket_positivity_and_both_residual_gates() -> Result {
    let p = hydr::runoff::ReservoirParameters::from_native(
        1024.0,
        1024.0,
        32.0 * 1e4,
        1.0 / 64.0,
        1.0 / 32.0,
        0.125,
    )?;
    let batch = single(p, 0.25 * 0.3048, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        64.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.0,
        }],
    )?;
    check_gates(&trial.diagnostics()?, batch.config());
    let output = &trial.outputs()?[0];
    assert!(output.depth_m > p.depression_depth_m());
    assert!(output.depth_m < 0.25 * 0.3048);
    assert!(output.export_volume_m3 > 0.0);
    Ok(())
}
