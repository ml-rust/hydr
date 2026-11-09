use super::fixture::{Result, changing, enabled, exchange, sources, tolerance};

#[test]
fn changing_area_sources_meet_frozen_first_order_envelopes_and_finest_targets() -> Result {
    // Targets come from the exact decimal piecewise-area equations, independent of this operator.
    for (name, initial, rates, exact, factors) in [
        (
            "filling",
            9.1,
            sources(0.3, 0.1, 0.5),
            [14.9, 101.245, 10.2, 3.4, 1.0],
            [2.0, 0.2, 3.0, 1.0],
        ),
        (
            "draining",
            12.1,
            sources(0.0, 0.1, 0.5),
            [8.26, 100.826, 0.0, 2.84, 1.0],
            [1.0, 0.1, 0.0, 1.0],
        ),
        (
            "rain_only",
            9.1,
            sources(0.2, 0.0, 0.0),
            [16.2, 101.31, 7.1, 0.0, 0.0],
            [2.0, 0.2, 2.0, 0.0],
        ),
        (
            "evaporation_only",
            12.1,
            sources(0.0, 0.1, 0.0),
            [9.05, 100.905, 0.0, 3.05, 0.0],
            [1.0, 0.1, 0.0, 1.0],
        ),
    ] {
        let mut previous_error: Option<f64> = None;
        for dt in [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125] {
            let mut batch = enabled(changing()?, initial)?;
            for step in 1..=(2.0 / dt) as usize {
                let trial = batch.trial_with_sources_to(
                    step as f64 * dt,
                    &[exchange(0.0, 0.0)],
                    &[rates],
                )?;
                let diagnostic = trial.diagnostics()?;
                assert!(
                    diagnostic.water_residual_m3.abs() <= tolerance(diagnostic.physical_scale_m3)
                );
                batch = trial.into_next();
            }
            let ledger = batch.element_ledgers()?;
            let process = ledger[0]
                .sources
                .as_ref()
                .ok_or("missing source counters")?;
            let output = &batch.outputs()?[0];
            let actual = [
                output.volume_m3,
                output.stage_m.ok_or("missing reference wet stage")?,
                process.cumulative_rain_volume_m3,
                process.cumulative_evaporation_volume_m3,
                process.cumulative_bottom_loss_volume_m3,
            ];
            let scale = initial
                .max(actual[0])
                .max(actual[2])
                .max(actual[3])
                .max(actual[4]);
            let allowance_m3 = tolerance(scale);
            let errors =
                std::array::from_fn::<_, 5, _>(|index| (actual[index] - exact[index]).abs());
            assert!(
                errors[0] <= factors[0] * dt + allowance_m3,
                "case={name},dt={dt},volume error={}",
                errors[0]
            );
            assert!(
                errors[1] <= factors[1] * dt + allowance_m3 / 10.0,
                "case={name},dt={dt},stage error={}",
                errors[1]
            );
            assert!(errors[2] <= factors[2] * dt + allowance_m3);
            assert!(errors[3] <= factors[3] * dt + allowance_m3);
            assert!(errors[4] <= allowance_m3);
            if dt == 0.0078125 {
                assert!(
                    errors[1] <= 0.0005,
                    "case={name},finest stage error={}",
                    errors[1]
                );
                for index in [0, 2, 3, 4] {
                    assert!(
                        errors[index] <= 0.01,
                        "case={name},quantity={index},error={}",
                        errors[index]
                    );
                }
            }
            let plateau =
                previous_error.is_some_and(|previous| (errors[0] - previous).abs() <= allowance_m3);
            eprintln!(
                "case={name},dt_s={dt:.7},volume_m3={:.17e},stage_m={:.17e},rain_m3={:.17e},evaporation_m3={:.17e},bottom_loss_m3={:.17e},errors={errors:?},plateau={plateau}",
                actual[0], actual[1], actual[2], actual[3], actual[4]
            );
            previous_error = Some(errors[0]);
        }
    }
    Ok(())
}
