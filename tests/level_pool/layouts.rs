use super::fixture::{Result, constant, element, input, multiple, single, tolerance, unit};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolState};

#[test]
fn shared_and_heterogeneous_curve_layouts_match_independent_element_execution() -> Result {
    for count in [1, 31, 32, 33, 64, 65] {
        for shared in [true, false] {
            let mut elements = Vec::with_capacity(count);
            let mut states = Vec::with_capacity(count);
            let mut inputs = Vec::with_capacity(count);
            for index in 0..count {
                let (curve, volume, exchange) = match if shared { 0 } else { index % 3 } {
                    0 => (constant()?, 1.0, input(0.2, 0.1)),
                    1 => (multiple()?, 5.0, input(1.0, 0.5)),
                    _ => (unit(1.0)?, 0.0, input(f64::from_bits(1), 0.0)),
                };
                elements.push(element(&format!("pool-{index}"), curve));
                states.push(LevelPoolState { volume_m3: volume });
                inputs.push(exchange);
            }
            let batch = LevelPoolBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                elements.clone(),
                states.clone(),
                0.0,
                LevelPoolConfig::default(),
            )?;
            let trial = batch.trial_to(1.0, &inputs)?;
            let outputs = trial.outputs()?;
            let diagnostics = trial.diagnostics()?;
            for (index, output) in outputs.iter().enumerate() {
                let reference = single(elements[index].curve.clone(), states[index].volume_m3)?
                    .trial_to(1.0, &[inputs[index]])?
                    .outputs()?;
                assert_eq!(output.element, index);
                assert_eq!(output.volume_m3.to_bits(), reference[0].volume_m3.to_bits());
                match (output.stage_m, reference[0].stage_m) {
                    (Some(actual), Some(expected)) => {
                        // Every fixture has secant area at least one square metre.
                        let stage_tolerance_m =
                            tolerance(output.volume_m3.max(reference[0].volume_m3));
                        assert!((actual - expected).abs() <= stage_tolerance_m);
                    }
                    (None, None) => {}
                    _ => return Err("stage support differs between independent layouts".into()),
                }
                let receipt = &diagnostics.receipts[index];
                assert!(receipt.water_residual_m3.abs() <= tolerance(receipt.physical_scale_m3));
                assert!(
                    receipt.witness_consistency_m3.abs() <= receipt.witness_consistency_bound_m3
                );
            }
            assert!(
                diagnostics.water_residual_m3.abs() <= tolerance(diagnostics.physical_scale_m3)
            );
        }
    }
    Ok(())
}

#[test]
fn repeated_trials_and_rejected_targets_preserve_every_accepted_tensor_projection() -> Result {
    let batch = single(constant()?, 1.0)?;
    let before = batch.snapshot_data()?;
    let first = batch.trial_to(1.0, &[input(0.2, 0.1)])?;
    let repeated = batch.trial_to(1.0, &[input(0.2, 0.1)])?;
    assert_eq!(first.outputs()?, repeated.outputs()?);
    assert_eq!(first.diagnostics()?, repeated.diagnostics()?);
    assert_eq!(batch.snapshot_data()?, before);
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(batch.trial_to(end, &[input(0.2, 0.1)]).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    let accepted = first.into_next();
    assert_eq!(accepted.time(), 1.0);
    assert_eq!(accepted.accepted_steps(), 1);
    assert_eq!(batch.accepted_steps(), 0);
    Ok(())
}
