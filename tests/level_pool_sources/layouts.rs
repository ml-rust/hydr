use super::fixture::{
    Result, changing, curve, element, enabled, exchange, policy, sources, tolerance,
};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolState};

#[test]
fn mixed_source_and_zero_lanes_preserve_independent_curve_and_source_support() -> Result {
    for count in [1, 31, 32, 33, 64, 65] {
        let mut elements = Vec::with_capacity(count);
        let mut states = Vec::with_capacity(count);
        let mut exchanges = Vec::with_capacity(count);
        let mut forcing = Vec::with_capacity(count);
        for index in 0..count {
            let (pool_curve, initial, rates, transfer) = match index % 5 {
                0 => (
                    changing()?,
                    0.0,
                    sources(0.25, 0.0, 0.0),
                    exchange(0.0, 0.0),
                ),
                1 => (
                    changing()?,
                    10.0,
                    sources(0.0, 0.1, 0.0),
                    exchange(0.0, 0.0),
                ),
                2 => (
                    changing()?,
                    30.0,
                    sources(0.25, 0.25, 0.0),
                    exchange(0.0, 0.0),
                ),
                3 => (
                    curve(&[(0.0, 100.0), (20.0, 102.0)])?,
                    5.0,
                    sources(0.0, 0.0, 0.0),
                    exchange(0.2, 0.1),
                ),
                _ => (
                    curve(&[(0.0, 100.0), (20.0, 102.0)])?,
                    2.0,
                    sources(0.2, 0.1, 0.5),
                    exchange(0.0, 0.0),
                ),
            };
            elements.push(element(&format!("pool-{index}"), pool_curve));
            states.push(LevelPoolState { volume_m3: initial });
            exchanges.push(transfer);
            forcing.push(rates);
        }
        let batch = LevelPoolBatch::<crate::runtime::Runtime>::new_with_sources(
            crate::runtime::client(),
            elements.clone(),
            states.clone(),
            0.0,
            LevelPoolConfig::default(),
            vec![policy(); count],
        )?;
        let trial = batch.trial_with_sources_to(2.0, &exchanges, &forcing)?;
        let outputs = trial.outputs()?;
        let diagnostic = trial.diagnostics()?;
        for index in 0..count {
            let reference = enabled(elements[index].curve.clone(), states[index].volume_m3)?
                .trial_with_sources_to(2.0, &[exchanges[index]], &[forcing[index]])?
                .outputs()?;
            let allowance_m3 = tolerance(outputs[index].volume_m3.max(reference[0].volume_m3));
            assert!((outputs[index].volume_m3 - reference[0].volume_m3).abs() <= allowance_m3);
            match (outputs[index].stage_m, reference[0].stage_m) {
                (Some(actual), Some(expected)) => {
                    assert!((actual - expected).abs() <= allowance_m3 / 10.0)
                }
                (None, None) => {}
                _ => return Err("source stage support differs between layouts".into()),
            }
            assert_eq!(diagnostic.receipts[index].source.is_none(), index % 5 == 3);
        }
    }
    Ok(())
}

#[test]
fn repeated_source_trials_leave_the_complete_accepted_state_immutable() -> Result {
    let batch = enabled(changing()?, 5.0)?;
    let before = batch.snapshot_data()?;
    let first =
        batch.trial_with_sources_to(1.0, &[exchange(0.2, 0.1)], &[sources(0.25, 0.1, 0.5)])?;
    let repeated =
        batch.trial_with_sources_to(1.0, &[exchange(0.2, 0.1)], &[sources(0.25, 0.1, 0.5)])?;
    assert_eq!(first.outputs()?, repeated.outputs()?);
    assert_eq!(first.diagnostics()?, repeated.diagnostics()?);
    assert_eq!(batch.snapshot_data()?, before);
    assert!(batch.source_policies().is_some());
    Ok(())
}
