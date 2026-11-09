use super::fixture::{Result, parameters, single};
use hydr::runoff::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirInput, ReservoirParameters,
    ReservoirState,
};

#[test]
fn immutable_repeated_trials_preserve_the_accepted_batch() -> Result {
    let batch = single(parameters()?, 0.25 * 0.3048, ReservoirConfig::default())?;
    let before = batch.snapshot_data()?;
    assert!(batch.last_step()?.is_none());
    let input = [ReservoirInput {
        incoming_rate_m_per_s: 0.0,
    }];
    let first = batch.trial_to(1.0, &input)?;
    let repeat = batch.trial_to(1.0, &input)?;
    assert_eq!(first.outputs()?, repeat.outputs()?);
    assert_eq!(first.diagnostics()?, repeat.diagnostics()?);
    assert_eq!(batch.snapshot_data()?, before);
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(batch.trial_to(end, &input).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    let next = first.into_next();
    assert_eq!(next.time(), 1.0);
    assert_eq!(next.accepted_steps(), 1);
    assert_eq!(batch.accepted_steps(), 0);
    assert_eq!(
        next.last_step()?.ok_or("missing accepted step")?,
        repeat.diagnostics()?
    );
    assert!(next.ledger().export_volume_m3 > 0.0);
    Ok(())
}

#[test]
fn mixed_simd_and_tail_layouts_meet_the_same_lane_accuracy_gates() -> Result {
    let p = parameters()?;
    let closed =
        ReservoirParameters::from_native(1024.0, 1024.0, 0.0, 1.0 / 64.0, 1.0 / 32.0, 0.125)?;
    let instant = ReservoirParameters::from_native(1024.0, 1024.0, 32.0, 1.0 / 64.0, 0.0, 0.125)?;
    let seed = [
        (p, 0.25 * 0.3048, 0.0),
        (p, 0.0, 0.3048 / 1024.0),
        (closed, 0.0, 0.3048 / 1024.0),
        (instant, 0.25 * 0.3048, 0.0),
        (p, p.depression_depth_m(), 0.0),
        (closed, 1e-15, 0.0),
    ];
    for length in [31, 32, 33, 64, 65] {
        let lanes: Vec<_> = (0..length)
            .map(|index| seed[(index + length) % seed.len()])
            .collect();
        let elements = lanes
            .iter()
            .enumerate()
            .map(|(index, lane)| ReservoirElement {
                id: format!("storage-{index}"),
                parameters: lane.0,
            })
            .collect();
        let states = lanes
            .iter()
            .map(|lane| ReservoirState { depth_m: lane.1 })
            .collect();
        let inputs: Vec<_> = lanes
            .iter()
            .map(|lane| ReservoirInput {
                incoming_rate_m_per_s: lane.2,
            })
            .collect();
        let batch = ReservoirBatch::new(elements, states, 0.0, ReservoirConfig::default())?;
        let trial = batch.trial_to(1.0, &inputs)?;
        let outputs = trial.outputs()?;
        for (index, (p, initial_depth, rate)) in lanes.into_iter().enumerate() {
            let one = single(p, initial_depth, ReservoirConfig::default())?;
            let reference = one
                .trial_to(
                    1.0,
                    &[ReservoirInput {
                        incoming_rate_m_per_s: rate,
                    }],
                )?
                .outputs()?;
            let actual = &outputs[index];
            let expected = &reference[0];
            let config = batch.config();
            let root_scale = (actual.depth_m - p.depression_depth_m())
                .abs()
                .max((expected.depth_m - p.depression_depth_m()).abs());
            let depth_tolerance = 2.0
                * (config.root_absolute_depth + config.root_relative_depth * root_scale)
                + 8.0 * f64::EPSILON * actual.depth_m.abs().max(expected.depth_m.abs());
            assert!(
                (actual.depth_m - expected.depth_m).abs() <= depth_tolerance,
                "length={length}, element={index}"
            );
            let volume_tolerance = config.volume_absolute
                + config.volume_relative
                    * actual
                        .export_volume_m3
                        .abs()
                        .max(expected.export_volume_m3.abs());
            assert!(
                (actual.export_volume_m3 - expected.export_volume_m3).abs() <= volume_tolerance,
                "length={length}, element={index}"
            );
            match (
                actual.endpoint_discharge_m3_per_s,
                expected.endpoint_discharge_m3_per_s,
            ) {
                (Some(actual), Some(expected)) => {
                    assert!((actual - expected).abs() <= 2.0 * volume_tolerance)
                }
                (None, None) => {}
                _ => return Err("endpoint branch differs between batch layouts".into()),
            }
        }
    }
    Ok(())
}
