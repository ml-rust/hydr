use super::fixture::{Result, constant, curve, element, input, multiple, single, unit};
use hydr::storage::{LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolState};

#[test]
fn constant_and_multiple_segments_return_exact_binary_stages_and_knots() -> Result {
    for (curve, queries) in [
        (constant()?, vec![(1.0, 10.5_f64), (2.0, 11.0), (4.0, 12.0)]),
        (
            multiple()?,
            vec![(1.0, 10.5), (2.0, 11.0), (5.0, 12.0), (8.0, 13.0)],
        ),
    ] {
        for (volume, stage) in queries {
            let batch = single(curve.clone(), volume)?;
            let output = batch
                .outputs()?
                .into_iter()
                .next()
                .ok_or("missing pool output")?;
            assert_eq!(output.volume_m3.to_bits(), volume.to_bits());
            assert_eq!(
                output.stage_m.ok_or("missing wet stage")?.to_bits(),
                stage.to_bits()
            );
        }
        assert!(single(curve.clone(), -f64::from_bits(1)).is_err());
        assert!(single(curve.clone(), curve.maximum_volume_m3().next_up()).is_err());
        let dry = single(curve, 0.0)?;
        assert_eq!(dry.outputs()?[0].stage_m, None);
    }
    Ok(())
}

#[test]
fn malformed_or_unrepresentable_curves_reject_without_extrapolation() -> Result {
    for knots in [
        vec![],
        vec![(0.0, 0.0)],
        vec![(1.0, 0.0), (2.0, 1.0)],
        vec![(0.0, 0.0), (0.0, 1.0)],
        vec![(0.0, 0.0), (1.0, 0.0)],
        vec![(0.0, 1.0), (1.0, 0.0)],
        vec![(0.0, 0.0), (-1.0, 1.0)],
        vec![(0.0, 0.0), (f64::INFINITY, 1.0)],
        vec![(0.0, 0.0), (1.0, f64::NAN)],
        vec![(0.0, -f64::MAX), (1.0, f64::MAX)],
        vec![(0.0, 0.0), (f64::MAX, f64::from_bits(1))],
        vec![(0.0, 0.0), (f64::from_bits(1), f64::MAX)],
    ] {
        assert!(curve(&knots).is_err(), "knots={knots:?}");
    }
    let knots = constant()?.knots().to_vec();
    assert!(LevelPoolCurve::new(knots.clone(), String::new(), "explicit table".into()).is_err());
    assert!(LevelPoolCurve::new(knots, "datum".into(), String::new()).is_err());
    Ok(())
}

#[test]
fn identities_connected_assertions_and_complete_states_remain_explicit() -> Result {
    let base = element("pool", constant()?);
    for mode in 0..5 {
        let mut elements = vec![base.clone()];
        match mode {
            0 => elements[0].id.clear(),
            1 => elements[0].connected_pool_assertion.clear(),
            2 => elements[0].storage_support_identity.clear(),
            3 => elements.push(base.clone()),
            4 => {
                let mut other = base.clone();
                other.id = "other".into();
                elements.push(other);
            }
            _ => return Err("unknown identity fixture".into()),
        }
        let states = vec![LevelPoolState { volume_m3: 0.0 }; elements.len()];
        assert!(LevelPoolBatch::new(elements, states, 0.0, LevelPoolConfig::default()).is_err());
    }
    assert!(
        LevelPoolBatch::new(vec![base.clone()], vec![], 0.0, LevelPoolConfig::default()).is_err()
    );
    assert!(LevelPoolBatch::new(vec![], vec![], 0.0, LevelPoolConfig::default()).is_err());
    assert!(
        LevelPoolBatch::new(
            vec![base],
            vec![LevelPoolState {
                volume_m3: f64::NAN
            }],
            0.0,
            LevelPoolConfig::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn large_datum_rejects_unresolved_wet_stage_atomically() -> Result {
    let batch = single(curve(&[(0.0, 1e16), (2.0, 1e16 + 2.0)])?, 0.0)?;
    let before = batch.snapshot_data()?;
    assert!(batch.trial_to(1.0, &[input(0.5, 0.0)]).is_err());
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn minimum_quantum_remains_positive_and_a_rounded_stage_plateau_is_admitted() -> Result {
    let quantum = f64::from_bits(1);
    for maximum in [1.0, quantum] {
        let batch = single(unit(maximum)?, quantum)?;
        let output = &batch.outputs()?[0];
        assert_eq!(output.volume_m3.to_bits(), quantum.to_bits());
        assert_eq!(
            output
                .stage_m
                .ok_or("missing minimum-quantum stage")?
                .to_bits(),
            quantum.to_bits()
        );
    }
    let batch = single(constant()?, 1.0)?;
    let trial = batch.trial_to(1.0, &[input(1e-15, 0.0)])?;
    let output = &trial.outputs()?[0];
    assert!(output.volume_m3 > 1.0);
    assert_eq!(output.stage_m, batch.outputs()?[0].stage_m);
    Ok(())
}
