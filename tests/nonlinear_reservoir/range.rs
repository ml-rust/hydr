use super::fixture::{Result, single};
use hydr::runoff::{ReservoirConfig, ReservoirError, ReservoirInput, ReservoirParameters};

#[test]
fn large_depression_cannot_absorb_a_positive_drainage_root() -> Result {
    let depression = 1e6_f64;
    let p = ReservoirParameters::from_si(1.0, 1.0, 1.0, 1.0, 1e-8, depression)?;
    let batch = single(p, depression.next_up(), ReservoirConfig::default())?;
    let before = batch.snapshot_data()?;
    let error = batch
        .trial_to(
            1.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: 0.0,
            }],
        )
        .err()
        .ok_or("positive reconstructed excess requires representation")?;
    assert!(matches!(error, ReservoirError::Numerical { .. }));
    assert_eq!(error.element(), Some(0));
    assert!(error.to_string().contains("represented implicit excess"));
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn closed_positive_subnormal_films_remain_represented() -> Result {
    let p = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.01, 0.03, 1.0)?;
    let film = f64::from_bits(1);
    let batch = single(p, film, ReservoirConfig::default())?;
    let trial = batch.trial_to(
        1.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.0,
        }],
    )?;
    let output = &trial.outputs()?[0];
    assert_eq!(output.depth_m.to_bits(), film.to_bits());
    assert_eq!(output.storage_m3.to_bits(), film.to_bits());
    assert_eq!(output.export_volume_m3, 0.0);
    Ok(())
}

#[test]
fn positive_incoming_depth_volume_and_represented_supply_cannot_disappear() -> Result {
    let closed = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.01, 0.03, 1.0)?;
    let tiny_area = ReservoirParameters::from_si(1e-200, 1e-200, 0.0, 0.01, 0.03, 1.0)?;
    for (p, initial, end, rate) in [
        (closed, 0.0, 0.5, f64::from_bits(1)),
        (tiny_area, 0.0, 1.0, 1e-200),
        (closed, 1.0, 1.0, f64::EPSILON / 4.0),
    ] {
        let batch = single(p, initial, ReservoirConfig::default())?;
        let before = batch.snapshot_data()?;
        assert!(
            batch
                .trial_to(
                    end,
                    &[ReservoirInput {
                        incoming_rate_m_per_s: rate
                    }]
                )
                .is_err()
        );
        assert_eq!(batch.snapshot_data()?, before);
    }
    assert!(single(tiny_area, 1e-200, ReservoirConfig::default()).is_err());
    Ok(())
}

#[test]
fn overflowing_supply_and_active_powers_reject_without_mutation() -> Result {
    let closed = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.01, 0.03, 0.0)?;
    let batch = single(closed, 0.0, ReservoirConfig::default())?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_to(
                2.0,
                &[ReservoirInput {
                    incoming_rate_m_per_s: f64::MAX
                }]
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    let active = ReservoirParameters::from_si(1.0, 1.0, 1.0, 0.01, 0.03, 0.0)?;
    let batch = single(active, 1e200, ReservoirConfig::default())?;
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
