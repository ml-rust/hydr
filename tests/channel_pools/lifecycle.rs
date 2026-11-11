use super::fixture::{Result, batch, check_water, forcing, single};
use super::reference::CASES;
use hydr::channel::{
    ChannelBoundary, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
};
use hydr::coupling::{ChannelPoolEndpoint, ChannelPoolError};
use hydr::forcing::WaterAmount;

#[test]
fn shortened_common_intervals_recompute_every_integrated_input_including_exact_zero() -> Result {
    for volume in [0.0, 0.0001] {
        let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
        setup.config.maximum_step_s = 0.0625;
        setup.reaches[0]
            .definition
            .lateral_bindings
            .push(ChannelLateralBinding {
                boundary_id: "source".into(),
                cell_id: "cell-0".into(),
                direction: ChannelLateralDirection::Inflow,
                ownership_assertion: "Independent complete lateral input".into(),
            });
        let origin = batch(setup.clone(), initial)?;
        let before = origin.snapshot_data()?;
        let mut held = forcing(&setup, 0.0, 0.125);
        let input = |amount| ChannelLateralInput::Inflow {
            boundary_id: "source".into(),
            amount,
            longitudinal_velocity_m_per_s: 0.0,
        };
        held.reaches[0].lateral_inputs = vec![input(WaterAmount::Rate {
            rate_m3_per_s: volume / 0.125,
        })];
        let rate = origin.trial_to(0.125, &held)?.into_next();
        assert_eq!(rate.time(), 0.0625);
        check_water(&rate)?;
        held.reaches[0].lateral_inputs =
            vec![input(WaterAmount::IntervalVolume { volume_m3: volume })];
        let error = origin
            .trial_to(0.125, &held)
            .err()
            .ok_or("Integrated input silently shortens")?;
        let ChannelPoolError::RecomputeInterval {
            start_s,
            requested_end_s,
            suggested_end_s,
            ..
        } = error
        else {
            return Err(format!("Unexpected exact-span error: {error}").into());
        };
        assert_eq!(start_s.to_bits(), 0.0_f64.to_bits());
        assert_eq!(requested_end_s, 0.125);
        assert!(suggested_end_s > 0.0 && suggested_end_s <= 0.0625);
        assert_eq!(origin.snapshot_data()?, before);
        held.end_s = suggested_end_s;
        held.reaches[0].lateral_inputs = vec![input(WaterAmount::IntervalVolume {
            volume_m3: volume / 0.125 * suggested_end_s,
        })];
        let exact = origin.trial_to(suggested_end_s, &held)?.into_next();
        assert_eq!(exact.time().to_bits(), suggested_end_s.to_bits());
        check_water(&exact)?;
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn rejected_common_trial_rolls_back_every_reach_pool_and_connection_counter() -> Result {
    let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    let mut second = setup.reaches[0].clone();
    second.id = "second".into();
    setup.reaches.push(second);
    initial.reaches.push(initial.reaches[0].clone());
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let mut held = forcing(&setup, 0.0, 0.125);
    held.reaches[1].right = Some(ChannelBoundary::Stage {
        elevation_m: f64::NAN,
    });
    assert!(origin.trial_to(0.125, &held).is_err());
    assert_eq!(origin.snapshot_data()?, before);
    let accepted = origin
        .trial_to(0.125, &forcing(&setup, 0.0, 0.125))?
        .into_next();
    let saved = accepted.snapshot_data()?;
    held = forcing(&setup, 0.125, 0.25);
    held.reaches[1].right = Some(ChannelBoundary::Discharge {
        discharge_m3_per_s: 1e100,
    });
    assert!(accepted.trial_to(0.25, &held).is_err());
    assert_eq!(accepted.snapshot_data()?, saved);
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn bounded_whole_interval_retry_recalculates_finite_feedback_without_scaling_transfers() -> Result {
    let case = CASES
        .iter()
        .find(|case| case.name == "triangular-inflow-ideal" && case.pool_area == 1.0)
        .copied()
        .ok_or("Missing frozen triangular inflow retry fixture")?;
    let (mut setup, initial) = single(case, ChannelPoolEndpoint::Right)?;
    setup.config.iteration_limit = 24;
    setup.config.retry_limit = 16;
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let current = origin
        .trial_to(0.25, &forcing(&setup, 0.0, 0.25))?
        .into_next();
    assert!(current.time() > 0.0 && current.time() < 0.25);
    let step = current
        .last_step()?
        .ok_or("Missing retried finite-pool receipt")?;
    assert!(step.retries > 0);
    let fresh = origin
        .trial_to(current.time(), &forcing(&setup, 0.0, current.time()))?
        .into_next();
    assert_eq!(
        current
            .reach("reach")
            .ok_or("Missing retried reach")?
            .state()?,
        fresh.reach("reach").ok_or("Missing fresh reach")?.state()?
    );
    assert_eq!(current.pools().states()?, fresh.pools().states()?);
    let retried_transfer = &step.transfers[0];
    let fresh_step = fresh
        .last_step()?
        .ok_or("Missing recomputed finite-pool receipt")?;
    assert_eq!(retried_transfer, &fresh_step.transfers[0]);
    check_water(&current)?;
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn common_duration_limits_intersect_parent_and_every_reach_without_changing_the_requested_horizon()
-> Result {
    for (parent_maximum, child_maximum, second_maximum, expected) in [
        (0.125, 0.0625, 0.125, 0.0625),
        (0.0625, 0.125, 0.125, 0.0625),
        (0.125, 0.125, 0.03125, 0.03125),
    ] {
        let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
        setup.config.maximum_step_s = parent_maximum;
        setup.reaches[0].definition.config.max_step = child_maximum;
        let mut second = setup.reaches[0].clone();
        second.id = "second".into();
        second.definition.config.max_step = second_maximum;
        setup.reaches.push(second);
        initial.reaches.push(initial.reaches[0].clone());
        let origin = batch(setup.clone(), initial)?;
        let before = origin.snapshot_data()?;
        let next = origin
            .trial_to(0.25, &forcing(&setup, 0.0, 0.25))?
            .into_next();
        assert_eq!(next.time(), expected);
        assert_eq!(
            next.reach("reach").ok_or("Missing limited reach")?.time(),
            expected
        );
        assert_eq!(
            next.reach("second")
                .ok_or("Missing second limited reach")?
                .time(),
            expected
        );
        let step = next.last_step()?.ok_or("Missing common duration receipt")?;
        assert_eq!(step.requested_end_s, 0.25);
        assert_eq!(step.end_s, expected);
        assert_eq!(step.retries, 0);
        check_water(&next)?;
        assert_eq!(origin.snapshot_data()?, before);
    }
    for parent_is_tighter in [false, true] {
        let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
        if parent_is_tighter {
            setup.config.maximum_step_s = 0.0625;
            setup.reaches[0].definition.config.min_step = 0.125;
        } else {
            setup.config.minimum_step_s = 0.125;
            setup.reaches[0].definition.config.max_step = 0.0625;
        }
        assert!(
            batch(setup, initial).is_err(),
            "Disjoint common duration range enters accepted state"
        );
    }
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition.config.min_step = 0.0625;
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    assert!(matches!(
        origin.trial_to(0.03125, &forcing(&setup, 0.0, 0.03125)),
        Err(ChannelPoolError::Unresolved { .. })
    ));
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn child_duration_cap_requires_complete_recomputation_of_an_exact_zero_lateral_volume() -> Result {
    let (mut setup, initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.reaches[0].definition.config.max_step = 0.0625;
    setup.reaches[0]
        .definition
        .lateral_bindings
        .push(ChannelLateralBinding {
            boundary_id: "source".into(),
            cell_id: "cell-0".into(),
            direction: ChannelLateralDirection::Inflow,
            ownership_assertion: "Explicit exact zero input on the common span".into(),
        });
    let origin = batch(setup.clone(), initial)?;
    let before = origin.snapshot_data()?;
    let mut held = forcing(&setup, 0.0, 0.125);
    held.reaches[0].lateral_inputs = vec![ChannelLateralInput::Inflow {
        boundary_id: "source".into(),
        amount: WaterAmount::IntervalVolume { volume_m3: 0.0 },
        longitudinal_velocity_m_per_s: 0.0,
    }];
    let error = origin
        .trial_to(0.125, &held)
        .err()
        .ok_or("Child duration cap silently shortens an exact zero input")?;
    let ChannelPoolError::RecomputeInterval {
        start_s,
        requested_end_s,
        suggested_end_s,
        ..
    } = error
    else {
        return Err(format!("Unexpected child duration error: {error}").into());
    };
    assert_eq!(start_s, 0.0);
    assert_eq!(requested_end_s, 0.125);
    assert_eq!(suggested_end_s, 0.0625);
    assert_eq!(origin.snapshot_data()?, before);
    held.end_s = suggested_end_s;
    let next = origin.trial_to(suggested_end_s, &held)?.into_next();
    assert_eq!(next.time(), suggested_end_s);
    check_water(&next)?;
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}
