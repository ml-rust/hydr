use super::fixture::*;
use hydr::structures::*;

#[test]
fn dyadic_switches_and_reversal_enclose_frozen_physical_positions_across_layouts() -> Result {
    for width in [1, 31, 32, 33, 64, 65] {
        let mut config = Config::switching();
        config.width = width;
        config.end = 6.0;
        config.rates = (0.5, 0.25);
        config.maximum_gap = 3.0;
        config.lateness = 0.0;
        let mut batch = config.build()?;
        for (time, stage, exact) in [
            (0.5, None, 0.0),
            (1.0, Some(2.0), 0.0),
            (1.5, None, 0.25),
            (2.0, Some(1.0), 0.5),
            (2.5, None, 0.375),
            (3.0, Some(2.0), 0.25),
            (4.0, None, 0.75),
            (4.5, None, 1.0),
            (6.0, None, 1.0),
        ] {
            batch = accept(batch, time, stage)?;
            position(&batch, exact)?;
            assert_eq!(
                batch.controls().time().to_bits(),
                batch.actuators().time().to_bits()
            );
            assert_eq!(
                batch.controls().accepted_steps(),
                batch.actuators().accepted_steps()
            );
        }
        for (control, actor) in batch
            .controls()
            .states()?
            .iter()
            .zip(batch.actuators().states()?)
        {
            assert_eq!(control.transition_count, 3);
            assert_eq!(actor.target_change_count, 3);
            assert_eq!(actor.arrival_count, 1);
        }
    }
    Ok(())
}

#[test]
fn sampled_qualification_delays_motion_until_the_qualified_endpoint() -> Result {
    let mut config = Config::switching();
    config.times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    config.end = 8.0;
    config.delays = (2.0, 1.0);
    config.rates = (0.5, 0.25);
    config.maximum_gap = 3.0;
    config.lateness = 0.0;
    let mut batch = config.build()?;
    for (time, stage, latch, pending, changes, exact) in [
        (
            1.0,
            2.0,
            ControlLatch::Low,
            Some((ControlLatch::High, 1.0)),
            0,
            0.0,
        ),
        (
            2.0,
            2.0,
            ControlLatch::Low,
            Some((ControlLatch::High, 1.0)),
            0,
            0.0,
        ),
        (3.0, 2.0, ControlLatch::High, None, 1, 0.0),
        (
            4.0,
            1.0,
            ControlLatch::High,
            Some((ControlLatch::Low, 4.0)),
            1,
            0.5,
        ),
        (5.0, 1.0, ControlLatch::Low, None, 2, 1.0),
    ] {
        batch = accept(batch, time, Some(stage))?;
        let state = batch.controls().states()?.remove(0);
        assert_eq!(state.latch, latch);
        assert_eq!(
            state.pending.map(|value| (value.target, value.since_s)),
            pending
        );
        assert_eq!(batch.actuators().states()?[0].target_change_count, changes);
        position(&batch, exact)?;
    }
    batch = accept(batch, 6.0, None)?;
    position(&batch, 0.75)?;
    batch = accept(batch, 8.0, None)?;
    position(&batch, 0.25)
}

#[test]
fn coincident_closure_and_decision_complete_old_motion_before_the_new_command() -> Result {
    let mut config = Config::switching();
    config.opening = 1.0;
    config.times = vec![2.0, 4.0];
    config.end = 6.0;
    config.maximum_gap = 2.0;
    config.lateness = 0.0;
    let batch = config.build()?;
    let event = batch.next_event()?.ok_or("Missing coincident event")?;
    assert_eq!(event.time_s, 2.0);
    assert_eq!(event.decision_index, Some(0));
    assert!(event.arrival_lower_boundary);
    assert!(event.arrival_upper_boundary);
    let mut observations = samples(1, 2.0, 2.0);
    observations[0].available_at_s = 1.5;
    let trial = batch.trial_to(2.0, &observations)?;
    let actor_step = trial
        .actuators()
        .last_step()?
        .ok_or("Missing actuator trial receipt")?;
    assert_eq!(actor_step.receipts[0].left_position.opening_m, 0.0);
    assert!(actor_step.receipts[0].completed_episode);
    assert!(actor_step.receipts[0].target_changed);
    let packet = actor_step.receipts[0]
        .update
        .as_ref()
        .ok_or("Missing computed command")?;
    assert_eq!(packet.issued_at_s.to_bits(), 2.0_f64.to_bits());
    assert_eq!(packet.available_at_s.to_bits(), 2.0_f64.to_bits());
    assert_eq!(
        trial.controls().states()?[0]
            .last_sample
            .as_ref()
            .ok_or("Missing sample")?
            .available_at_s,
        1.5
    );
    let batch = accept(trial.into_next(), 3.0, None)?;
    position(&batch, 0.25)?;
    let batch = accept(batch, 4.0, Some(1.5))?;
    position(&batch, 0.5)?;
    let batch = accept(batch, 6.0, None)?;
    position(&batch, 1.0)?;
    assert!(batch.next_event()?.is_none());
    assert_eq!(batch.actuators().states()?[0].arrival_count, 2);
    Ok(())
}

#[test]
fn equal_signed_zero_commands_preserve_active_target_and_retain_selected_packet_bits() -> Result {
    let mut config = Config::switching();
    config.target = -0.0;
    config.low = -0.0;
    config.high = 0.0;
    config.rates = (0.0, 0.0);
    config.times = vec![1.0, 2.0];
    config.end = 3.0;
    config.maximum_gap = 1.0;
    config.lateness = 0.0;
    let mut batch = config.build()?;
    let initial_anchor = batch.actuators().states()?[0].anchor;
    for (time, stage, raw) in [(1.0, 2.0, 0.0_f64), (2.0, 1.0, -0.0_f64)] {
        batch = accept(batch, time, Some(stage))?;
        let actor = batch.actuators().states()?.remove(0);
        assert_eq!(actor.target_opening_m.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(actor.target_change_count, 0);
        assert_eq!(actor.anchor, initial_anchor);
        assert_eq!(
            actor.anchor.time_s.to_bits(),
            initial_anchor.time_s.to_bits()
        );
        assert_eq!(
            actor.anchor.position.opening_m.to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(
            actor
                .latest_update
                .ok_or("Missing repeated target")?
                .target_opening_m
                .to_bits(),
            raw.to_bits()
        );
    }
    assert_eq!(batch.controls().states()?[0].transition_count, 2);
    let mut imported = config.build()?;
    imported.restore_data(&batch.snapshot_data()?)?;
    let mut trusted = config.build()?;
    trusted.restore(&batch.snapshot())?;
    for restored in [imported, trusted] {
        let actor = restored.actuators().states()?.remove(0);
        assert_eq!(actor.target_opening_m.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(
            actor.anchor.position.opening_m.to_bits(),
            (-0.0_f64).to_bits()
        );
        assert_eq!(actor.position.opening_m.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(
            actor
                .latest_update
                .ok_or("Missing restored repeated packet")?
                .target_opening_m
                .to_bits(),
            (-0.0_f64).to_bits()
        );
    }
    Ok(())
}

#[test]
fn first_target_change_retains_the_distinct_signed_zero_initial_target() -> Result {
    let mut config = Config::switching();
    config.target = -0.0;
    let batch = accept(config.build()?, 1.0, Some(2.0))?;
    let actor = batch.actuators().states()?.remove(0);
    assert_eq!(actor.target_change_count, 1);
    let change = actor.last_target_change.ok_or("Missing first change")?;
    assert_eq!(
        change.previous_target_opening_m.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(change.update.target_opening_m.to_bits(), 1.0_f64.to_bits());
    let mut imported = config.build()?;
    imported.restore_data(&batch.snapshot_data()?)?;
    assert_eq!(imported.snapshot_data()?, batch.snapshot_data()?);
    let restored = imported.actuators().states()?.remove(0);
    assert_eq!(restored.target_opening_m.to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        restored
            .last_target_change
            .ok_or("Missing restored first change")?
            .previous_target_opening_m
            .to_bits(),
        (-0.0_f64).to_bits()
    );
    Ok(())
}

#[test]
fn rejected_old_target_endpoint_discards_a_successful_new_control_decision() -> Result {
    let eta = f64::from_bits(1);
    let mut config = Config::switching();
    config.opening = eta;
    config.rates = (eta, eta);
    config.times = vec![0.5];
    config.end = 2.0;
    config.maximum_gap = 1.5;
    config.lateness = 0.0;
    config.accuracy.maximum_absolute_opening_error_m = 2.0 * eta;
    config.accuracy.maximum_relative_opening_error = 0.0;
    let batch = config.build()?;
    let before = batch.snapshot_data()?;
    let observations = samples(1, 0.5, 2.0);
    let control_candidate = batch.controls().trial_to(0.5, &observations)?.into_next();
    assert_eq!(control_candidate.states()?[0].latch, ControlLatch::High);
    assert_eq!(control_candidate.states()?[0].transition_count, 1);
    let error = rejected(batch.trial_to(0.5, &observations))?;
    assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
    assert_eq!(batch.snapshot_data()?, before);
    assert_eq!(batch.time().to_bits(), 0.0_f64.to_bits());
    assert_eq!(batch.accepted_steps(), 0);
    Ok(())
}

#[test]
fn operating_sources_consume_actual_child_position_without_substituting_its_target() -> Result {
    let config = Config::switching();
    let context = config.context()?;
    let batch = ControlledGateBatch::new(config.controls(&context)?, config.actuators(&context)?)?;
    let registry = context.with_operating_sources(vec![OperatingSourceBinding::Actuated {
        source_id: "controlled-response".into(),
        actuator: batch.actuators(),
    }])?;
    let batch = accept(batch, 1.0, Some(2.0))?;
    let heads = vec![StructureHeads {
        structure_id: device(0),
        endpoint_a: EndpointHead {
            id: "pool".into(),
            support: HeadSupport::Unknown,
        },
        endpoint_b: EndpointHead {
            id: "river".into(),
            support: HeadSupport::Unknown,
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }];
    let proposal = registry.propose_at(
        1.0,
        &heads,
        &[],
        &[OperatingSourceFrame::ActuatedGate {
            source_id: "controlled-response",
            actuator: batch.actuators(),
        }],
    )?;
    assert_eq!(
        proposal.operations()?[0].operating_point,
        StructureOperatingPoint::GateOpening { opening_m: 0.0 }
    );
    assert_eq!(batch.actuators().states()?[0].target_opening_m, 1.0);
    let batch = accept(batch, 1.5, None)?;
    let error = rejected(registry.propose_at(
        1.5,
        &heads,
        &[],
        &[OperatingSourceFrame::ActuatedGate {
            source_id: "controlled-response",
            actuator: batch.actuators(),
        }],
    ))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::UnknownSupport)
    );
    Ok(())
}
