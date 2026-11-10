use super::fixture::*;
use hydr::structures::*;

#[test]
fn ideal_response_applies_selected_stopped_and_running_operations_on_the_decision_right() -> Result
{
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let config = Config::switching();
    let batch = config.build(&context, &["pump"])?;
    let registry = registry(&context, &batch)?;
    let stopped = batch.outputs()?.remove(0);
    assert_eq!(
        stopped.operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    assert_eq!(stopped.model, PumpResponseModel::IdealInstantaneous);
    assert_eq!(stopped.accepted_steps, 0);
    let batch = accept(batch, 1.0, Some(2.0))?;
    let running = batch.outputs()?.remove(0);
    assert_eq!(
        running.operating_point,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
    );
    assert_eq!(running.controller_id, controller("pump"));
    assert_eq!(running.time_s.to_bits(), 1.0_f64.to_bits());
    assert_eq!(running.accepted_steps, 1);
    let before = batch.snapshot_data()?;
    let proposal = registry.propose_at(1.0, &[wet("pump", 0.0)], &[], &[frame(&batch)])?;
    assert_eq!(proposal.operations()?, batch.operations()?);
    assert_eq!(proposal.outputs()?[0].hydraulic.discharge_m3_per_s, 4.0);
    assert_eq!(batch.snapshot_data()?, before);
    let batch = accept(batch, 2.0, Some(1.5))?;
    assert_eq!(batch.outputs()?[0].latch, ControlLatch::High);
    let batch = accept(batch, 3.0, Some(1.0))?;
    assert_eq!(
        batch.operations()?[0].operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    let batch = accept(batch, 4.0, None)?;
    assert!(batch.next_event().is_none());
    assert_eq!(batch.controls().states()?[0].transition_count, 2);
    Ok(())
}

#[test]
fn running_speed_selection_preserves_original_binary_and_nonbinary_literals_across_layouts()
-> Result {
    for width in [1, 31, 32, 33, 64, 65] {
        let names = (0..width).map(|i| format!("pump-{i}")).collect::<Vec<_>>();
        let ids = names.iter().map(String::as_str).collect::<Vec<_>>();
        let context = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            ids.iter().map(|id| pump(id)).collect::<Result<Vec<_>>>()?,
        )?;
        let mut config = Config::switching();
        config.commands = [
            ControlCommand::PumpRunning { speed_ratio: 0.5 },
            ControlCommand::PumpRunning { speed_ratio: 0.75 },
        ];
        let mut batch = config.build(&context, &ids)?;
        let registry = registry(&context, &batch)?;
        for (time, stage, speed) in [
            (0.0, None, 0.5_f64),
            (1.0, Some(2.0), 0.75_f64),
            (2.0, Some(1.0), 0.5_f64),
        ] {
            if time > 0.0 {
                batch = accept(batch, time, stage)?;
            }
            let packets = ids.iter().map(|id| wet(id, 0.0)).collect::<Vec<_>>();
            let outputs = point(&registry, &batch, &packets)?;
            for (lane, output) in outputs.iter().enumerate() {
                assert_eq!(output.hydraulic.element, lane);
                assert_eq!(
                    output.hydraulic.operation,
                    StructureOperatingPoint::PumpRunning { speed_ratio: speed }
                );
                assert_eq!(output.hydraulic.discharge_m3_per_s, 4.0 * speed);
                match output.source.detail {
                    OperatingSourceDetail::ControlledPump {
                        ref controller_id,
                        accepted_steps,
                        accepted_time_s,
                        response_model,
                        operating_point,
                        ..
                    } => {
                        assert_eq!(controller_id, &controller(&names[lane]));
                        assert_eq!(accepted_steps, batch.accepted_steps());
                        assert_eq!(accepted_time_s.to_bits(), time.to_bits());
                        assert_eq!(response_model, PumpResponseModel::IdealInstantaneous);
                        let StructureOperatingPoint::PumpRunning { speed_ratio } = operating_point
                        else {
                            return Err("Running receipt lost its operation".into());
                        };
                        assert_eq!(speed_ratio.to_bits(), speed.to_bits());
                    }
                    _ => return Err("Controlled pump receipt lost its response provenance".into()),
                }
            }
        }
    }
    Ok(())
}

#[test]
fn qualification_and_band_hold_change_no_operation_before_the_qualified_endpoint() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let mut config = Config::switching();
    config.times = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    config.end = 6.0;
    config.delays = (2.0, 1.0);
    let mut batch = config.build(&context, &["pump"])?;
    for (time, stage, running, pending) in [
        (1.0, 2.0, false, Some(1.0)),
        (2.0, 2.0, false, Some(1.0)),
        (3.0, 2.0, true, None),
        (4.0, 1.5, true, None),
        (5.0, 1.0, true, Some(5.0)),
    ] {
        batch = accept(batch, time, Some(stage))?;
        assert_eq!(
            matches!(
                batch.operations()?[0].operating_point,
                StructureOperatingPoint::PumpRunning { .. }
            ),
            running
        );
        assert_eq!(
            batch.controls().states()?[0]
                .pending
                .map(|value| value.since_s),
            pending
        );
    }
    let before = batch.controls().states()?;
    let batch = accept(batch, 6.0, None)?;
    assert_eq!(batch.controls().states()?, before);
    assert!(matches!(
        batch.operations()?[0].operating_point,
        StructureOperatingPoint::PumpRunning { .. }
    ));
    Ok(())
}

#[test]
fn empty_calendar_response_has_only_original_terminal_support() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let mut config = Config::switching();
    config.times.clear();
    let batch = config.build(&context, &["pump"])?;
    let event = batch.next_event().ok_or("Missing terminal event")?;
    assert_eq!(event.time_s, 4.0);
    assert!(event.terminal);
    assert!(event.decision_index.is_none());
    let batch = accept(batch, 4.0, None)?;
    assert_eq!(batch.accepted_steps(), 1);
    assert!(batch.next_event().is_none());
    assert_eq!(
        batch.outputs()?[0].operating_point,
        StructureOperatingPoint::PumpStoppedIsolated
    );
    Ok(())
}

#[test]
fn trusted_and_cold_continuation_retain_pending_memory_and_explicit_response_identity() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let mut config = Config::switching();
    config.delays = (1.0, 0.0);
    let source = accept(config.build(&context, &["pump"])?, 1.0, Some(2.0))?;
    let source = accept(source, 1.5, None)?;
    let trusted = source.snapshot();
    let data = source.snapshot_data()?;
    assert_eq!(trusted.time().to_bits(), source.time().to_bits());
    assert_eq!(trusted.data()?, data);
    let mut restored = config.build(&context, &["pump"])?;
    let mut imported = config.build(&context, &["pump"])?;
    restored.restore(&trusted)?;
    imported.restore_data(&data)?;
    for (time, stage) in [(2.0, Some(2.0)), (3.0, Some(1.0)), (4.0, None)] {
        restored = accept(restored, time, stage)?;
        imported = accept(imported, time, stage)?;
        assert_eq!(restored.snapshot_data()?, imported.snapshot_data()?);
        assert_eq!(restored.operations()?, imported.operations()?);
    }
    let mut definitions = responses(&["pump"]);
    definitions[0]
        .ideal_response_applicability
        .push_str(" distinct");
    let mut incompatible = ControlledPumpBatch::<crate::runtime::Runtime>::new(
        config.controls(&context, &["pump"])?,
        definitions,
    )?;
    let before = incompatible.snapshot_data()?;
    kind(
        incompatible.restore(&trusted),
        StructureInputErrorKind::Definition,
    )?;
    assert_eq!(incompatible.snapshot_data()?, before);
    kind(
        incompatible.restore_data(&data),
        StructureInputErrorKind::Definition,
    )?;
    assert_eq!(incompatible.snapshot_data()?, before);
    let mut malformed = data;
    malformed.definitions[0]
        .response_provenance
        .push_str(" altered");
    let before = imported.snapshot_data()?;
    kind(
        imported.restore_data(&malformed),
        StructureInputErrorKind::Definition,
    )?;
    assert_eq!(imported.snapshot_data()?, before);
    Ok(())
}
