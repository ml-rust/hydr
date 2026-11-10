use super::fixture::*;
use hydr::structures::*;

#[test]
fn dynamic_pump_frames_require_registered_count_order_and_source_identity() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("a")?, pump("b")?],
    )?;
    let config = Config::switching();
    let a = config.build(&context, &["a"])?;
    let b = config.build(&context, &["b"])?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::ControlledPump {
            source_id: "b".into(),
            pump: &b,
        },
        OperatingSourceBinding::ControlledPump {
            source_id: "a".into(),
            pump: &a,
        },
    ])?;
    let packets = [wet("a", 0.0), wet("b", 0.0)];
    let fa = OperatingSourceFrame::ControlledPump {
        source_id: "a",
        pump: &a,
    };
    let fb = OperatingSourceFrame::ControlledPump {
        source_id: "b",
        pump: &b,
    };
    let foreign = OperatingSourceFrame::ControlledPump {
        source_id: "foreign",
        pump: &a,
    };
    for frames in [
        Vec::new(),
        vec![fb],
        vec![fa, fb],
        vec![fb, foreign],
        vec![fb, fa, fa],
    ] {
        kind(
            registry.propose_at(0.0, &packets, &[], &frames),
            StructureInputErrorKind::Identity,
        )?;
        kind(
            registry.next_event_after(0.0, &frames),
            StructureInputErrorKind::Identity,
        )?;
    }
    let outputs = registry
        .propose_at(0.0, &packets, &[], &[fb, fa])?
        .outputs()?;
    assert_eq!(outputs[0].source.source_id, "a");
    assert_eq!(outputs[1].source.source_id, "b");
    let event = registry
        .next_event_after(0.0, &[fb, fa])?
        .ok_or("Missing simultaneous decision")?;
    assert_eq!(event.time_s, 1.0);
    assert_eq!(
        event
            .origins
            .iter()
            .map(|origin| origin.source_id.as_str())
            .collect::<Vec<_>>(),
        ["b", "a"]
    );
    Ok(())
}

#[test]
fn dynamic_frame_kind_must_match_the_registered_response_owner() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?, gate("gate")?],
    )?;
    let batch = Config::switching().build(&context, &["pump"])?;
    let actual = actual_gate(&context, "gate", 4.0)?;
    let registry = context.with_operating_sources(vec![
        OperatingSourceBinding::ControlledPump {
            source_id: "pump-source".into(),
            pump: &batch,
        },
        OperatingSourceBinding::Actuated {
            source_id: "gate-source".into(),
            actuator: &actual,
        },
    ])?;
    let packets = [wet("pump", 0.0), wet("gate", 0.0)];
    let wrong = [
        OperatingSourceFrame::ActuatedGate {
            source_id: "pump-source",
            actuator: &actual,
        },
        OperatingSourceFrame::ControlledPump {
            source_id: "gate-source",
            pump: &batch,
        },
    ];
    kind(
        registry.propose_at(0.0, &packets, &[], &wrong),
        StructureInputErrorKind::Identity,
    )?;
    kind(
        registry.next_event_after(0.0, &wrong),
        StructureInputErrorKind::Identity,
    )
}

#[test]
fn stale_and_incompatible_pump_frames_cannot_reuse_matching_current_operations() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let config = Config::switching();
    let initial = config.build(&context, &["pump"])?;
    let registry = registry(&context, &initial)?;
    let later = accept(initial.clone(), 0.5, None)?;
    kind(
        registry.propose_at(0.0, &[wet("pump", 0.0)], &[], &[frame(&later)]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    kind(
        registry.next_event_after(0.0, &[frame(&later)]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    kind(
        registry.propose_at(0.5, &[wet("pump", 0.0)], &[], &[frame(&initial)]),
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    for variant in 0..4 {
        let mut definitions = responses(&["pump"]);
        let mut changed = config.clone();
        let mut elements = context.elements().to_vec();
        match variant {
            0 => definitions[0].response_provenance.push_str(" changed"),
            1 => definitions[0]
                .ideal_response_applicability
                .push_str(" changed"),
            2 => changed.start = -0.0,
            3 => elements[0].endpoint_b.push_str(" changed"),
            _ => return Err("Unknown pump frame variant".into()),
        }
        let foreign_context =
            StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), elements)?;
        let foreign = ControlledPumpBatch::<crate::runtime::Runtime>::new(
            changed.controls(&foreign_context, &["pump"])?,
            definitions,
        )?;
        assert_eq!(foreign.operations()?, initial.operations()?);
        kind(
            registry.propose_at(0.0, &[wet("pump", 0.0)], &[], &[frame(&foreign)]),
            StructureInputErrorKind::Definition,
        )?;
    }
    Ok(())
}

#[test]
fn reconstructed_and_cold_restored_response_frames_preserve_original_model_identity() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let config = Config::switching();
    let initial = config.build(&context, &["pump"])?;
    let registry = registry(&context, &initial)?;
    let rebuilt_context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        context.elements().to_vec(),
    )?;
    let mut rebuilt = config.build(&rebuilt_context, &["pump"])?;
    let advanced = accept(initial, 1.0, Some(2.0))?;
    rebuilt.restore_data(&advanced.snapshot_data()?)?;
    assert_eq!(
        point(&registry, &advanced, &[wet("pump", 0.0)])?,
        point(&registry, &rebuilt, &[wet("pump", 0.0)])?
    );
    Ok(())
}

#[test]
fn numeric_zero_query_matching_retains_separate_frame_and_query_time_bits() -> Result {
    for (accepted_time, query_time) in [(-0.0, 0.0), (0.0, -0.0)] {
        let context = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![pump("pump")?],
        )?;
        let mut config = Config::switching();
        config.start = accepted_time;
        let batch = config.build(&context, &["pump"])?;
        let registry = registry(&context, &batch)?;
        let proposal =
            registry.propose_at(query_time, &[wet("pump", 0.0)], &[], &[frame(&batch)])?;
        assert_eq!(proposal.time_s().to_bits(), query_time.to_bits());
        let output = proposal.outputs()?.remove(0);
        assert_eq!(output.source.time_s.to_bits(), query_time.to_bits());
        match output.source.detail {
            OperatingSourceDetail::ControlledPump {
                accepted_time_s, ..
            } => assert_eq!(accepted_time_s.to_bits(), accepted_time.to_bits()),
            _ => return Err("Pump source lost accepted-time provenance".into()),
        }
        let mut trusted = config.build(&context, &["pump"])?;
        let mut imported = config.build(&context, &["pump"])?;
        trusted.restore(&batch.snapshot())?;
        imported.restore_data(&batch.snapshot_data()?)?;
        for restored in [trusted, imported] {
            assert_eq!(restored.time().to_bits(), accepted_time.to_bits());
            assert_eq!(
                restored.controls().schedule().initial_time_s.to_bits(),
                accepted_time.to_bits()
            );
            assert_eq!(
                restored.outputs()?[0].time_s.to_bits(),
                accepted_time.to_bits()
            );
        }
    }
    Ok(())
}
