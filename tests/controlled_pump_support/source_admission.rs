use super::fixture::*;
use hydr::structures::*;

#[test]
fn selected_stopped_response_waives_only_unknown_influential_support() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let batch = Config::switching().build(&context, &["pump"])?;
    let registry = registry(&context, &batch)?;
    for packet in [
        heads("pump", HeadSupport::Unknown, HeadSupport::Unknown),
        heads(
            "pump",
            HeadSupport::Dry {
                bed_elevation_m: 100.0,
            },
            HeadSupport::Unknown,
        ),
        heads(
            "pump",
            HeadSupport::Wet { stage_m: f64::MAX },
            HeadSupport::Wet { stage_m: -f64::MAX },
        ),
    ] {
        let outputs = point(&registry, &batch, std::slice::from_ref(&packet))?;
        let output = &outputs[0].hydraulic;
        assert_eq!(output.input, packet);
        assert_eq!(
            output.operation,
            StructureOperatingPoint::PumpStoppedIsolated
        );
        assert_eq!(output.status, StructureStatus::PumpStoppedIsolated);
        assert_eq!(output.regime, StructureRegime::PumpIsolated);
        assert_eq!(output.discharge_m3_per_s.to_bits(), 0.0_f64.to_bits());
        assert!(output.hydraulic_operands.is_none());
    }
    Ok(())
}

#[test]
fn running_and_curve_shutoff_supply_no_unknown_support_isolation_authority() -> Result {
    for speed in [0.5, 1.0] {
        let context = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![pump("pump")?],
        )?;
        let mut config = Config::switching();
        config.commands[1] = ControlCommand::PumpRunning { speed_ratio: speed };
        let batch = config.build(&context, &["pump"])?;
        let registry = registry(&context, &batch)?;
        let batch = accept(batch, 1.0, Some(2.0))?;
        let shutoff_head = 8.0 * speed * speed;
        let output = point(&registry, &batch, &[wet("pump", shutoff_head)])?.remove(0);
        assert_eq!(output.hydraulic.status, StructureStatus::PumpShutoff);
        assert_eq!(output.hydraulic.discharge_m3_per_s, 0.0);
        assert!(matches!(
            output.hydraulic.operation,
            StructureOperatingPoint::PumpRunning { .. }
        ));
        for packet in [
            heads(
                "pump",
                HeadSupport::Unknown,
                HeadSupport::Wet {
                    stage_m: shutoff_head,
                },
            ),
            heads(
                "pump",
                HeadSupport::Wet { stage_m: 0.0 },
                HeadSupport::Unknown,
            ),
        ] {
            kind(
                registry.propose_at(1.0, &[packet], &[], &[frame(&batch)]),
                StructureInputErrorKind::UnknownSupport,
            )?;
        }
    }
    Ok(())
}

#[test]
fn known_malformed_inputs_reject_before_selected_stopped_isolation() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let batch = Config::switching().build(&context, &["pump"])?;
    let registry = registry(&context, &batch)?;
    let base = heads("pump", HeadSupport::Unknown, HeadSupport::Unknown);
    for variant in 0..5 {
        let mut packet = base.clone();
        let expected = match variant {
            0 => {
                packet.endpoint_a.support = HeadSupport::Wet { stage_m: f64::NAN };
                StructureInputErrorKind::NonfiniteInput
            }
            1 => {
                packet.endpoint_b.support = HeadSupport::Dry {
                    bed_elevation_m: f64::INFINITY,
                };
                StructureInputErrorKind::NonfiniteInput
            }
            2 => {
                packet.datum.push_str(" different");
                StructureInputErrorKind::Datum
            }
            3 => {
                packet.endpoint_a.id.push_str(" different");
                StructureInputErrorKind::Identity
            }
            4 => {
                packet.convention = HeadConvention::EnergyHead;
                StructureInputErrorKind::HeadConvention
            }
            _ => return Err("Unknown malformed head variant".into()),
        };
        kind(
            registry.propose_at(0.0, &[packet], &[], &[frame(&batch)]),
            expected,
        )?;
    }
    Ok(())
}

#[test]
fn isolated_unknown_lane_cannot_hide_another_lanes_malformed_known_head() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("a")?, pump("b")?],
    )?;
    let batch = Config::switching().build(&context, &["a", "b"])?;
    let registry = registry(&context, &batch)?;
    let packets = [
        heads("a", HeadSupport::Unknown, HeadSupport::Unknown),
        heads(
            "b",
            HeadSupport::Wet { stage_m: f64::NAN },
            HeadSupport::Unknown,
        ),
    ];
    let error = rejected(registry.propose_at(0.0, &packets, &[], &[frame(&batch)]))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::NonfiniteInput)
    );
    assert_eq!(error.element(), Some(1));
    Ok(())
}

#[test]
fn selected_running_response_retains_suction_and_receiving_regime_checks() -> Result {
    let context = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![pump("pump")?],
    )?;
    let initial = Config::switching().build(&context, &["pump"])?;
    let registry = registry(&context, &initial)?;
    let batch = accept(initial, 1.0, Some(2.0))?;
    for packet in [
        heads(
            "pump",
            HeadSupport::Wet { stage_m: -2.0 },
            HeadSupport::Wet { stage_m: 0.0 },
        ),
        heads(
            "pump",
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
            HeadSupport::Wet { stage_m: 0.0 },
        ),
        heads(
            "pump",
            HeadSupport::Wet { stage_m: 0.0 },
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
        ),
    ] {
        kind(
            registry.propose_at(1.0, &[packet], &[], &[frame(&batch)]),
            StructureInputErrorKind::UnsupportedRegime,
        )?;
    }
    Ok(())
}
