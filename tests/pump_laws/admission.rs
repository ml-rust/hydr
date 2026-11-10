use super::fixture::*;
use hydr::structures::*;

#[test]
fn operation_identity_and_law_are_explicit_before_physical_shortcuts() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("first")?, standard("second")?],
    )?;
    let packets = [heads("first", 0.0, 4.0), heads("second", 0.0, 4.0)];
    let operations = [running("first", 1.0), running("second", 1.0)];
    let error = rejection(batch.propose(&packets))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::OperatingPoint)
    );
    for wrong in [
        vec![],
        vec![operations[0].clone()],
        vec![operations[0].clone(), operations[0].clone()],
        vec![operations[1].clone(), operations[0].clone()],
        vec![running("foreign", 1.0), operations[1].clone()],
    ] {
        assert!(batch.propose_with_operations(&packets, &wrong).is_err());
    }
    kind(
        &batch,
        &packets,
        &[
            operation("first", StructureOperatingPoint::Passive),
            operations[1].clone(),
        ],
        StructureInputErrorKind::OperatingPoint,
    )?;
    let passive = element(
        "passive",
        StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            0.0,
            1.0,
            WeirApplicability {
                coefficient_provenance: "Analytical SI fixture".into(),
                free_nappe_and_approach_assertion: "Suppressed sharp-crested analytical fixture"
                    .into(),
                submergence_correction_assertion: "Adopted analytical correction".into(),
                symmetric_reversal: true,
            },
        )?),
        DirectionPolicy::Bidirectional,
    );
    let passive =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), vec![passive])?;
    kind(
        &passive,
        &[heads("passive", 0.0, 0.0)],
        &[operation(
            "passive",
            StructureOperatingPoint::PumpStoppedIsolated,
        )],
        StructureInputErrorKind::OperatingPoint,
    )?;
    kind(
        &passive,
        &[heads("passive", 0.0, 0.0)],
        &[running("passive", 1.0)],
        StructureInputErrorKind::OperatingPoint,
    )?;
    Ok(())
}

#[test]
fn running_shutoff_still_requires_known_wet_suction_and_receiver() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("pump")?],
    )?;
    for operation in [running("pump", 1.0), running("pump", 0.5)] {
        for (a, b, expected) in [
            (
                HeadSupport::Unknown,
                HeadSupport::Wet { stage_m: 8.0 },
                StructureInputErrorKind::UnknownSupport,
            ),
            (
                HeadSupport::Wet { stage_m: 0.0 },
                HeadSupport::Unknown,
                StructureInputErrorKind::UnknownSupport,
            ),
            (
                HeadSupport::Dry {
                    bed_elevation_m: 0.0,
                },
                HeadSupport::Wet { stage_m: 8.0 },
                StructureInputErrorKind::UnsupportedRegime,
            ),
            (
                HeadSupport::Wet { stage_m: 0.0 },
                HeadSupport::Dry {
                    bed_elevation_m: 8.0,
                },
                StructureInputErrorKind::UnsupportedRegime,
            ),
            (
                HeadSupport::Wet { stage_m: -2.0 },
                HeadSupport::Wet { stage_m: 6.0 },
                StructureInputErrorKind::UnsupportedRegime,
            ),
        ] {
            kind(
                &batch,
                &[packet("pump", a, b)],
                std::slice::from_ref(&operation),
                expected,
            )?;
        }
    }
    Ok(())
}

#[test]
fn stopped_isolation_admits_uninfluential_support_but_checks_every_supplied_value() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("pump")?],
    )?;
    let stopped = operation("pump", StructureOperatingPoint::PumpStoppedIsolated);
    for packet in [
        packet("pump", HeadSupport::Unknown, HeadSupport::Unknown),
        packet(
            "pump",
            HeadSupport::Dry {
                bed_elevation_m: 100.0,
            },
            HeadSupport::Unknown,
        ),
        heads("pump", f64::MAX, -f64::MAX),
    ] {
        let output = first(&batch.propose_with_operations(
            std::slice::from_ref(&packet),
            std::slice::from_ref(&stopped),
        )?)?;
        assert_eq!(output.input, packet);
        assert_eq!(
            output.operation,
            StructureOperatingPoint::PumpStoppedIsolated
        );
        assert_eq!(output.status, StructureStatus::PumpStoppedIsolated);
        assert_eq!(output.regime, StructureRegime::PumpIsolated);
        assert_eq!(output.discharge_m3_per_s, 0.0);
        assert_eq!(output.proposed_direction, None);
        assert!(output.hydraulic_operands.is_none());
    }
    let base = packet("pump", HeadSupport::Unknown, HeadSupport::Unknown);
    for (support, expected) in [
        (
            HeadSupport::Wet { stage_m: f64::NAN },
            StructureInputErrorKind::NonfiniteInput,
        ),
        (
            HeadSupport::Dry {
                bed_elevation_m: f64::INFINITY,
            },
            StructureInputErrorKind::NonfiniteInput,
        ),
    ] {
        let mut invalid = base.clone();
        invalid.endpoint_a.support = support;
        kind(&batch, &[invalid], std::slice::from_ref(&stopped), expected)?;
    }
    let mut invalid = base.clone();
    invalid.convention = HeadConvention::EnergyHead;
    kind(
        &batch,
        &[invalid],
        std::slice::from_ref(&stopped),
        StructureInputErrorKind::HeadConvention,
    )?;
    let mut invalid = base.clone();
    invalid.datum = "Foreign datum".into();
    kind(
        &batch,
        &[invalid],
        std::slice::from_ref(&stopped),
        StructureInputErrorKind::Datum,
    )?;
    let mut invalid = base;
    invalid.endpoint_b.id = "foreign".into();
    kind(
        &batch,
        &[invalid],
        &[stopped],
        StructureInputErrorKind::Identity,
    )?;
    Ok(())
}

#[test]
fn speed_policy_has_no_implicit_affinity_or_stopped_state() -> Result {
    let reference = element(
        "pump",
        StructureLaw::Pump(PumpParameters::new(
            curve(&[0.0, 8.0], &[4.0, 0.0])?,
            -2.0,
            PumpSpeedPolicy::ReferenceOnly,
            applicability(),
        )?),
        DirectionPolicy::AtoBOnly,
    );
    let batch =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), vec![reference])?;
    for speed in [0.0, -1.0, 0.5, 1.0_f64.next_up(), f64::INFINITY, f64::NAN] {
        kind(
            &batch,
            &[heads("pump", 0.0, 0.0)],
            &[running("pump", speed)],
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    let limited = PumpSpeedPolicy::DeclaredAffinity {
        minimum_ratio: 0.5,
        maximum_ratio: 2.0,
        applicability: "Declared analytical range".into(),
    };
    let limited = element(
        "pump",
        StructureLaw::Pump(PumpParameters::new(
            curve(&[0.0, 8.0], &[4.0, 0.0])?,
            -2.0,
            limited,
            applicability(),
        )?),
        DirectionPolicy::AtoBOnly,
    );
    let batch =
        StructureBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), vec![limited])?;
    for speed in [0.25, 4.0] {
        kind(
            &batch,
            &[heads("pump", 0.0, 0.0)],
            &[running("pump", speed)],
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    Ok(())
}
