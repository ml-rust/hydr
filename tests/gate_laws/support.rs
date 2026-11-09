use super::fixture::*;
use hydr::structures::*;

#[test]
fn closed_gate_retains_support_without_evaluating_influential_heads() -> Result {
    let batch = StructureBatch::new(vec![standard("gate", DirectionPolicy::Bidirectional)?])?;
    for input in [
        packet("gate", HeadSupport::Unknown, HeadSupport::Unknown),
        packet(
            "gate",
            HeadSupport::Dry {
                bed_elevation_m: 100.0,
            },
            HeadSupport::Unknown,
        ),
        heads("gate", f64::MAX, -f64::MAX),
    ] {
        let output = first(
            &batch
                .propose_with_operations(std::slice::from_ref(&input), &[opening("gate", 0.0)])?,
        )?;
        assert_eq!(output.input, input);
        assert_eq!(
            output.operation,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 }
        );
        assert_eq!(output.law, StructureLawKind::HydrostaticRectangularGate);
        assert_eq!(output.regime, StructureRegime::GateClosed);
        assert_eq!(output.status, StructureStatus::PhysicalZero);
        assert_eq!(output.proposed_direction, None);
        assert_eq!(output.discharge_m3_per_s, 0.0);
        assert!(output.hydraulic_operands.is_none());
    }
    Ok(())
}

#[test]
fn closed_gate_still_checks_every_supplied_identity_datum_convention_and_value() -> Result {
    let batch = StructureBatch::new(vec![standard("gate", DirectionPolicy::BtoAOnly)?])?;
    let closed = [opening("gate", 0.0)];
    let base = packet("gate", HeadSupport::Unknown, HeadSupport::Unknown);
    for field in 0..3 {
        let mut invalid = base.clone();
        match field {
            0 => invalid.structure_id = "foreign".into(),
            1 => invalid.endpoint_a.id = "foreign".into(),
            _ => invalid.endpoint_b.id = "foreign".into(),
        }
        kind(
            &batch,
            &[invalid],
            &closed,
            StructureInputErrorKind::Identity,
        )?;
    }
    let mut invalid = base.clone();
    invalid.datum = "Foreign datum".into();
    kind(&batch, &[invalid], &closed, StructureInputErrorKind::Datum)?;
    let mut invalid = base.clone();
    invalid.convention = HeadConvention::EnergyHead;
    kind(
        &batch,
        &[invalid],
        &closed,
        StructureInputErrorKind::HeadConvention,
    )?;
    for support in [
        HeadSupport::Wet { stage_m: f64::NAN },
        HeadSupport::Wet {
            stage_m: f64::INFINITY,
        },
        HeadSupport::Dry {
            bed_elevation_m: f64::NEG_INFINITY,
        },
    ] {
        for endpoint_a in [false, true] {
            let mut invalid = base.clone();
            if endpoint_a {
                invalid.endpoint_a.support = support.clone();
            } else {
                invalid.endpoint_b.support = support.clone();
            }
            kind(
                &batch,
                &[invalid],
                &closed,
                StructureInputErrorKind::NonfiniteInput,
            )?;
        }
    }
    kind(&batch, &[], &closed, StructureInputErrorKind::Identity)?;
    kind(
        &batch,
        &[base.clone(), base],
        &closed,
        StructureInputErrorKind::Identity,
    )?;
    Ok(())
}

#[test]
fn positive_opening_checks_support_before_ideal_flap_blocking() -> Result {
    let batch = StructureBatch::new(vec![standard("gate", DirectionPolicy::BtoAOnly)?])?;
    let open = [opening("gate", 1.0)];
    for (a, b) in [
        (HeadSupport::Unknown, HeadSupport::Wet { stage_m: 2.0 }),
        (HeadSupport::Wet { stage_m: 2.0 }, HeadSupport::Unknown),
    ] {
        kind(
            &batch,
            &[packet("gate", a, b)],
            &open,
            StructureInputErrorKind::UnknownSupport,
        )?;
    }
    for (a, b) in [
        (
            HeadSupport::Wet { stage_m: 2.0 },
            HeadSupport::Dry {
                bed_elevation_m: 0.1,
            },
        ),
        (
            HeadSupport::Dry {
                bed_elevation_m: 0.1,
            },
            HeadSupport::Wet { stage_m: 2.0 },
        ),
        (
            HeadSupport::Dry {
                bed_elevation_m: 0.1,
            },
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
        ),
    ] {
        kind(
            &batch,
            &[packet("gate", a, b)],
            &open,
            StructureInputErrorKind::UnsupportedRegime,
        )?;
    }
    let blocked = first(&batch.propose_with_operations(&[heads("gate", 3.0, 2.0)], &open)?)?;
    assert_eq!(blocked.status, StructureStatus::FlapBlocked);
    assert_eq!(blocked.proposed_direction, Some(FlowDirection::AtoB));
    assert_eq!(blocked.discharge_m3_per_s, 0.0);
    assert!(blocked.hydraulic_operands.is_none());
    Ok(())
}
