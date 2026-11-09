use super::fixture::*;
use hydr::structures::*;

fn check_kind(
    batch: &StructureBatch,
    packets: &[StructureHeads],
    kind: StructureInputErrorKind,
) -> Result {
    let error = rejection(batch.propose(packets))?;
    assert_eq!(error.input_kind(), Some(kind), "{error}");
    Ok(())
}

#[test]
fn support_eligibility_precedes_equal_heads_and_flap_policy() -> Result {
    let law = StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
        2.0,
        0.0,
        1.0,
        0.6,
        9.81,
        orifice_evidence(),
    )?);
    let batch = StructureBatch::new(vec![element("orifice", law, DirectionPolicy::BtoAOnly)])?;
    for packet in [
        heads("orifice", 1.0, 1.0),
        heads("orifice", 3.0, 1.0),
        packet(
            "orifice",
            HeadSupport::Wet { stage_m: 3.0 },
            HeadSupport::Dry {
                bed_elevation_m: 0.0,
            },
        ),
    ] {
        check_kind(
            &batch,
            &[packet],
            StructureInputErrorKind::UnsupportedRegime,
        )?;
    }
    check_kind(
        &batch,
        &[packet(
            "orifice",
            HeadSupport::Unknown,
            HeadSupport::Wet { stage_m: 2.0 },
        )],
        StructureInputErrorKind::UnknownSupport,
    )?;
    let blocked = first(&batch.propose(&[heads("orifice", 3.0, 2.0)])?)?;
    assert_eq!(blocked.status, StructureStatus::FlapBlocked);
    assert_eq!(blocked.regime, StructureRegime::SubmergedOrifice);
    assert_eq!(blocked.proposed_direction, Some(FlowDirection::AtoB));
    assert_eq!(blocked.discharge_m3_per_s, 0.0);
    assert!(blocked.hydraulic_operands.is_none());
    let equal = first(&batch.propose(&[heads("orifice", 2.0, 2.0)])?)?;
    assert_eq!(equal.status, StructureStatus::PhysicalZero);
    assert_eq!(equal.regime, StructureRegime::EqualHeads);
    assert_eq!(equal.proposed_direction, None);
    Ok(())
}

#[test]
fn known_dry_weir_support_retains_identity_without_invented_stage() -> Result {
    let batch = StructureBatch::new(vec![element(
        "weir",
        weir(1.7, 2.0, 10.0)?,
        DirectionPolicy::Bidirectional,
    )])?;
    let dry = packet(
        "weir",
        HeadSupport::Wet { stage_m: 11.0 },
        HeadSupport::Dry {
            bed_elevation_m: 9.0,
        },
    );
    let output = first(&batch.propose(&[dry])?)?;
    close(output.discharge_m3_per_s, 3.4, true);
    assert_eq!(output.regime, StructureRegime::FreeWeirOverflow);
    assert!(matches!(
        output.input.endpoint_b.support,
        HeadSupport::Dry {
            bed_elevation_m: 9.0
        }
    ));
    for (a, b) in [
        (
            HeadSupport::Dry {
                bed_elevation_m: 10.0,
            },
            HeadSupport::Dry {
                bed_elevation_m: 9.0,
            },
        ),
        (
            HeadSupport::Wet { stage_m: 10.0 },
            HeadSupport::Dry {
                bed_elevation_m: 9.0,
            },
        ),
        (
            HeadSupport::Wet { stage_m: 10.0 },
            HeadSupport::Wet { stage_m: 9.0 },
        ),
    ] {
        let output = first(&batch.propose(&[packet("weir", a, b)])?)?;
        assert_eq!(output.status, StructureStatus::PhysicalZero);
        assert_eq!(output.regime, StructureRegime::BelowCrest);
        assert_eq!(output.discharge_m3_per_s, 0.0);
        assert!(output.hydraulic_operands.is_none());
    }
    check_kind(
        &batch,
        &[packet(
            "weir",
            HeadSupport::Wet { stage_m: 11.0 },
            HeadSupport::Dry {
                bed_elevation_m: 10.1,
            },
        )],
        StructureInputErrorKind::UnsupportedRegime,
    )?;
    Ok(())
}

#[test]
fn packet_metadata_rejects_before_zero_and_blocked_shortcuts() -> Result {
    let batch = StructureBatch::new(vec![element(
        "weir",
        weir(1.0, 1.0, 0.0)?,
        DirectionPolicy::BtoAOnly,
    )])?;
    for (a, b) in [(0.0, 0.0), (1.0, 0.0)] {
        let base = heads("weir", a, b);
        let mut wrong = base.clone();
        wrong.datum = "Foreign datum".into();
        check_kind(&batch, &[wrong], StructureInputErrorKind::Datum)?;
        let mut wrong = base.clone();
        wrong.convention = HeadConvention::EnergyHead;
        check_kind(&batch, &[wrong], StructureInputErrorKind::HeadConvention)?;
        let mut wrong = base.clone();
        wrong.structure_id = "foreign".into();
        check_kind(&batch, &[wrong], StructureInputErrorKind::Identity)?;
        let mut wrong = base.clone();
        std::mem::swap(&mut wrong.endpoint_a.id, &mut wrong.endpoint_b.id);
        check_kind(&batch, &[wrong], StructureInputErrorKind::Identity)?;
        let mut wrong = base.clone();
        wrong.endpoint_a.support = HeadSupport::Unknown;
        check_kind(&batch, &[wrong], StructureInputErrorKind::UnknownSupport)?;
        let mut wrong = base.clone();
        wrong.endpoint_b.support = HeadSupport::Wet { stage_m: f64::NAN };
        check_kind(&batch, &[wrong], StructureInputErrorKind::NonfiniteInput)?;
        assert!(batch.propose(&[]).is_err());
        assert!(batch.propose(&[base.clone(), base]).is_err());
    }
    Ok(())
}

#[test]
fn ordered_packets_cannot_change_parallel_structure_destinations() -> Result {
    let law = weir(1.0, 1.0, 0.0)?;
    let batch = StructureBatch::new(vec![
        element("first", law.clone(), DirectionPolicy::Bidirectional),
        element("second", law, DirectionPolicy::Bidirectional),
    ])?;
    let packets = [heads("first", 1.0, 0.0), heads("second", 2.0, 0.0)];
    let original = batch.propose(&packets)?.outputs()?;
    check_kind(
        &batch,
        &[packets[1].clone(), packets[0].clone()],
        StructureInputErrorKind::Identity,
    )?;
    assert_eq!(batch.propose(&packets)?.outputs()?, original);
    Ok(())
}

#[test]
fn blocked_supported_regime_skips_unneeded_unrepresentable_law() -> Result {
    let batch = StructureBatch::new(vec![element(
        "blocked",
        weir(f64::MAX, 1.0, -f64::MAX)?,
        DirectionPolicy::BtoAOnly,
    )])?;
    let output = first(&batch.propose(&[heads("blocked", f64::MAX, 0.0)])?)?;
    assert_eq!(output.status, StructureStatus::FlapBlocked);
    assert_eq!(output.regime, StructureRegime::SubmergedWeirOverflow);
    assert_eq!(output.discharge_m3_per_s, 0.0);
    assert!(output.hydraulic_operands.is_none());
    Ok(())
}

#[test]
fn definitions_require_physical_geometry_identity_and_explicit_applicability() -> Result {
    for value in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert!(
            SubmergedOrificeParameters::new(value, 0.0, 1.0, 0.6, 9.81, orifice_evidence())
                .is_err()
        );
        assert!(
            SubmergedOrificeParameters::new(1.0, 0.0, 1.0, value, 9.81, orifice_evidence())
                .is_err()
        );
        assert!(
            SubmergedOrificeParameters::new(1.0, 0.0, 1.0, 0.6, value, orifice_evidence()).is_err()
        );
        assert!(
            SuppressedRectangularWeirParameters::new(value, 0.0, 1.0, weir_evidence()).is_err()
        );
        assert!(
            SuppressedRectangularWeirParameters::new(1.0, 0.0, value, weir_evidence()).is_err()
        );
    }
    assert!(SubmergedOrificeParameters::new(1.0, 1.0, 1.0, 0.6, 9.81, orifice_evidence()).is_err());
    let mut evidence = orifice_evidence();
    evidence.coefficient_provenance.clear();
    assert!(SubmergedOrificeParameters::new(1.0, 0.0, 1.0, 0.6, 9.81, evidence).is_err());
    let mut evidence = weir_evidence();
    evidence.submergence_correction_assertion = "invalid\nevidence".into();
    assert!(SuppressedRectangularWeirParameters::new(1.0, 0.0, 1.0, evidence).is_err());
    let base = element("weir", weir(1.0, 1.0, 0.0)?, DirectionPolicy::Bidirectional);
    assert!(StructureBatch::new(vec![base.clone(), base.clone()]).is_err());
    let mut same = base.clone();
    same.endpoint_b = same.endpoint_a.clone();
    assert!(StructureBatch::new(vec![same]).is_err());
    for field in 0..4 {
        let mut invalid = base.clone();
        match field {
            0 => invalid.id.clear(),
            1 => invalid.endpoint_a.clear(),
            2 => invalid.endpoint_b.clear(),
            _ => invalid.datum.clear(),
        }
        assert!(StructureBatch::new(vec![invalid]).is_err());
    }
    let mut asymmetric = weir_evidence();
    asymmetric.symmetric_reversal = false;
    let law = StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
        1.0, 0.0, 1.0, asymmetric,
    )?);
    assert!(
        StructureBatch::new(vec![element(
            "asymmetric",
            law.clone(),
            DirectionPolicy::Bidirectional
        )])
        .is_err()
    );
    assert!(StructureBatch::new(vec![element("one-way", law, DirectionPolicy::AtoBOnly)]).is_ok());
    Ok(())
}
