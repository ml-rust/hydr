use super::{fixture::*, references::catalog};
use hydr::structures::*;

#[test]
fn held_heads_preserve_signed_single_product_and_original_packet() -> Result {
    let batch = StructureBatch::new(vec![element(
        "orifice",
        orifice(0.6, 2.0, 9.81)?,
        DirectionPolicy::Bidirectional,
    )])?;
    for (a, b) in [(3.0, 2.0), (2.0, 3.0)] {
        let proposal = batch.propose(&[heads("orifice", a, b)])?;
        let original = first(&proposal)?;
        let held = proposal.held_volumes(HeldHeadInterval {
            start: 1.0,
            end: 3.0,
        })?;
        assert_eq!(held.interval().start.to_bits(), 1.0_f64.to_bits());
        assert_eq!(held.interval().end.to_bits(), 3.0_f64.to_bits());
        let output = held
            .outputs()?
            .into_iter()
            .next()
            .ok_or("Missing held output")?;
        assert_eq!(
            output.volume_m3.to_bits(),
            (original.discharge_m3_per_s * 2.0).to_bits()
        );
        assert_eq!(format!("{:?}", output.proposal), format!("{original:?}"));
        assert_eq!(output.interval.start, 1.0);
        assert_eq!(output.interval.end, 3.0);
    }
    Ok(())
}

#[test]
fn held_intervals_and_active_volume_representation_reject_atomically() -> Result {
    let batch = StructureBatch::new(vec![element(
        "orifice",
        orifice(0.6, 2.0, 9.81)?,
        DirectionPolicy::Bidirectional,
    )])?;
    let proposal = batch.propose(&[heads("orifice", 3.0, 2.0)])?;
    let before = format!("{:?}", proposal.outputs()?);
    for (start, end) in [
        (-1.0, 1.0),
        (0.0, 0.0),
        (2.0, 1.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (f64::INFINITY, f64::INFINITY),
    ] {
        assert!(
            proposal
                .held_volumes(HeldHeadInterval { start, end })
                .is_err()
        );
        assert_eq!(format!("{:?}", proposal.outputs()?), before);
    }
    for (id, end) in [
        ("weir_normalized_gap_underflow", 1e100),
        ("weir_coefficient_product_underflow", f64::from_bits(1)),
    ] {
        let row = catalog()?
            .into_iter()
            .find(|row| row.id == id)
            .ok_or("Missing held range fixture")?;
        let batch =
            StructureBatch::new(vec![element(id, row.law, DirectionPolicy::Bidirectional)])?;
        let proposal = batch.propose(&[heads(id, row.high, row.low)])?;
        let before = format!("{:?}", proposal.outputs()?);
        let error = rejection(proposal.held_volumes(HeldHeadInterval { start: 0.0, end }))?;
        assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        assert_eq!(error.element(), Some(0));
        assert_eq!(format!("{:?}", proposal.outputs()?), before);
    }
    Ok(())
}

#[test]
fn repeated_and_rejected_proposals_preserve_earlier_projections() -> Result {
    let batch = StructureBatch::new(vec![element(
        "orifice",
        orifice(0.6, 2.0, 9.81)?,
        DirectionPolicy::Bidirectional,
    )])?;
    let definitions = format!("{:?}", batch.elements());
    let packets = [heads("orifice", 3.0, 2.0)];
    let proposal = batch.propose(&packets)?;
    let original = format!("{:?}", proposal.outputs()?);
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: 2.0,
    })?;
    let held_original = format!("{:?}", held.outputs()?);
    assert_eq!(
        format!("{:?}", batch.propose(&packets)?.outputs()?),
        original
    );
    let _reverse = batch.propose(&[heads("orifice", 2.0, 3.0)])?;
    assert!(
        batch
            .propose(&[packet(
                "orifice",
                HeadSupport::Unknown,
                HeadSupport::Unknown
            )])
            .is_err()
    );
    assert_eq!(format!("{:?}", batch.elements()), definitions);
    assert_eq!(format!("{:?}", proposal.outputs()?), original);
    assert_eq!(format!("{:?}", held.outputs()?), held_original);
    Ok(())
}

#[test]
fn active_discharge_rejections_preserve_existing_zero_proposals() -> Result {
    for row in catalog()?.into_iter().filter(|row| row.discharge.is_none()) {
        let batch = StructureBatch::new(vec![element(
            row.id,
            row.law,
            DirectionPolicy::Bidirectional,
        )])?;
        let equal_stage = row.high;
        let original = batch.propose(&[heads(row.id, equal_stage, equal_stage)])?;
        let before = original.outputs()?;
        assert_eq!(first(&original)?.status, StructureStatus::PhysicalZero);
        let error = rejection(batch.propose(&[heads(row.id, row.high, row.low)]))?;
        assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        assert_eq!(original.outputs()?, before);
        assert_eq!(
            batch
                .propose(&[heads(row.id, equal_stage, equal_stage)])?
                .outputs()?,
            before
        );
    }
    Ok(())
}

#[test]
fn zero_and_blocked_lanes_retain_exact_zero_at_large_duration() -> Result {
    for count in [1, 31, 32, 33, 64, 65] {
        let mut elements = Vec::new();
        let mut packets = Vec::new();
        for lane in 0..count {
            let id = format!("zero-{lane}");
            let blocked = lane % 2 == 1;
            elements.push(element(
                &id,
                weir(1.0, 1.0, 0.0)?,
                if blocked {
                    DirectionPolicy::BtoAOnly
                } else {
                    DirectionPolicy::Bidirectional
                },
            ));
            packets.push(heads(&id, if blocked { 1.0 } else { 0.0 }, 0.0));
        }
        let batch = StructureBatch::new(elements)?;
        let proposal = batch.propose(&packets)?;
        let outputs = proposal
            .held_volumes(HeldHeadInterval {
                start: 0.0,
                end: f64::MAX,
            })?
            .outputs()?;
        assert_eq!(outputs.len(), count);
        for (lane, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.volume_m3, 0.0);
            assert_eq!(output.proposal.discharge_m3_per_s, 0.0);
            assert_eq!(
                output.proposal.status,
                if lane % 2 == 1 {
                    StructureStatus::FlapBlocked
                } else {
                    StructureStatus::PhysicalZero
                }
            );
            assert!(output.proposal.hydraulic_operands.is_none());
        }
    }
    Ok(())
}
