use super::fixture::*;
use hydr::structures::*;

#[test]
fn ordered_gate_operations_precede_unknown_head_isolation() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![
            standard("first", DirectionPolicy::Bidirectional)?,
            standard("second", DirectionPolicy::Bidirectional)?,
        ],
    )?;
    let packets = [
        packet("first", HeadSupport::Unknown, HeadSupport::Unknown),
        packet("second", HeadSupport::Unknown, HeadSupport::Unknown),
    ];
    let operations = [opening("first", 0.0), opening("second", 0.0)];
    let error = rejection(batch.propose(&packets))?;
    assert_eq!(
        error.input_kind(),
        Some(StructureInputErrorKind::OperatingPoint)
    );
    for incomplete in [vec![], vec![operations[0].clone()]] {
        kind(
            &batch,
            &packets,
            &incomplete,
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    for mismatched in [
        vec![operations[0].clone(), operations[0].clone()],
        vec![operations[1].clone(), operations[0].clone()],
        vec![opening("foreign", 0.0), operations[1].clone()],
    ] {
        kind(
            &batch,
            &packets,
            &mismatched,
            StructureInputErrorKind::Identity,
        )?;
    }
    for wrong in [
        StructureOperatingPoint::Passive,
        StructureOperatingPoint::PumpStoppedIsolated,
        StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
    ] {
        kind(
            &batch,
            &packets,
            &[operation("first", wrong), operations[1].clone()],
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    assert_eq!(
        batch
            .propose_with_operations(&packets, &operations)?
            .outputs()?
            .len(),
        2
    );
    Ok(())
}

#[test]
fn opening_requires_finite_declared_range_before_closed_support_waiver() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("gate", DirectionPolicy::Bidirectional)?],
    )?;
    let packets = [packet("gate", HeadSupport::Unknown, HeadSupport::Unknown)];
    for value in [
        -1.0,
        1.0_f64.next_up(),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        kind(
            &batch,
            &packets,
            &[opening("gate", value)],
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    Ok(())
}

#[test]
fn passive_and_pump_laws_cannot_borrow_closed_gate_isolation() -> Result {
    for (law, direction) in [
        (passive_law()?, DirectionPolicy::Bidirectional),
        (pump_law()?, DirectionPolicy::AtoBOnly),
    ] {
        let batch = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![element("other", law, direction)],
        )?;
        kind(
            &batch,
            &[packet("other", HeadSupport::Unknown, HeadSupport::Unknown)],
            &[opening("other", 0.0)],
            StructureInputErrorKind::OperatingPoint,
        )?;
    }
    Ok(())
}
