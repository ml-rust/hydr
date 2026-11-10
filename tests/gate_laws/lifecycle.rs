use super::fixture::*;
use hydr::structures::*;

#[test]
fn gate_proposals_hold_original_opening_and_remain_immutable_after_rejection() -> Result {
    let batch = StructureBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![standard("gate", DirectionPolicy::Bidirectional)?],
    )?;
    let packets = [heads("gate", 2.0, 0.5)];
    let operations = [opening("gate", 0.75)];
    let proposal = batch.propose_with_operations(&packets, &operations)?;
    let before = proposal.outputs()?;
    assert!(first(&proposal)?.discharge_m3_per_s > 0.0);
    let interval = HeldHeadInterval {
        start: 1.0,
        end: 4.0,
    };
    let held = proposal.held_volumes(interval)?;
    let held_before = held.outputs()?;
    for output in &held_before {
        assert_eq!(output.proposal.operation, operations[0].operating_point);
        assert_eq!(output.interval, interval);
        assert_eq!(
            output.volume_m3.to_bits(),
            (output.proposal.discharge_m3_per_s * 3.0).to_bits()
        );
    }
    for wrong in [opening("foreign", 0.75), opening("gate", -1.0)] {
        assert!(batch.propose_with_operations(&packets, &[wrong]).is_err());
    }
    let closed = batch.propose_with_operations(
        &[packet("gate", HeadSupport::Unknown, HeadSupport::Unknown)],
        &[opening("gate", 0.0)],
    )?;
    assert_eq!(first(&closed)?.discharge_m3_per_s, 0.0);
    assert_eq!(proposal.outputs()?, before);
    assert_eq!(held.outputs()?, held_before);
    assert_eq!(
        batch
            .propose_with_operations(&packets, &operations)?
            .outputs()?,
        before
    );
    Ok(())
}

#[test]
fn held_arithmetic_limits_preserve_the_original_gate_proposal() -> Result {
    for (id, duration) in [
        ("positive_subnormal_discharge", 0.25),
        ("coefficient_width_product_overflow", 2.0),
    ] {
        let case = super::reference_inputs::cases()
            .find(|case| case.id == id)
            .ok_or("Missing held gate arithmetic input")?;
        let batch = StructureBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            vec![super::reference_case::definition(case, id)?],
        )?;
        let proposal = batch.propose_with_operations(
            &[super::reference_case::packet(case, id, false)],
            &[super::reference_case::operation(case, id)],
        )?;
        let before = proposal.outputs()?;
        let error = rejection(proposal.held_volumes(HeldHeadInterval {
            start: 0.0,
            end: duration,
        }))?;
        assert!(matches!(error, StructureError::Numerical { .. }), "{error}");
        assert_eq!(error.element(), Some(0));
        assert_eq!(proposal.outputs()?, before);
    }
    Ok(())
}
