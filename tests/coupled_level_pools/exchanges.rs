use super::fixture::{Result, assemble, near, orifice_case};
use super::laws::{passive, pump, running, weir};
use hydr::coupling::{
    CoupledLevelPoolBatch, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledLevelPoolStepDiagnostics,
};
use hydr::structures::{PumpFreeOutlet, StructureElement, StructureLaw};

fn advance(
    setup: CoupledLevelPoolSetup,
    initial: CoupledLevelPoolInitialState,
    end: f64,
    transfers: &[f64],
    volumes: &[f64],
) -> Result<(
    CoupledLevelPoolBatch<crate::runtime::Runtime>,
    CoupledLevelPoolStepDiagnostics,
)> {
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let original = batch.snapshot_data()?;
    let trial = batch.trial_to(end)?;
    assert_eq!(batch.snapshot_data()?, original);
    let next = trial.into_next();
    assert_eq!(next.time(), end);
    let step = next
        .last_step()?
        .ok_or("Missing accepted coupled interval")?;
    assert_eq!(step.end_s, end);
    assert_eq!(step.requested_end_s, end);
    assert!(step.head_residual_m <= 1e-10);
    assert!(step.transfer_residual_m3 <= 1e-10);
    assert!(step.mass_residual_m3.abs() <= 1e-9 + 1e-12 * volumes.iter().sum::<f64>());
    assert!(step.aggregation_roundoff_bound_m3.is_finite());
    assert!(step.aggregation_roundoff_bound_m3 >= 0.0);
    assert_eq!(step.transfers.len(), transfers.len());
    for (actual, &expected) in step.transfers.iter().zip(transfers) {
        near(actual.signed_volume_m3, expected);
        assert!((0.0..=1.0).contains(&actual.donor_limit_factor));
    }
    let states = next.pools().states()?;
    assert_eq!(states.len(), volumes.len());
    for (state, &expected) in states.iter().zip(volumes) {
        assert!(state.volume_m3 >= 0.0);
        near(state.volume_m3, expected);
    }
    Ok((next, step))
}

#[test]
fn submerged_exchange_and_reversal_match_the_midpoint_quadratic() -> Result {
    for (initial, signed) in [([4.0, 2.0], 1.0), ([2.0, 4.0], -1.0)] {
        let (setup, state) = orifice_case(initial)?;
        let (_, step) = advance(setup, state, 1.0, &[signed], &[3.0, 3.0])?;
        let transfer = &step.transfers[0];
        near(transfer.proposed_volume_m3, signed);
        near(transfer.donor_limit_factor, 1.0);
        let midpoint_a = transfer.midpoint_stage_a_m.ok_or("Missing wet A stage")?;
        let midpoint_b = transfer.midpoint_stage_b_m.ok_or("Missing wet B stage")?;
        near(midpoint_a, (initial[0] + 3.0) * 0.5);
        near(midpoint_b, (initial[1] + 3.0) * 0.5);
    }
    Ok(())
}

#[test]
fn linear_pump_curve_advances_both_pools_with_one_volume() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[4.0, 2.0],
        vec![pump("pump", "a", "b", &[(-4.0, 7.0), (3.0, 0.0)])?],
        vec![running("pump")],
    )?;
    let (_, step) = advance(setup, initial, 0.25, &[1.0], &[3.0, 3.0])?;
    near(step.transfers[0].proposed_volume_m3, 1.0);
    near(step.transfers[0].donor_limit_factor, 1.0);
    Ok(())
}

#[test]
fn held_gate_opening_matches_the_submerged_midpoint_quadratic() -> Result {
    use super::fixture::DATUM;
    use hydr::structures::{
        DirectionPolicy, GateApplicability, GateParameters, StructureCondition, StructureElement,
        StructureLaw, StructureOperatingPoint, StructureOperation,
    };

    let gate = StructureElement {
        id: "gate".into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
            2.0,
            0.0,
            1.0,
            1.0,
            0.5,
            GateApplicability {
                coefficient_provenance:
                    "Synthetic unit coefficient with explicitly artificial gravity 0.5 m/s²".into(),
                hydrostatic_quiescent_pool_assertion:
                    "Quiescent synthetic pools share one vertical datum".into(),
                free_submerged_aperture_applicability:
                    "Both stages remain above the held half-metre aperture".into(),
                unobstructed_aperture_assertion:
                    "Both connections expose the full two-metre-wide aperture".into(),
                symmetric_reversal: true,
            },
        )?),
        condition: StructureCondition::AsBuilt,
    };
    let operation = StructureOperation {
        structure_id: "gate".into(),
        operating_point: StructureOperatingPoint::GateOpening { opening_m: 0.5 },
    };
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", 0.0)],
        &[4.0, 2.0],
        vec![gate],
        vec![operation],
    )?;
    let (_, step) = advance(setup, initial, 1.0, &[1.0], &[3.0, 3.0])?;
    let transfer = &step.transfers[0];
    near(transfer.proposed_volume_m3, 1.0);
    near(transfer.donor_limit_factor, 1.0);
    near(
        transfer
            .midpoint_stage_a_m
            .ok_or("Missing gate A midpoint")?,
        3.5,
    );
    near(
        transfer
            .midpoint_stage_b_m
            .ok_or("Missing gate B midpoint")?,
        2.5,
    );
    Ok(())
}

#[test]
fn parallel_weirs_share_one_donor_budget_without_erasing_a_positive_film() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0), ("c", -10.0)],
        &[1.0, 1.0, 1.0],
        vec![
            weir("wide", "a", "b", 0.0, 4.0)?,
            weir("narrow", "a", "c", 0.0, 2.0)?,
        ],
        vec![passive("wide"), passive("narrow")],
    )?;
    let (next, step) = advance(
        setup,
        initial,
        1.0,
        &[2.0 / 3.0, 1.0 / 3.0],
        &[0.0, 5.0 / 3.0, 4.0 / 3.0],
    )?;
    let wide = &step.transfers[0];
    let narrow = &step.transfers[1];
    assert_eq!(
        wide.donor_limit_factor.to_bits(),
        narrow.donor_limit_factor.to_bits()
    );
    assert!(wide.donor_limit_factor > 0.0 && wide.donor_limit_factor < 1.0);
    near(wide.signed_volume_m3, 2.0 * narrow.signed_volume_m3);
    let donor = next.pools().outputs()?.remove(0);
    if donor.volume_m3 > 0.0 {
        assert!(donor.stage_m.is_some());
    } else {
        assert_eq!(donor.stage_m, None);
    }
    Ok(())
}

#[test]
fn incoming_water_does_not_finance_the_middle_pools_outlet() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0), ("c", -20.0)],
        &[1.0, 1.0, 1.0],
        vec![
            weir("upstream", "a", "b", 0.0, 6.0)?,
            weir("downstream", "b", "c", -10.0, 6.0)?,
        ],
        vec![passive("upstream"), passive("downstream")],
    )?;
    advance(setup, initial, 1.0, &[1.0, 1.0], &[0.0, 1.0, 2.0])?;
    Ok(())
}

#[test]
fn known_dry_receiver_uses_explicit_support_until_it_wets() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0)],
        &[1.25, 0.0],
        vec![weir("weir", "a", "b", 0.0, 1.0)?],
        vec![passive("weir")],
    )?;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup.clone(),
        initial.clone(),
    )?;
    assert_eq!(batch.pools().outputs()?[1].stage_m, None);
    let (next, step) = advance(setup, initial, 0.5, &[0.5], &[0.75, 0.5])?;
    near(
        next.pools().outputs()?[1]
            .stage_m
            .ok_or("Receiver remains dry")?,
        -9.5,
    );
    near(
        step.transfers[0]
            .midpoint_stage_b_m
            .ok_or("Missing wetted midpoint")?,
        -9.75,
    );
    Ok(())
}

fn pump_with_outlet(
    id: &str,
    a: &str,
    b: &str,
    knots: &[(f64, f64)],
    elevation_m: f64,
) -> Result<StructureElement> {
    let mut element = pump(id, a, b, knots)?;
    let StructureLaw::Pump(parameters) = element.law else {
        return Err("fixture pump law is not a pump".into());
    };
    element.law = StructureLaw::Pump(parameters.with_free_outlet(PumpFreeOutlet {
        elevation_m,
        evidence:
            "Synthetic delivery pipe discharging to atmosphere above the receiving bed".into(),
    })?);
    Ok(element)
}

#[test]
fn free_outlet_pump_fills_a_dry_pool_and_a_plain_pump_rejects_dry_delivery() -> Result {
    // A plateau curve over the whole lift range: one cubic metre per second.
    let knots = [(-20.0, 1.0), (20.0, 1.0)];
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0)],
        &[4.0, 0.0],
        vec![pump_with_outlet("pump", "a", "b", &knots, 0.0)?],
        vec![running("pump")],
    )?;
    let (next, step) = advance(setup, initial, 0.5, &[0.5], &[3.5, 0.5])?;
    near(step.transfers[0].donor_limit_factor, 1.0);
    near(
        next.pools().outputs()?[1]
            .stage_m
            .ok_or("Receiver remains dry")?,
        -9.5,
    );
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0)],
        &[4.0, 0.0],
        vec![pump("pump", "a", "b", &knots)?],
        vec![running("pump")],
    )?;
    let batch = CoupledLevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        setup,
        initial,
    )?;
    let error = batch
        .trial_to(0.5)
        .err()
        .ok_or("plain pump delivered to a dry pool")?;
    assert!(
        error
            .to_string()
            .contains("a known wet receiver or a declared free outlet"),
        "{error}"
    );
    Ok(())
}

#[test]
fn represented_positive_film_retains_its_water_and_stage_below_an_outlet() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0)],
        &[1e-20, 1.0],
        vec![weir("raised", "a", "b", 1.0, 1.0)?],
        vec![passive("raised")],
    )?;
    let (next, _) = advance(setup, initial, 1.0, &[0.0], &[1e-20, 1.0])?;
    let output = next.pools().outputs()?.remove(0);
    assert_eq!(output.volume_m3.to_bits(), 1e-20_f64.to_bits());
    assert_eq!(output.stage_m, Some(1e-20));
    Ok(())
}

#[test]
fn permuting_parallel_connections_preserves_named_transfers_and_storage() -> Result {
    let (setup, initial) = assemble(
        &[("a", 0.0), ("b", -10.0), ("c", -10.0)],
        &[1.0, 1.0, 1.0],
        vec![
            weir("wide", "a", "b", 0.0, 4.0)?,
            weir("narrow", "a", "c", 0.0, 2.0)?,
        ],
        vec![passive("wide"), passive("narrow")],
    )?;
    let mut reversed = setup.clone();
    reversed.structures.reverse();
    reversed.operations.reverse();
    reversed.connections.reverse();
    let (first, a) = advance(
        setup,
        initial.clone(),
        1.0,
        &[2.0 / 3.0, 1.0 / 3.0],
        &[0.0, 5.0 / 3.0, 4.0 / 3.0],
    )?;
    let (second, b) = advance(
        reversed,
        initial,
        1.0,
        &[1.0 / 3.0, 2.0 / 3.0],
        &[0.0, 5.0 / 3.0, 4.0 / 3.0],
    )?;
    for (left, right) in first.pools().states()?.iter().zip(second.pools().states()?) {
        near(left.volume_m3, right.volume_m3);
    }
    for transfer in &a.transfers {
        let other = b
            .transfers
            .iter()
            .find(|other| other.structure_id == transfer.structure_id)
            .ok_or("Missing named transfer after permutation")?;
        near(transfer.signed_volume_m3, other.signed_volume_m3);
    }
    Ok(())
}
