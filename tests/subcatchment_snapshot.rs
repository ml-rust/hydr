//! Complete subcatchment continuation, source receipts, and numerical budget admission.

#[path = "subcatchment/fixture.rs"]
mod fixture;

use fixture::{Result, geometry, parameters, single};
use hydr::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentError, SubcatchmentInput,
    SubcatchmentLedger, SubcatchmentParameters, SubcatchmentRouting, SubcatchmentSnapshotData,
    SubcatchmentState,
};
use hydr::soil::{ModifiedGreenAmptParameters, NativeSoilState};

fn create() -> Result<SubcatchmentBatch> {
    let p = parameters(
        0.5,
        0.5,
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    )?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    single(p, state, SubcatchmentConfig::default())
}

fn input(rain: f64) -> SubcatchmentInput {
    SubcatchmentInput {
        rain_rate_m_per_s: rain,
        potential_evaporation_m_per_s: 0.0,
        external_runon_volumes_m3: [0.0; 3],
    }
}

fn direct_history() -> Result<SubcatchmentBatch> {
    let p = parameters(0.5, 0.5, SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    let mut batch = single(p, state, SubcatchmentConfig::default())?;
    for end in [32.0, 64.0] {
        batch = batch.trial_to(end, &[input(0.0)])?.into_next();
    }
    Ok(batch)
}

fn recalculate_residual(ledger: &mut SubcatchmentLedger) {
    ledger.residual_m3 = (ledger.storage_m3 - ledger.initial_storage_m3)
        - (ledger.applied_rain_volume_m3 + ledger.external_runon_volume_m3
            - ledger.infiltration_volume_m3
            - ledger.evaporation_volume_m3
            - ledger.outlet_volume_m3
            + ledger.soil_water_adjustment_volume_m3);
}

fn rejects_charge(
    destination: &mut SubcatchmentBatch,
    package: &SubcatchmentSnapshotData,
    expected: &str,
) -> Result {
    let error = match destination.restore_data(package) {
        Ok(()) => return Err("historical charge forgery requires rejection".into()),
        Err(error) => error,
    };
    assert!(matches!(error, SubcatchmentError::Numerical { .. }));
    assert!(
        error.to_string().contains(expected),
        "unexpected rejection: {error}"
    );
    Ok(())
}

#[test]
fn trusted_and_owned_restarts_preserve_complete_soil_transfer_and_budget_state() -> Result {
    let mut continuous = create()?
        .trial_to(64.0, &[input(0.3048 / 1024.0)])?
        .into_next();
    let snapshot = continuous.snapshot();
    let package = continuous.snapshot_data()?;
    let mut trusted = create()?;
    let mut imported = create()?;
    trusted.restore(&snapshot)?;
    imported.restore_data(&package)?;
    assert_eq!(imported.snapshot_data()?, package);
    for (end, rain) in [(128.0, 0.0), (192.0, 0.3048 / 1024.0), (256.0, 0.0)] {
        let inputs = [input(rain)];
        continuous = continuous.trial_to(end, &inputs)?.into_next();
        trusted = trusted.trial_to(end, &inputs)?.into_next();
        imported = imported.trial_to(end, &inputs)?.into_next();
        assert_eq!(continuous.snapshot_data()?, trusted.snapshot_data()?);
        assert_eq!(continuous.snapshot_data()?, imported.snapshot_data()?);
    }
    Ok(())
}

#[test]
fn signed_native_memory_and_zero_bits_survive_initial_owned_restore() -> Result {
    let soil = ModifiedGreenAmptParameters::from_native(0.125, 1.0 / 1024.0, 0.0)?;
    let p =
        SubcatchmentParameters::new(geometry(0.5, 0.5), Some(soil), SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.soil = Some(NativeSoilState {
        imd: -0.0,
        f_ft: -0.0,
        fu_ft: -0.0,
        saturated: false,
        t_s: -0.0,
    });
    let original = single(p.clone(), state, SubcatchmentConfig::default())?;
    let initial = SubcatchmentState::initialized(&p)?;
    let mut destination = single(p, initial, SubcatchmentConfig::default())?;
    destination.restore_data(&original.snapshot_data()?)?;
    let native = destination.states()?[0]
        .soil
        .ok_or("missing restored native state")?;
    for value in [native.imd, native.f_ft, native.fu_ft, native.t_s] {
        assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
    }
    assert_eq!(destination.ledger().infiltration_volume_m3, 0.0);
    Ok(())
}

#[test]
fn forged_source_counters_state_and_physical_scale_reject_atomically() -> Result {
    let source = create()?
        .trial_to(64.0, &[input(0.3048 / 1024.0)])?
        .into_next();
    let good = source.snapshot_data()?;
    assert!(good.ledger.transfer_uncertainty_charge_m3 > 0.0);
    let changes: &[fn(&mut SubcatchmentSnapshotData)] = &[
        |data| data.time = f64::NAN,
        |data| data.time = 32.0,
        |data| data.states.clear(),
        |data| data.states[0].depths_m[1] = -1.0,
        |data| data.states[0].soil = None,
        |data| data.ledger.storage_m3 += 1.0,
        |data| data.ledger.infiltration_volume_m3 = 0.0,
        |data| data.ledger.evaporation_volume_m3 = -1.0,
        |data| data.ledger.outlet_volume_m3 = -1.0,
        |data| data.ledger.internal_transfer_volume_m3 = 0.0,
        |data| data.ledger.transfer_uncertainty_charge_m3 = 0.0,
        |data| data.ledger.physical_scale_lower_m3 += 1.0,
        |data| data.ledger.soil_water_adjustment_abs_bound_m3 = -1.0,
        |data| data.ledger.residual_m3 = 1.0,
        |data| data.accepted_steps = 0,
        |data| data.last_step = None,
    ];
    let mut destination = create()?;
    let before = destination.snapshot_data()?;
    for (index, change) in changes.iter().enumerate() {
        let mut corrupted = good.clone();
        change(&mut corrupted);
        assert!(
            destination.restore_data(&corrupted).is_err(),
            "package corruption {index}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn forged_interface_bridge_soil_and_drainage_receipts_reject_atomically() -> Result {
    let source = create()?
        .trial_to(64.0, &[input(0.3048 / 1024.0)])?
        .into_next();
    let good = source.snapshot_data()?;
    let mut destination = create()?;
    let before = destination.snapshot_data()?;
    for quantity in 0..23 {
        let mut corrupted = good.clone();
        let step = corrupted
            .last_step
            .as_mut()
            .ok_or("missing accepted step")?;
        let catchment = step
            .receipts
            .first_mut()
            .ok_or("missing catchment receipt")?;
        let receiver = catchment.subareas[2]
            .as_mut()
            .ok_or("missing pervious receipt")?;
        match quantity {
            0 => step.dt *= 2.0,
            1 => step.root_iterations = usize::MAX,
            2 => step.transfer_uncertainty_charge_m3 = 0.0,
            3 => step.physical_scale_lower_m3 += 1.0,
            4 => step.phase2 = None,
            5 => step.transfers[0].volume_m3 += 1.0,
            6 => step.transfers[0].outlet_volume_m3 += 1.0,
            7 => step.transfers.clear(),
            8 => catchment.input.rain_rate_m_per_s = 1.0,
            9 => receiver.old_depth_m = -1.0,
            10 => receiver.received_by_donor_m3[0] += 1.0,
            11 => receiver.bridge.merged_volume_m3 += 1.0,
            12 => receiver.bridge.total_rate_m_per_s = 0.0,
            13 => receiver.bridge.input_depth_m = 0.0,
            14 => receiver.bridge.uncertainty_m3 = -1.0,
            15 => receiver.infiltration_depth_m = -1.0,
            16 => receiver.evaporation_depth_m = 1.0,
            17 => receiver.soil = None,
            18 => receiver.available_depth_m = -1.0,
            19 => receiver.drainage.root_lower_m = 1.0 + receiver.drainage.root_upper_m,
            20 => receiver.drainage.endpoint_rate_m_per_s = None,
            21 => receiver.partition.routed_volume_m3 += 1.0,
            22 => catchment.subareas[2] = None,
            _ => return Err("unknown receipt corruption".into()),
        }
        assert!(
            destination.restore_data(&corrupted).is_err(),
            "receipt corruption {quantity}"
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn exact_definition_identity_includes_supplied_signed_zero_geometry() -> Result {
    let mut g = geometry(0.5, 0.5);
    g.width_m = 0.0;
    let soil = parameters(0.5, 0.5, SubcatchmentRouting::Direct)?.soil_parameters();
    let p = SubcatchmentParameters::new(g, soil, SubcatchmentRouting::Direct)?;
    let state = SubcatchmentState::initialized(&p)?;
    let original = single(p.clone(), state, SubcatchmentConfig::default())?;
    let mut package = original.snapshot_data()?;
    g.width_m = -0.0;
    package.definitions[0].parameters =
        SubcatchmentParameters::new(g, soil, SubcatchmentRouting::Direct)?;
    let mut destination = single(p, state, SubcatchmentConfig::default())?;
    let before = destination.snapshot_data()?;
    assert!(destination.restore_data(&package).is_err());
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn historical_soil_adjustment_requires_a_covering_cumulative_charge() -> Result {
    let source = direct_history()?;
    let mut package = source.snapshot_data()?;
    let step = package.last_step.as_ref().ok_or("missing accepted step")?;
    assert_eq!(step.soil_water_adjustment_volume_m3, 0.0);
    assert_eq!(step.soil_water_adjustment_abs_charge_m3, 0.0);
    assert_eq!(package.ledger.soil_water_adjustment_volume_m3, 0.0);
    package.ledger.soil_water_adjustment_volume_m3 += 1.0;
    package.ledger.outlet_volume_m3 += 1.0;
    package.ledger.soil_water_adjustment_abs_bound_m3 = 1e-12;
    recalculate_residual(&mut package.ledger);
    let mut destination = direct_history()?;
    let before = destination.snapshot_data()?;
    rejects_charge(
        &mut destination,
        &package,
        "reported soil corrections exceed their independent cumulative admission budget",
    )?;
    assert_eq!(destination.snapshot_data()?, before);
    Ok(())
}

#[test]
fn historical_positive_traffic_requires_a_covering_uncertainty_charge() -> Result {
    let good = direct_history()?.snapshot_data()?;
    let step = good.last_step.as_ref().ok_or("missing accepted step")?;
    assert_eq!(step.external_runon_volume_m3, 0.0);
    assert_eq!(step.internal_transfer_volume_m3, 0.0);
    assert_eq!(step.transfer_uncertainty_charge_m3, 0.0);
    assert_eq!(good.ledger.transfer_uncertainty_charge_m3, 0.0);
    let mut destination = direct_history()?;
    let before = destination.snapshot_data()?;
    for external in [true, false] {
        for charge in [0.0, f64::EPSILON / 4.0] {
            let mut package = good.clone();
            if external {
                package.ledger.external_runon_volume_m3 += 1.0;
                package.ledger.outlet_volume_m3 += 1.0;
            } else {
                package.ledger.internal_transfer_volume_m3 += 1.0;
            }
            package.ledger.transfer_uncertainty_charge_m3 = charge;
            recalculate_residual(&mut package.ledger);
            rejects_charge(
                &mut destination,
                &package,
                "cumulative transfer uncertainty charge cannot support declared historical traffic",
            )?;
            assert_eq!(destination.snapshot_data()?, before);
        }
    }
    Ok(())
}
