use super::fixture::{Result, geometry, parameters, single};
use hydr::runoff::{
    ReservoirBatch, ReservoirElement, ReservoirInput, ReservoirState, Subarea, SubcatchmentConfig,
    SubcatchmentInput, SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::{ModifiedGreenAmptBatch, SoilInput};

#[test]
fn direct_lossless_subarea_reproduces_the_reservoir_primitive() -> Result {
    let p = parameters(1.0, 0.0, SubcatchmentRouting::Direct)?;
    let closure = p
        .closure(Subarea::ImperviousStorage)
        .ok_or("missing impervious closure")?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m[1] = 0.25 * 0.3048;
    let config = SubcatchmentConfig::default();
    let primitive = ReservoirBatch::new(
        vec![ReservoirElement {
            id: "storage".into(),
            parameters: closure,
        }],
        vec![ReservoirState {
            depth_m: state.depths_m[1],
        }],
        0.0,
        config.reservoir.clone(),
    )?;
    let component = single(p, state, config)?;
    let rain = 0.3048 / 1024.0;
    let primitive = primitive
        .trial_to(
            64.0,
            &[ReservoirInput {
                incoming_rate_m_per_s: rain,
            }],
        )?
        .into_next();
    let component = component
        .trial_to(
            64.0,
            &[SubcatchmentInput {
                rain_rate_m_per_s: rain,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            }],
        )?
        .into_next();
    assert_eq!(
        component.states()?[0].depths_m[1].to_bits(),
        primitive.states()?[0].depth_m.to_bits()
    );
    assert!(component.states()?[0].soil.is_none());
    assert_eq!(
        component.ledger().storage_m3.to_bits(),
        primitive.ledger().storage_m3.to_bits()
    );
    assert_eq!(
        component.ledger().applied_rain_volume_m3.to_bits(),
        primitive.ledger().incoming_volume_m3.to_bits()
    );
    assert_eq!(
        component.ledger().outlet_volume_m3.to_bits(),
        primitive.ledger().export_volume_m3.to_bits()
    );
    let component_step = component.last_step()?.ok_or("missing component step")?;
    let primitive_step = primitive.last_step()?.ok_or("missing primitive step")?;
    let source = component_step.receipts[0].subareas[1]
        .as_ref()
        .ok_or("missing impervious receipt")?;
    let reference = &primitive_step.receipts[0];
    assert!(source.direct_timing);
    assert_eq!(
        source.drainage.root_lower_m.to_bits(),
        reference.root_lower_m.to_bits()
    );
    assert_eq!(
        source.drainage.root_upper_m.to_bits(),
        reference.root_upper_m.to_bits()
    );
    assert_eq!(
        source.drainage.export_volume_m3.to_bits(),
        reference.export_volume_m3.to_bits()
    );
    assert_eq!(
        source.drainage.endpoint_discharge_m3_per_s,
        reference.endpoint_discharge_m3_per_s
    );
    Ok(())
}

#[test]
fn native_soil_receives_original_ponding_and_rain_once() -> Result {
    let mut g = geometry(0.0, 0.0);
    g.width_m = 0.0;
    let soil = parameters(0.0, 0.0, SubcatchmentRouting::Direct)?
        .soil_parameters()
        .ok_or("missing soil parameters")?;
    let p = SubcatchmentParameters::new(g, Some(soil), SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m[2] = 0.125 * 0.3048;
    let native = state.soil.ok_or("missing native antecedent")?;
    let rain = 0.3048 / 2048.0;
    let raw = ModifiedGreenAmptBatch::from_native_states(&[soil], &[native])?.trial(
        64.0,
        &[SoilInput {
            incoming_rate_m_per_s: rain,
            ponded_depth_m: state.depths_m[2],
        }],
    )?;
    let expected = raw.into_next().native_states()?[0];
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let next = batch
        .trial_to(
            64.0,
            &[SubcatchmentInput {
                rain_rate_m_per_s: rain,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            }],
        )?
        .into_next();
    assert_eq!(next.states()?[0].soil, Some(expected));
    Ok(())
}

#[test]
fn pervious_loss_priority_retains_native_soil_and_caps_evaporation() -> Result {
    let p = parameters(0.0, 0.0, SubcatchmentRouting::Direct)?;
    let soil = p.soil_parameters().ok_or("missing pervious parameters")?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m[2] = 0.125 * 0.3048;
    let native = state.soil.ok_or("missing pervious antecedent")?;
    let raw = ModifiedGreenAmptBatch::from_native_states(&[soil], &[native])?.trial(
        128.0,
        &[SoilInput {
            incoming_rate_m_per_s: 0.0,
            ponded_depth_m: state.depths_m[2],
        }],
    )?;
    let expected_infiltration = raw.outputs()?[0].depth_m * p.area_m2(Subarea::Pervious);
    let expected = raw.into_next().native_states()?[0];
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let next = batch
        .trial_to(
            128.0,
            &[SubcatchmentInput {
                rain_rate_m_per_s: 0.0,
                potential_evaporation_m_per_s: 0.3048 / 1024.0,
                external_runon_volumes_m3: [0.0; 3],
            }],
        )?
        .into_next();
    assert_eq!(next.states()?[0].depths_m, [0.0; 3]);
    assert_eq!(next.states()?[0].soil, Some(expected));
    assert_eq!(
        next.ledger().infiltration_volume_m3.to_bits(),
        expected_infiltration.to_bits()
    );
    assert_eq!(next.ledger().evaporation_volume_m3, 0.0);
    assert_eq!(next.ledger().outlet_volume_m3, 0.0);
    Ok(())
}

#[test]
fn explicit_destination_runon_keeps_the_original_volume_authoritative() -> Result {
    let mut g = geometry(1.0, 0.0);
    g.total_area_m2 = 3.0;
    g.width_m = 0.0;
    let p = SubcatchmentParameters::new(g, None, SubcatchmentRouting::Direct)?;
    let state = SubcatchmentState::initialized(&p)?;
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let supplied = 0.1_f64;
    let trial = batch.trial_to(
        3.0,
        &[SubcatchmentInput {
            rain_rate_m_per_s: 0.0,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0, supplied, 0.0],
        }],
    )?;
    let step = trial.diagnostics()?;
    let source = step.receipts[0].subareas[1]
        .as_ref()
        .ok_or("missing supported destination")?;
    assert_eq!(source.external_volume_m3.to_bits(), supplied.to_bits());
    assert_eq!(source.bridge.merged_volume_m3.to_bits(), supplied.to_bits());
    assert!(source.bridge.residual_m3.abs() <= source.bridge.residual_bound_m3);
    let next = trial.into_next();
    assert_eq!(
        next.ledger().external_runon_volume_m3.to_bits(),
        supplied.to_bits()
    );
    assert_eq!(next.ledger().soil_water_adjustment_volume_m3, 0.0);
    assert!(next.ledger().transfer_uncertainty_charge_m3 > 0.0);
    assert!((next.ledger().storage_m3 - supplied).abs() <= 1e-9 + 1e-12 * supplied);
    Ok(())
}
