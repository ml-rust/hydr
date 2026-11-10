use super::fixture::{Result, geometry, parameters, single};
use hydr::runoff::{
    SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentInput,
    SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};

#[test]
fn repeated_trials_leave_every_accepted_component_field_unchanged() -> Result {
    let p = parameters(
        0.5,
        0.5,
        SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
    )?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m = [0.125 * 0.3048, 0.25 * 0.3048, 0.0];
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let before = batch.snapshot_data()?;
    let inputs = [SubcatchmentInput {
        rain_rate_m_per_s: 0.3048 / 1024.0,
        potential_evaporation_m_per_s: 0.0,
        external_runon_volumes_m3: [0.0; 3],
    }];
    let first = batch.trial_to(64.0, &inputs)?;
    let repeat = batch.trial_to(64.0, &inputs)?;
    assert_eq!(first.outputs()?, repeat.outputs()?);
    assert_eq!(first.diagnostics()?, repeat.diagnostics()?);
    assert_eq!(batch.snapshot_data()?, before);
    for end in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(batch.trial_to(end, &inputs).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    let next = first.into_next();
    assert_eq!(next.time(), 64.0);
    assert_eq!(next.accepted_steps(), 1);
    assert_eq!(batch.accepted_steps(), 0);
    Ok(())
}

#[test]
fn recipient_bridge_rejection_discards_completed_donor_work() -> Result {
    let mut g = geometry(0.5, 1.0);
    g.total_area_m2 = 2.0;
    g.width_m = 1.0;
    g.slope = 1.0;
    g.impervious_manning = 0.0;
    let soil = parameters(0.5, 1.0, SubcatchmentRouting::Direct)?.soil_parameters();
    let p = SubcatchmentParameters::new(
        g,
        soil,
        SubcatchmentRouting::ImperviousToPervious { fraction: 1.0 },
    )?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m = [1.0, 0.0, 2.0_f64.powi(53)];
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let before = batch.snapshot_data()?;
    assert!(
        batch
            .trial_to(
                1.0,
                &[SubcatchmentInput {
                    rain_rate_m_per_s: 0.0,
                    potential_evaporation_m_per_s: 0.0,
                    external_runon_volumes_m3: [0.0; 3],
                }]
            )
            .is_err()
    );
    assert_eq!(batch.snapshot_data()?, before);
    Ok(())
}

#[test]
fn mixed_role_batches_advance_each_native_timer_exactly_once() -> Result {
    let models = [
        parameters(0.5, 0.5, SubcatchmentRouting::Direct)?,
        parameters(
            0.5,
            0.5,
            SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 },
        )?,
        parameters(
            0.5,
            0.5,
            SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 },
        )?,
    ];
    for count in [31, 32, 33, 64, 65] {
        let mut definitions = Vec::with_capacity(count);
        let mut states = Vec::with_capacity(count);
        for index in 0..count {
            let parameters = models[index % models.len()].clone();
            let mut state = SubcatchmentState::initialized(&parameters)?;
            let soil = state
                .soil
                .as_mut()
                .ok_or("missing active pervious antecedent")?;
            soil.saturated = true;
            soil.t_s = 10.0 + index as f64;
            definitions.push(SubcatchmentDefinition {
                id: format!("catchment-{index}"),
                parameters,
            });
            states.push(state);
        }
        let batch = SubcatchmentBatch::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            definitions,
            states.clone(),
            0.0,
            SubcatchmentConfig::default(),
        )?;
        let next = batch
            .trial_to(
                1.0,
                &vec![
                    SubcatchmentInput {
                        rain_rate_m_per_s: 0.0,
                        potential_evaporation_m_per_s: 0.0,
                        external_runon_volumes_m3: [0.0; 3],
                    };
                    count
                ],
            )?
            .into_next();
        for (initial, actual) in states.iter().zip(next.states()?) {
            let mut expected = initial.soil.ok_or("missing initial soil")?;
            expected.t_s -= 1.0;
            assert_eq!(actual.soil, Some(expected));
            assert_eq!(actual.depths_m, [0.0; 3]);
        }
        assert_eq!(next.ledger().transfer_uncertainty_charge_m3, 0.0);
        assert_eq!(next.ledger().soil_water_adjustment_abs_bound_m3, 0.0);
    }
    Ok(())
}
