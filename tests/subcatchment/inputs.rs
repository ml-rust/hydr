use super::fixture::{Result, geometry, parameters, single};
use hydr::runoff::{
    Subarea, SubcatchmentBatch, SubcatchmentConfig, SubcatchmentDefinition, SubcatchmentInput,
    SubcatchmentParameters, SubcatchmentRouting, SubcatchmentState,
};
use hydr::soil::ModifiedGreenAmptParameters;

#[test]
fn subarea_order_and_fraction_areas_preserve_class_coefficients() -> Result {
    let p = parameters(0.5, 0.5, SubcatchmentRouting::Direct)?;
    for (index, subarea) in Subarea::ALL.into_iter().enumerate() {
        assert_eq!(subarea.index(), index);
        assert_eq!(Subarea::from_index(index), Some(subarea));
    }
    assert_eq!(Subarea::from_index(3), None);
    let total = p.geometry().total_area_m2;
    assert_eq!(p.area_m2(Subarea::ImperviousZeroStorage), total / 4.0);
    assert_eq!(p.area_m2(Subarea::ImperviousStorage), total / 4.0);
    assert_eq!(p.area_m2(Subarea::Pervious), total / 2.0);
    let zero = p
        .closure(Subarea::ImperviousZeroStorage)
        .ok_or("missing zero-storage closure")?;
    let stored = p
        .closure(Subarea::ImperviousStorage)
        .ok_or("missing impervious closure")?;
    assert_eq!(zero.class_area_m2(), total / 2.0);
    assert_eq!(stored.class_area_m2(), total / 2.0);
    assert_eq!(zero.coefficient_si(), stored.coefficient_si());
    assert_eq!(zero.depression_depth_m(), 0.0);
    Ok(())
}

#[test]
fn absent_pervious_support_has_no_soil_memory_or_soil_call() -> Result {
    let p = parameters(1.0, 0.0, SubcatchmentRouting::Direct)?;
    assert_eq!(p.area_m2(Subarea::Pervious), 0.0);
    assert!(p.closure(Subarea::Pervious).is_none());
    assert!(p.soil_parameters().is_none());
    let state = SubcatchmentState::initialized(&p)?;
    assert!(state.soil.is_none());
    let batch = single(p, state, SubcatchmentConfig::default())?;
    let next = batch
        .trial_to(
            1.0,
            &[SubcatchmentInput {
                rain_rate_m_per_s: 0.3048 / 1024.0,
                potential_evaporation_m_per_s: 0.0,
                external_runon_volumes_m3: [0.0; 3],
            }],
        )?
        .into_next();
    assert!(next.states()?[0].soil.is_none());
    assert_eq!(next.states()?[0].depths_m[2], 0.0);
    let receipt = next.last_step()?.ok_or("missing accepted step")?;
    assert!(receipt.receipts[0].subareas[0].is_none());
    assert!(receipt.receipts[0].subareas[1].is_some());
    assert!(receipt.receipts[0].subareas[2].is_none());
    Ok(())
}

#[test]
fn soil_presence_and_positive_routing_require_active_destinations() -> Result {
    let soil = ModifiedGreenAmptParameters::from_native(0.125, 1.0 / 1024.0, 1.0)?;
    assert!(
        SubcatchmentParameters::new(geometry(1.0, 0.5), Some(soil), SubcatchmentRouting::Direct)
            .is_err()
    );
    assert!(
        SubcatchmentParameters::new(geometry(0.0, 0.5), None, SubcatchmentRouting::Direct).is_err()
    );
    assert!(
        parameters(
            1.0,
            0.5,
            SubcatchmentRouting::ImperviousToPervious { fraction: 0.5 }
        )
        .is_err()
    );
    assert!(
        parameters(
            0.0,
            0.5,
            SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 }
        )
        .is_err()
    );
    assert!(
        parameters(
            0.5,
            1.0,
            SubcatchmentRouting::PerviousToImpervious { fraction: 0.5 }
        )
        .is_err()
    );
    for fraction in [-0.1, 1.1, f64::NAN] {
        assert!(
            parameters(
                0.5,
                0.5,
                SubcatchmentRouting::ImperviousToPervious { fraction }
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn invalid_geometry_and_absent_class_water_reject_explicitly() -> Result {
    for (impervious, zero_storage) in [(-0.1, 0.5), (1.1, 0.5), (0.5, -0.1), (0.5, f64::NAN)] {
        assert!(parameters(impervious, zero_storage, SubcatchmentRouting::Direct).is_err());
    }
    let p = parameters(1.0, 0.0, SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.depths_m[0] = 1.0;
    assert!(single(p, state, SubcatchmentConfig::default()).is_err());
    let p = parameters(0.0, 0.0, SubcatchmentRouting::Direct)?;
    let mut state = SubcatchmentState::initialized(&p)?;
    state.soil = None;
    assert!(single(p, state, SubcatchmentConfig::default()).is_err());
    Ok(())
}

#[test]
fn identities_shapes_and_forcing_reject_before_any_accepted_change() -> Result {
    let p = parameters(1.0, 0.0, SubcatchmentRouting::Direct)?;
    let initialized = SubcatchmentState::initialized(&p)?;
    for ids in [
        vec![],
        vec![""],
        vec!["invalid\nidentifier"],
        vec!["duplicate", "duplicate"],
    ] {
        let definitions = ids
            .iter()
            .map(|id| SubcatchmentDefinition {
                id: (*id).into(),
                parameters: p.clone(),
            })
            .collect();
        let states = vec![initialized; ids.len()];
        assert!(
            SubcatchmentBatch::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                definitions,
                states,
                0.0,
                SubcatchmentConfig::default()
            )
            .is_err()
        );
    }
    let batch = single(p, initialized, SubcatchmentConfig::default())?;
    let before = batch.snapshot_data()?;
    for inputs in [
        vec![],
        vec![SubcatchmentInput {
            rain_rate_m_per_s: -1.0,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0; 3],
        }],
        vec![SubcatchmentInput {
            rain_rate_m_per_s: 0.0,
            potential_evaporation_m_per_s: f64::NAN,
            external_runon_volumes_m3: [0.0; 3],
        }],
        vec![SubcatchmentInput {
            rain_rate_m_per_s: 0.0,
            potential_evaporation_m_per_s: 0.0,
            external_runon_volumes_m3: [0.0, 0.0, 1.0],
        }],
    ] {
        assert!(batch.trial_to(1.0, &inputs).is_err());
        assert_eq!(batch.snapshot_data()?, before);
    }
    Ok(())
}
