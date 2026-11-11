use super::{Batch, Result, profiles::*};
use hydr::channel::{CrossSectionError, CrossSectionRecoveryConfig};

#[test]
fn analytical_areas_recover_ordered_stages_and_geometry() -> Result {
    let batch = Batch::new(crate::runtime::client(), all())?;
    let config = CrossSectionRecoveryConfig::default();
    assert_eq!(config.height_atol_m, 1e-14);
    assert_eq!(config.height_rtol, 1e-14);
    assert_eq!(config.area_atol_m2, 1e-12);
    assert_eq!(config.area_rtol, 1e-12);
    assert_eq!(config.max_iter, 256);
    for h in [0.125, 0.5, 1.0, 1.5] {
        let expected = [
            rectangle_expected(h),
            trapezoid_expected(h),
            triangle_expected(h),
            bench_expected(h),
        ];
        let areas = expected.map(|row| row.area);
        let recovered = batch.recover_stages(&areas, &config)?.properties()?;
        for (index, row) in recovered.iter().enumerate() {
            assert_eq!(row.requested_area_m2.to_bits(), areas[index].to_bits());
            assert_eq!(row.geometry.section_id, batch.definitions()[index].id);
            near(row.geometry.query_stage_m, h);
            near(row.geometry.area_m2, expected[index].area);
            assert!(!row.geometry.dry);
            if index != 3 || h != 1.0 {
                assert_wet(&row.geometry, expected[index])?;
            }
        }
    }
    Ok(())
}

#[test]
fn scaled_profiles_recover_normalized_stages_and_geometry() -> Result {
    let heights = [1.0, 1.0, 1.0, 1.5];
    let expected = [
        rectangle_expected(1.0),
        trapezoid_expected(1.0),
        triangle_expected(1.0),
        bench_expected(1.5),
    ];
    for exponent in [-100, 100] {
        let scale = 2.0_f64.powi(exponent);
        let area_scale = scale * scale;
        let profiles = all()
            .into_iter()
            .map(|mut profile| {
                for point in &mut profile.points {
                    point.station_m *= scale;
                    point.elevation_m *= scale;
                }
                profile.maximum_stage_m *= scale;
                profile
            })
            .collect();
        let batch = Batch::new(crate::runtime::client(), profiles)?;
        let config = CrossSectionRecoveryConfig {
            height_atol_m: 1e-14 * scale,
            area_atol_m2: 1e-12 * area_scale,
            ..Default::default()
        };
        let areas = expected.map(|row| row.area * area_scale);
        let rows = batch.recover_stages(&areas, &config)?.properties()?;
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(row.requested_area_m2.to_bits(), areas[index].to_bits());
            assert_eq!(row.geometry.section_id, batch.definitions()[index].id);
            let normalized_stage = row.geometry.query_stage_m / scale;
            assert!(normalized_stage.is_finite());
            assert!((normalized_stage - heights[index]).abs() <= 1e-12);
            assert_scaled_wet(&row.geometry, expected[index], scale)?;
        }
    }
    Ok(())
}

#[test]
fn zero_targets_retain_requested_bits_and_return_canonical_beds() -> Result {
    for area in [0.0_f64, -0.0] {
        let batch = Batch::new(crate::runtime::client(), all())?;
        let recovered = batch
            .recover_stages(&[area; 4], &CrossSectionRecoveryConfig::default())?
            .properties()?;
        for row in recovered {
            assert_eq!(row.requested_area_m2.to_bits(), area.to_bits());
            assert_dry(&row.geometry, 0.0);
        }
    }
    Ok(())
}

#[test]
fn capacity_targets_retain_caps_and_next_represented_areas_reject() -> Result {
    let expected = [
        rectangle_expected(4.0),
        trapezoid_expected(3.0),
        triangle_expected(3.0),
        bench_expected(2.5),
    ];
    for (profile, expected) in all().into_iter().zip(expected) {
        let cap = profile.maximum_stage_m;
        let batch = Batch::new(crate::runtime::client(), vec![profile])?;
        let rows = batch
            .recover_stages(&[expected.area], &CrossSectionRecoveryConfig::default())?
            .properties()?;
        assert_eq!(rows[0].geometry.query_stage_m.to_bits(), cap.to_bits());
        assert_wet(&rows[0].geometry, expected)?;
        let above = f64::from_bits(expected.area.to_bits() + 1);
        let error = batch
            .recover_stages(&[above], &CrossSectionRecoveryConfig::default())
            .err()
            .ok_or("Above-capacity area was accepted")?;
        assert!(matches!(
            error,
            CrossSectionError::InvalidInput {
                section: Some(0),
                ..
            }
        ));
    }
    Ok(())
}

#[test]
fn capacity_target_retains_cap_operand_after_inexact_local_height_subtraction() -> Result {
    let bed = -1e16_f64;
    let cap = 1.0_f64;
    let height = cap - bed;
    assert_ne!((bed + height).to_bits(), cap.to_bits());
    let profile = definition(
        "inexact-cap-height",
        &[(0.0, cap), (0.0, bed), (2.0, bed), (2.0, cap)],
        cap,
    );
    let batch = Batch::new(crate::runtime::client(), vec![profile])?;
    let capacity = 2.0 * height;
    let rows = batch
        .recover_stages(&[capacity], &CrossSectionRecoveryConfig::default())?
        .properties()?;
    assert_eq!(rows[0].requested_area_m2.to_bits(), capacity.to_bits());
    assert_eq!(rows[0].geometry.query_stage_m.to_bits(), cap.to_bits());
    assert_eq!(rows[0].geometry.area_m2.to_bits(), capacity.to_bits());
    assert!(!rows[0].geometry.dry);
    Ok(())
}

#[test]
fn exact_bench_root_retains_left_limit_geometry() -> Result {
    let mut profile = bench();
    profile.maximum_stage_m = 2.0;
    let batch = Batch::new(crate::runtime::client(), vec![profile])?;
    let rows = batch
        .recover_stages(&[2.0], &CrossSectionRecoveryConfig::default())?
        .properties()?;
    assert_eq!(rows[0].geometry.query_stage_m.to_bits(), 1.0_f64.to_bits());
    assert_eq!(rows[0].geometry.top_width_m, Some(2.0));
    assert_eq!(rows[0].geometry.wetted_perimeter_m, 4.0);
    for h in [
        f64::from_bits(1.0_f64.to_bits() - 1),
        1.0,
        f64::from_bits(1.0_f64.to_bits() + 1),
    ] {
        let direct = batch.evaluate(&[h])?.properties()?;
        assert_wet(&direct[0], bench_expected(h))?;
    }
    Ok(())
}

#[test]
fn invalid_targets_and_config_reject_without_mutating_retained_results() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let definitions = batch.definitions().to_vec();
    let query = batch.evaluate(&[1.0])?;
    let original = query.properties()?;
    let recovery = batch.recover_stages(&[2.0], &CrossSectionRecoveryConfig::default())?;
    let recovered = recovery.properties()?;
    for areas in [
        vec![],
        vec![1.0, 2.0],
        vec![-1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::NEG_INFINITY],
    ] {
        let error = batch
            .recover_stages(&areas, &CrossSectionRecoveryConfig::default())
            .err()
            .ok_or("Invalid target was accepted")?;
        assert!(matches!(error, CrossSectionError::InvalidInput { .. }));
    }
    let mut configs = Vec::new();
    for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        configs.push(CrossSectionRecoveryConfig {
            height_atol_m: value,
            ..Default::default()
        });
        configs.push(CrossSectionRecoveryConfig {
            height_rtol: value,
            ..Default::default()
        });
        configs.push(CrossSectionRecoveryConfig {
            area_atol_m2: value,
            ..Default::default()
        });
        configs.push(CrossSectionRecoveryConfig {
            area_rtol: value,
            ..Default::default()
        });
    }
    configs.push(CrossSectionRecoveryConfig {
        height_atol_m: 0.0,
        height_rtol: 0.0,
        ..Default::default()
    });
    configs.push(CrossSectionRecoveryConfig {
        area_atol_m2: 0.0,
        area_rtol: 0.0,
        ..Default::default()
    });
    configs.push(CrossSectionRecoveryConfig {
        max_iter: 0,
        ..Default::default()
    });
    for config in configs {
        let error = batch
            .recover_stages(&[2.0], &config)
            .err()
            .ok_or("Invalid recovery config was accepted")?;
        assert!(matches!(error, CrossSectionError::InvalidInput { .. }));
    }
    assert_eq!(batch.definitions(), definitions.as_slice());
    assert_same(&query.properties()?[0], &original[0]);
    assert_same(&batch.evaluate(&[1.0])?.properties()?[0], &original[0]);
    assert_same(&recovery.properties()?[0].geometry, &recovered[0].geometry);
    Ok(())
}

#[test]
fn loose_height_convergence_does_not_bypass_area_acceptance() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let config = CrossSectionRecoveryConfig {
        height_atol_m: 10.0,
        height_rtol: 0.0,
        area_atol_m2: 0.0,
        area_rtol: 1e-15,
        ..Default::default()
    };
    let error = batch
        .recover_stages(&[0.6], &config)
        .err()
        .ok_or("Inadequate area residual was accepted")?;
    assert!(matches!(error, CrossSectionError::Numerical { .. }));
    Ok(())
}

#[test]
fn positive_targets_remain_wet_under_generous_area_tolerance() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let config = CrossSectionRecoveryConfig {
        height_atol_m: 1e-24,
        area_atol_m2: 1.0,
        ..Default::default()
    };
    let rows = batch.recover_stages(&[2e-12], &config)?.properties()?;
    assert!(!rows[0].geometry.dry);
    assert!(rows[0].geometry.query_stage_m > 0.0);
    assert!(rows[0].geometry.area_m2 > 0.0);
    assert!(rows[0].geometry.hydrostatic_integral_m3 > 0.0);
    let allowance = config.area_atol_m2 + config.area_rtol * rows[0].requested_area_m2.abs();
    assert!((rows[0].geometry.area_m2 - rows[0].requested_area_m2).abs() <= allowance);
    Ok(())
}

#[test]
fn positive_targets_exclude_dry_endpoint_with_smaller_area_residual() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let config = CrossSectionRecoveryConfig {
        height_atol_m: 10.0,
        height_rtol: 0.0,
        area_atol_m2: 10.0,
        area_rtol: 0.0,
        ..Default::default()
    };
    let target = 2e-12;
    let rows = batch.recover_stages(&[target], &config)?.properties()?;
    assert_eq!(rows[0].requested_area_m2.to_bits(), target.to_bits());
    assert!(!rows[0].geometry.dry);
    assert!(rows[0].geometry.query_stage_m > 0.0);
    assert!(rows[0].geometry.area_m2 > 0.0);
    assert!(rows[0].geometry.hydrostatic_integral_m3 > 0.0);
    let residual = (rows[0].geometry.area_m2 - target).abs();
    assert!(residual <= config.area_atol_m2);
    assert!(target < residual);
    Ok(())
}

#[test]
fn exact_dyadic_datums_preserve_local_recovered_geometry() -> Result {
    let heights: [f64; 4] = [2.0, 1.5, 1.5, 1.25];
    let areas = [4.0, 5.25, 2.25, 3.5];
    let base = Batch::new(crate::runtime::client(), all())?
        .recover_stages(&areas, &CrossSectionRecoveryConfig::default())?
        .properties()?;
    for datum in [-1024.0, 1024.0] {
        let profiles = all()
            .into_iter()
            .map(|mut profile| {
                for point in &mut profile.points {
                    point.elevation_m += datum;
                }
                profile.maximum_stage_m += datum;
                profile
            })
            .collect();
        let batch = Batch::new(crate::runtime::client(), profiles)?;
        let rows = batch
            .recover_stages(&areas, &CrossSectionRecoveryConfig::default())?
            .properties()?;
        for (index, row) in rows.iter().enumerate() {
            assert_eq!(
                row.geometry.query_stage_m.to_bits(),
                (datum + heights[index]).to_bits()
            );
            assert_geometry_bits(&row.geometry, &base[index].geometry);
        }
    }
    Ok(())
}

#[test]
fn unrepresentable_positive_increments_and_pressure_integrals_reject() -> Result {
    let datum = 2.0_f64.powi(54);
    let mut profile = rectangle();
    for point in &mut profile.points {
        point.elevation_m += datum;
    }
    profile.maximum_stage_m += datum;
    let batch = Batch::new(crate::runtime::client(), vec![profile])?;
    let error = batch
        .recover_stages(&[2.0], &CrossSectionRecoveryConfig::default())
        .err()
        .ok_or("Unrepresentable positive increment was accepted")?;
    assert!(matches!(error, CrossSectionError::Numerical { .. }));
    let ordinary = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let config = CrossSectionRecoveryConfig {
        height_atol_m: 1e-220,
        area_atol_m2: 0.0,
        max_iter: 1024,
        ..Default::default()
    };
    let error = ordinary
        .recover_stages(&[2e-200], &config)
        .err()
        .ok_or("Unresolved positive pressure integral was accepted")?;
    assert!(matches!(
        error,
        CrossSectionError::Numerical {
            section: Some(0),
            ..
        }
    ));
    Ok(())
}

#[test]
fn exhausted_iteration_budget_rejects_without_changing_earlier_queries() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![triangle()])?;
    let query = batch.evaluate(&[1.0])?;
    let original = query.properties()?;
    let config = CrossSectionRecoveryConfig {
        max_iter: 1,
        ..Default::default()
    };
    let error = batch
        .recover_stages(&[0.7], &config)
        .err()
        .ok_or("Exhausted recovery was accepted")?;
    assert!(matches!(error, CrossSectionError::Numerical { .. }));
    assert_same(&query.properties()?[0], &original[0]);
    assert_same(&batch.evaluate(&[1.0])?.properties()?[0], &original[0]);
    Ok(())
}
