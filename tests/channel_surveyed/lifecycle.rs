use super::sources::{definition, initial, interval};
use super::support::{TestResult, check_water, forcing, near, profile, reach, walls};
use hydr::channel::{ChannelConfig, ChannelGeometry, ChannelReach, ChannelSurveyedConfig};
use hydr::forcing::WaterAmount;

#[test]
fn uniform_rest_at_original_survey_knots_preserves_storage() -> TestResult {
    let points = [
        [-6.0_f64, 4.0_f64],
        [-4.0, 2.0],
        [-1.0, 0.0],
        [1.0, 0.0],
        [4.0, 2.0],
        [6.0, 4.0],
    ];
    for (depth, area) in [(0.0, 0.0), (2.0, 10.0), (4.0, 30.0)] {
        let section = profile(
            "integer-knots",
            &points.map(|p| p.map(f64::to_bits)),
            4.0_f64.to_bits(),
        );
        let origin = reach::<crate::runtime::Runtime>(
            crate::runtime::client(),
            section,
            &[2.0, 4.0, 8.0],
            &[0.03; 3],
            &[area; 3],
            &[0.0; 3],
            ChannelConfig::default(),
            ChannelSurveyedConfig::default(),
        )?;
        let before = origin.snapshot_data()?;
        let next = origin.step(0.001, &forcing(0.0, 0.001, walls()))?.next;
        for (cell, length) in next.state()?.iter().zip([2.0, 4.0, 8.0]) {
            near(cell.volume_m3, area * length, 1e-11);
            near(cell.depth_m, depth, 1e-11);
            near(cell.longitudinal_momentum_m4_per_s, 0.0, 1e-10);
        }
        assert_eq!(origin.snapshot_data()?, before);
        check_water(&next)?;
    }
    Ok(())
}

#[test]
fn ordinary_manning_advancement_replays_complete_sourced_packages() -> TestResult {
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        definition(3, 0.03, true),
        initial(3, true),
    )?;
    let before = origin.snapshot_data()?;
    let held = interval(
        0.0,
        0.001,
        WaterAmount::IntervalVolume { volume_m3: 0.0002 },
        WaterAmount::Rate { rate_m3_per_s: 0.1 },
    );
    let current = origin.step(0.001, &held)?.next;
    let receipt = current.last_step()?.ok_or("Missing moving receipt")?;
    assert!(receipt.friction_impulse_m4_per_s.iter().all(|v| *v < 0.0));
    let geometry = receipt
        .friction_geometry
        .as_ref()
        .ok_or("Missing friction geometry bounds")?;
    assert_eq!(geometry.momentum_abs_bound_m4_per_s.len(), 3);
    assert!(
        geometry
            .momentum_abs_bound_m4_per_s
            .iter()
            .all(|v| v.is_finite() && *v >= 0.0)
    );
    let riemann = receipt
        .riemann
        .as_ref()
        .ok_or("Missing surveyed flux receipt")?;
    let work = riemann
        .surveyed_work
        .as_ref()
        .ok_or("Missing surveyed work")?;
    assert_eq!(riemann.root_iterations, work.maximum_root_iterations);
    let saved = current.snapshot_data()?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&saved)?;
    assert_eq!(trusted.snapshot_data()?, saved);
    assert_eq!(cold.snapshot_data()?, saved);
    let held = interval(
        0.001,
        0.002,
        WaterAmount::IntervalVolume { volume_m3: 0.0003 },
        WaterAmount::Rate { rate_m3_per_s: 0.1 },
    );
    let expected = current.step(0.002, &held)?.next.snapshot_data()?;
    for restored in [trusted, cold] {
        assert_eq!(restored.step(0.002, &held)?.next.snapshot_data()?, expected);
    }
    assert_eq!(origin.snapshot_data()?, before);
    assert_eq!(current.snapshot_data()?, saved);
    check_water(&current)?;
    for mutation in 0..5 {
        let mut altered = saved.clone();
        let diagnostics = &mut altered
            .last_interval
            .as_mut()
            .ok_or("Missing stored interval")?
            .diagnostics;
        match mutation {
            0 => diagnostics.friction_geometry = None,
            1 => {
                let geometry = diagnostics
                    .friction_geometry
                    .as_mut()
                    .ok_or("Missing stored geometry")?;
                let value = &mut geometry.momentum_abs_bound_m4_per_s[0];
                *value = f64::from_bits(value.to_bits() + 1);
            }
            2 => {
                diagnostics
                    .riemann
                    .as_mut()
                    .ok_or("Missing stored flux")?
                    .surveyed_work = None
            }
            3 => {
                diagnostics
                    .riemann
                    .as_mut()
                    .ok_or("Missing stored flux")?
                    .surveyed_work
                    .as_mut()
                    .ok_or("Missing stored work")?
                    .maximum_root_iterations += 1
            }
            _ => {
                let geometry = diagnostics
                    .friction_geometry
                    .as_mut()
                    .ok_or("Missing stored geometry")?;
                geometry.max_momentum_abs_bound_m4_per_s =
                    f64::from_bits(geometry.max_momentum_abs_bound_m4_per_s.to_bits() + 1);
            }
        }
        assert!(origin.import_snapshot(&altered).is_err());
        assert_eq!(origin.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn original_profile_control_and_signed_zero_bits_bind_both_restore_paths() -> TestResult {
    let layout = definition(1, 0.03, false);
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        layout.clone(),
        initial(1, true),
    )?;
    let current = origin.step(0.001, &forcing(0.0, 0.001, walls()))?.next;
    let saved = current.snapshot_data()?;
    for mutation in 0..8 {
        let mut other = layout.clone();
        if let ChannelGeometry::PrismaticSurveyed {
            profile,
            numerical,
            cells,
        } = &mut other.geometry
        {
            match mutation {
                0 => profile.points[1].elevation_m = -0.0,
                1 => profile.points[0].station_m = f64::from_bits((-5.0_f64).to_bits() + 1),
                2 => profile.profile_provenance.push(' '),
                3 => {
                    numerical.normalized_root.atol =
                        f64::from_bits(numerical.normalized_root.atol.to_bits() + 1)
                }
                4 => numerical.max_native_query_points += 1,
                5 => {
                    numerical.friction_geometry_rtol =
                        f64::from_bits(numerical.friction_geometry_rtol.to_bits() + 1)
                }
                6 => cells[0].manning_n = f64::from_bits(0.03_f64.to_bits() + 1),
                _ => numerical.recovery.depth_rtol = -0.0,
            }
        }
        let destination = ChannelReach::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            other,
            initial(1, true),
        )?;
        let before = destination.snapshot_data()?;
        assert!(destination.restore(&current.snapshot()).is_err());
        assert!(destination.import_snapshot(&saved).is_err());
        assert_eq!(destination.snapshot_data()?, before);
    }
    Ok(())
}
