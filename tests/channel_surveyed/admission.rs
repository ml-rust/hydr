use super::sources::{definition, initial};
use super::support::{TestResult, forcing, walls};
use hydr::channel::{ChannelBoundary, ChannelGeometry, ChannelReach};

#[test]
fn malformed_original_profiles_cells_and_datums_reject_public_construction() -> TestResult {
    for mutation in 0..13 {
        let mut layout = definition(1, 0.03, false);
        if let ChannelGeometry::PrismaticSurveyed { profile, cells, .. } = &mut layout.geometry {
            match mutation {
                0 => profile.points.clear(),
                1 => profile.points[1].station_m = profile.points[0].station_m.next_down(),
                2 => profile.points[1].elevation_m = f64::NAN,
                3 => profile.maximum_stage_m = 0.0,
                4 => profile.maximum_stage_m = 5.0,
                5 => profile.vertical_datum.push_str(" other"),
                6 => profile.profile_provenance.clear(),
                7 => cells[0].length_m = 0.0,
                8 => cells[0].length_m = f64::INFINITY,
                9 => cells[0].manning_n = -0.03,
                10 => cells[0].manning_n = f64::NAN,
                11 => cells.push(cells[0].clone()),
                _ => cells[0].id.clear(),
            }
        }
        let count = if mutation == 11 { 2 } else { 1 };
        assert!(
            ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                layout,
                initial(count, false),
            )
            .is_err(),
            "Accepted malformed geometry mutation {mutation}"
        );
    }
    Ok(())
}

#[test]
fn equal_stations_admit_an_original_vertical_bank() -> TestResult {
    let mut layout = definition(1, 0.03, false);
    if let ChannelGeometry::PrismaticSurveyed { profile, .. } = &mut layout.geometry {
        profile.points[1].station_m = profile.points[0].station_m;
    }
    let reach = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        layout,
        initial(1, false),
    )?;
    let state = reach.state()?;
    assert_eq!(state[0].volume_m3, 12.0);
    assert!(!state[0].dry);
    Ok(())
}

#[test]
fn invalid_numerical_controls_and_unsupported_initial_states_reject() -> TestResult {
    for mutation in 0..10 {
        let mut layout = definition(1, 0.03, false);
        if let ChannelGeometry::PrismaticSurveyed { numerical, .. } = &mut layout.geometry {
            match mutation {
                0 => numerical.recovery.depth_atol_m = f64::NAN,
                1 => numerical.normalized_root.rtol = -1.0,
                2 => numerical.initial_simpson_panels = 3,
                3 => numerical.max_simpson_panels = 2,
                4 => numerical.normalized_root.max_points = 0,
                5 => numerical.max_native_query_points = 0,
                6 => numerical.friction_geometry_atol_m4_per_s = f64::INFINITY,
                7 => numerical.friction_geometry_rtol = -1.0,
                8 => layout.config.gravity = 0.0,
                _ => layout.config.riemann_flux_relative = f64::NAN,
            }
        }
        assert!(
            ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                layout,
                initial(1, false),
            )
            .is_err()
        );
    }
    for (volume, momentum) in [
        (-1.0, 0.0),
        (f64::NAN, 0.0),
        (97.0, 0.0),
        (0.0, 1.0),
        (12.0, f64::INFINITY),
    ] {
        let mut state = initial(1, false);
        state.volume_m3[0] = volume;
        state.longitudinal_momentum_m4_per_s[0] = momentum;
        assert!(
            ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                definition(1, 0.03, false),
                state,
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn bracket_expansion_control_retains_snapshot_identity() -> TestResult {
    let layout = definition(1, 0.03, false);
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        layout.clone(),
        initial(1, false),
    )?;
    let mut endpoint_layout = layout;
    if let ChannelGeometry::PrismaticSurveyed { numerical, .. } = &mut endpoint_layout.geometry {
        numerical.max_bracket_expansions = 0;
    }
    let endpoint_only = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        endpoint_layout,
        initial(1, false),
    )?;
    assert!(origin.restore(&endpoint_only.snapshot()).is_err());
    assert!(
        origin
            .import_snapshot(&endpoint_only.snapshot_data()?)
            .is_err()
    );
    assert!(
        endpoint_only
            .import_snapshot(&origin.snapshot_data()?)
            .is_err()
    );
    Ok(())
}

#[test]
#[cfg(target_pointer_width = "64")]
fn bracket_expansion_ceiling_rejects_counts_above_i64() {
    let mut layout = definition(1, 0.03, false);
    if let ChannelGeometry::PrismaticSurveyed { numerical, .. } = &mut layout.geometry {
        numerical.max_bracket_expansions = usize::MAX;
    }
    assert!(
        ChannelReach::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            layout,
            initial(1, false),
        )
        .is_err()
    );
}

#[test]
fn native_cap_volume_rejects_overflow_with_finite_original_operands() -> TestResult {
    let mut layout = definition(1, 0.03, false);
    if let ChannelGeometry::PrismaticSurveyed { cells, .. } = &mut layout.geometry {
        cells[0].length_m = f64::MAX / 16.0;
        assert!(cells[0].length_m.is_finite());
    }
    let mut state = initial(1, false);
    state.volume_m3[0] = 0.0;
    assert!(
        ChannelReach::<crate::runtime::Runtime>::new(crate::runtime::client(), layout, state,)
            .is_err()
    );
    Ok(())
}

#[test]
fn characteristic_policies_and_invalid_complete_exterior_states_preserve_accepted_state()
-> TestResult {
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        definition(1, 0.03, false),
        initial(1, false),
    )?;
    let before = origin.snapshot_data()?;
    // Certified surveyed traces admit the characteristic policies. A step leaves the
    // accepted origin unchanged.
    let characteristic = [
        ChannelBoundary::Stage { elevation_m: 1.0 },
        ChannelBoundary::Discharge {
            discharge_m3_per_s: 0.0,
        },
        ChannelBoundary::QuiescentReservoir {
            elevation_m: 1.0,
            entrance_loss_coefficient: 0.0,
        },
    ];
    for policy in characteristic {
        for left in [false, true] {
            let mut boundaries = walls();
            boundaries[usize::from(!left)] = policy.clone();
            origin.step(0.001, &forcing(0.0, 0.001, boundaries))?;
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    let policies = [
        ChannelBoundary::Exterior {
            depth_m: -1.0,
            velocity_m_per_s: 0.0,
        },
        ChannelBoundary::Exterior {
            depth_m: 5.0,
            velocity_m_per_s: 0.0,
        },
        ChannelBoundary::Exterior {
            depth_m: 0.0,
            velocity_m_per_s: 1.0,
        },
        ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: f64::INFINITY,
        },
    ];
    for policy in policies {
        for left in [false, true] {
            let mut boundaries = walls();
            boundaries[usize::from(!left)] = policy.clone();
            assert!(
                origin
                    .step(0.001, &forcing(0.0, 0.001, boundaries))
                    .is_err()
            );
            assert_eq!(origin.snapshot_data()?, before);
        }
    }
    Ok(())
}
