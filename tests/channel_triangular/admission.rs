use super::common::{GRAVITY, TestResult, cell, definition, forcing, reach, walls};
use hydr::channel::{
    ChannelGeometry, ChannelInitialState, ChannelReach, ChannelTriangularReconstruction,
    RectangularChannelCell,
};

#[test]
fn invalid_triangle_geometry_rejects_before_allocating_accepted_state() -> TestResult {
    for (left, right, bed) in [
        (-1.0, 1.0, 0.0),
        (0.0, 0.0, 0.0),
        (f64::NAN, 1.0, 0.0),
        (1.0, f64::INFINITY, 0.0),
        (1.0, 1.0, f64::NAN),
    ] {
        let mut layout = definition(vec![cell(0, 1.0)], 1.0);
        if let ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v,
            right_bank_slope_h_per_v,
            bed_elevation_m,
            ..
        } = &mut layout.geometry
        {
            *left_bank_slope_h_per_v = left;
            *right_bank_slope_h_per_v = right;
            *bed_elevation_m = bed;
        }
        assert!(reach(layout, &[1.0], &[0.0]).is_err());
    }
    assert!(reach(definition(Vec::new(), 1.0), &[], &[]).is_err());
    for mutation in 0..6 {
        let mut cells = vec![cell(0, 1.0)];
        match mutation {
            0 => cells[0].length_m = 0.0,
            1 => cells[0].length_m = f64::INFINITY,
            2 => cells[0].maximum_depth_m = 0.0,
            3 => cells[0].manning_n = -0.01,
            4 => cells[0].id.clear(),
            5 => cells.push(cells[0].clone()),
            _ => return Err("Unknown admission mutation".into()),
        }
        let count = cells.len();
        assert!(reach(definition(cells, 1.0), &vec![1.0; count], &vec![0.0; count]).is_err());
    }
    assert!(reach(definition(vec![cell(0, 1.0)], 1.0), &[121.0], &[0.0]).is_err());
    assert!(reach(definition(vec![cell(0, 1.0)], 1.0), &[0.0], &[1.0]).is_err());
    Ok(())
}

#[test]
fn slope_bits_and_geometry_variant_bind_trusted_and_cold_restart() -> TestResult {
    let layout = definition(vec![cell(0, 2.0)], 1.0);
    let initial = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0],
        longitudinal_momentum_m4_per_s: vec![0.0],
    };
    let origin = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        layout.clone(),
        initial.clone(),
    )?;
    let current = origin.step(0.01, &forcing(0.0, 0.01, walls()))?.next;
    for mutation in 0..3 {
        let mut other = layout.clone();
        match &mut other.geometry {
            ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v,
                right_bank_slope_h_per_v,
                bed_elevation_m,
                ..
            } => match mutation {
                0 => *left_bank_slope_h_per_v = f64::from_bits(1.0_f64.to_bits() + 1),
                1 => {
                    *left_bank_slope_h_per_v = 0.0;
                    *right_bank_slope_h_per_v = 2.0;
                }
                2 => *bed_elevation_m = -0.0,
                _ => return Err("Unknown identity mutation".into()),
            },
            _ => return Err("Expected triangle".into()),
        }
        let destination = ChannelReach::<crate::runtime::Runtime>::new(
            crate::runtime::client(),
            other,
            initial.clone(),
        )?;
        let before = destination.snapshot_data()?;
        assert!(destination.restore(&current.snapshot()).is_err());
        assert!(
            destination
                .import_snapshot(&current.snapshot_data()?)
                .is_err()
        );
        assert_eq!(destination.snapshot_data()?, before);
    }
    let mut rectangular = layout;
    rectangular.geometry = ChannelGeometry::Rectangular {
        cells: vec![RectangularChannelCell {
            id: "cell-0".into(),
            length_m: 2.0,
            width_m: 1.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 10.0,
            manning_n: 0.0,
        }],
    };
    let destination = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        rectangular,
        initial,
    )?;
    assert!(destination.restore(&current.snapshot()).is_err());
    assert!(
        destination
            .import_snapshot(&current.snapshot_data()?)
            .is_err()
    );
    assert!(origin.restore(&destination.snapshot()).is_err());
    assert!(
        origin
            .import_snapshot(&destination.snapshot_data()?)
            .is_err()
    );
    Ok(())
}

#[test]
fn first_order_positive_films_retain_finite_dry_front_scale_wave_limits() -> TestResult {
    let left_sound = (GRAVITY / 2.0).sqrt();
    for exponent in 1..=150 {
        let film_depth = 10.0_f64.powi(-exponent);
        let right_sound = (GRAVITY * film_depth / 2.0).sqrt();
        // The deep cell takes `c_L` from its wall and `4 c_L - 3 c_R` from the film face.
        // Its frequency is their sum over twice its unit length.
        let expected_limit = 2.0 * 0.45 / (5.0 * left_sound - 3.0 * right_sound);
        let mut layout = definition(vec![cell(0, 1.0), cell(1, 1.0)], 1.0);
        layout.config.triangular_reconstruction =
            ChannelTriangularReconstruction::FirstOrderReference;
        let initial = reach(layout, &[1.0, film_depth * film_depth], &[0.0; 2])?;
        assert!(!initial.state()?[1].dry);
        let before = initial.snapshot_data()?;
        let next = initial.step(1e-6, &forcing(0.0, 1e-6, walls()))?.next;
        let receipt = next.last_step()?.ok_or("Missing positive-film receipt")?;
        assert!(receipt.cfl_limit_s.is_finite());
        assert!(receipt.cfl_limit_s >= 0.99 * expected_limit);
        assert!(receipt.cfl_limit_s <= 1.01 * expected_limit);
        assert!(next.state()?[1].volume_m3 > 0.0);
        assert!(!next.state()?[1].dry);
        assert_eq!(initial.snapshot_data()?, before);
        super::common::check_water(&next)?;
    }
    Ok(())
}

#[test]
fn unsupported_exterior_states_reject_without_mutating_accepted_state() -> TestResult {
    let initial = reach(definition(vec![cell(0, 2.0)], 1.0), &[1.0], &[0.0])?;
    let before = initial.snapshot_data()?;
    for (depth, velocity) in [
        (11.0, 0.0),
        (-1.0, 0.0),
        (0.0, 1.0),
        (f64::NAN, 0.0),
        (1.0, f64::INFINITY),
    ] {
        let interval = forcing(
            0.0,
            0.01,
            [
                super::common::exterior(depth, velocity),
                hydr::channel::ChannelBoundary::Wall,
            ],
        );
        assert!(initial.step(0.01, &interval).is_err());
        assert_eq!(initial.snapshot_data()?, before);
    }
    Ok(())
}

#[test]
fn invalid_riemann_flux_budgets_reject_public_construction() -> TestResult {
    for invalid in [-1.0, f64::INFINITY, f64::NAN] {
        for field in 0..3 {
            let mut layout = definition(vec![cell(0, 2.0)], 1.0);
            match field {
                0 => layout.config.riemann_water_flux_absolute_m3_per_s = invalid,
                1 => layout.config.riemann_momentum_flux_absolute_m4_per_s2 = invalid,
                _ => layout.config.riemann_flux_relative = invalid,
            }
            assert!(reach(layout, &[1.0], &[0.0]).is_err());
        }
    }
    for water in [true, false] {
        let mut layout = definition(vec![cell(0, 2.0)], 1.0);
        layout.config.riemann_flux_relative = 0.0;
        if water {
            layout.config.riemann_water_flux_absolute_m3_per_s = 0.0;
        } else {
            layout.config.riemann_momentum_flux_absolute_m4_per_s2 = 0.0;
        }
        assert!(reach(layout, &[1.0], &[0.0]).is_err());
    }
    Ok(())
}

#[test]
fn limited_linear_positive_films_advance_with_their_reported_cfl_limits() -> TestResult {
    for exponent in 1..=150 {
        let film_depth = 10.0_f64.powi(-exponent);
        let mut layout = definition(vec![cell(0, 1.0), cell(1, 1.0)], 1.0);
        layout.config.triangular_reconstruction = ChannelTriangularReconstruction::LimitedLinear;
        let initial = reach(layout, &[1.0, film_depth * film_depth], &[0.0; 2])?;
        let before = initial.snapshot_data()?;
        assert!(!initial.state()?[1].dry);
        let next = initial.step(1e-6, &forcing(0.0, 1e-6, walls()))?.next;
        let receipt = next
            .last_step()?
            .ok_or("Missing limited-linear film receipt")?;
        let dt = receipt.end_s - receipt.start_s;
        assert!(receipt.cfl_limit_s.is_finite() && receipt.cfl_limit_s > 0.0);
        assert!(dt <= receipt.cfl_limit_s);
        assert!(dt <= receipt.accepted_start_cfl_limit_s);
        assert!(next.state()?[1].volume_m3 > 0.0);
        assert!(!next.state()?[1].dry);
        assert_eq!(initial.snapshot_data()?, before);
        super::common::check_water(&next)?;
    }
    Ok(())
}
