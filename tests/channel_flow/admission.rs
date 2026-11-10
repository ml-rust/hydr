use super::common::{TestResult, cell, definition, exterior, reach, walls};
use hydr::channel::{ChannelBoundaryInterval, ChannelInitialState, ChannelReach};

#[test]
fn rectangular_definitions_reject_unresolved_or_incomplete_geometry() -> TestResult {
    let valid = definition(vec![cell(0, 1.0, 2.0, 0.0)], 1.0);
    let mut cases = Vec::new();
    let mut candidate = valid.clone();
    candidate.cells.clear();
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.vertical_datum.clear();
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.provenance.clear();
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.cells.push(candidate.cells[0].clone());
    cases.push(candidate);
    for width in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut candidate = valid.clone();
        candidate.cells[0].width_m = width;
        cases.push(candidate);
    }
    for length in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut candidate = valid.clone();
        candidate.cells[0].length_m = length;
        cases.push(candidate);
    }
    let mut candidate = valid.clone();
    candidate.cells[0].maximum_depth_m = 0.0;
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.cells[0].manning_n = -0.01;
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.cells[0].bed_elevation_m = f64::INFINITY;
    cases.push(candidate);
    let mut candidate = valid.clone();
    candidate.cells[0].width_m = f64::MIN_POSITIVE;
    candidate.cells[0].length_m = f64::MIN_POSITIVE;
    cases.push(candidate);
    for candidate in cases {
        assert!(
            ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                candidate,
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: vec![0.0],
                    longitudinal_momentum_m4_per_s: vec![0.0],
                },
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn initial_state_rejects_negative_water_nonfinite_values_and_dry_momentum() -> TestResult {
    let definition = definition(vec![cell(0, 1.0, 2.0, 0.0)], 1.0);
    for (time, volume, momentum) in [
        (f64::NAN, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, f64::NAN, 0.0),
        (0.0, f64::INFINITY, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, f64::INFINITY),
        (0.0, 21.0, 0.0),
    ] {
        assert!(
            ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                definition.clone(),
                ChannelInitialState {
                    time_s: time,
                    volume_m3: vec![volume],
                    longitudinal_momentum_m4_per_s: vec![momentum],
                },
            )
            .is_err()
        );
    }
    Ok(())
}

#[test]
fn rejected_boundary_intervals_preserve_complete_accepted_state() -> TestResult {
    let initial = reach(
        definition(vec![cell(0, 1.0, 2.0, 0.0)], 1.0),
        &[1.0],
        &[0.0],
    )?;
    let accepted = initial
        .step(
            0.01,
            &ChannelBoundaryInterval {
                start_s: 0.0,
                end_s: 0.01,
                left: walls()[0].clone(),
                right: walls()[1].clone(),
            },
        )?
        .next;
    let before = accepted.snapshot_data()?;
    let valid = ChannelBoundaryInterval {
        start_s: accepted.time(),
        end_s: 1.0,
        left: walls()[0].clone(),
        right: walls()[1].clone(),
    };
    let mut cases = Vec::new();
    let mut forcing = valid.clone();
    forcing.start_s = 0.0;
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.end_s = 0.05;
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.left = exterior(-1.0, 0.0);
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.right = exterior(0.0, 1.0);
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.left = exterior(f64::NAN, 0.0);
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.right = exterior(1.0, f64::INFINITY);
    cases.push((0.1, forcing));
    let mut forcing = valid.clone();
    forcing.left = exterior(11.0, 0.0);
    cases.push((0.1, forcing));
    cases.push((accepted.time(), valid.clone()));
    cases.push((f64::NAN, valid));
    for (target, forcing) in cases {
        assert!(accepted.step(target, &forcing).is_err());
        assert_eq!(accepted.snapshot_data()?, before);
    }
    Ok(())
}
