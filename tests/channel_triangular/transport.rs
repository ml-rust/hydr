use super::common::{
    GRAVITY, TestResult, advance, cell, definition, exterior, forcing, near, reach, walls,
};
use hydr::channel::ChannelGeometry;

#[test]
fn uniform_motion_preserves_depth_and_discharge_in_both_directions() -> TestResult {
    for velocity in [-1.0, 1.0] {
        let initial = reach(
            definition((0..20).map(|i| cell(i, 5.0)).collect(), 1.0),
            &[1.0; 20],
            &[velocity; 20],
        )?;
        let result = advance(
            initial,
            5.0,
            [exterior(1.0, velocity), exterior(1.0, velocity)],
        )?;
        for cell in result.state()? {
            near(cell.depth_m, 1.0, 1e-12);
            near(cell.discharge_m3_per_s, velocity, 1e-11);
        }
        near(result.ledger().boundary_in_m3, 5.0, 1e-11);
        near(result.ledger().boundary_out_m3, 5.0, 1e-11);
    }
    Ok(())
}

#[test]
fn closed_rest_preserves_unequal_length_cells_and_zero_face_water() -> TestResult {
    let lengths = [0.5, 2.0, 7.0, 1.25, 3.0];
    let initial = reach(
        definition(
            lengths
                .iter()
                .enumerate()
                .map(|(i, &dx)| cell(i, dx))
                .collect(),
            1.0,
        ),
        &[1.0; 5],
        &[0.0; 5],
    )?;
    let mut current = initial;
    while current.time() < 2.0 {
        current = current
            .step(2.0, &forcing(current.time(), 2.0, walls()))?
            .next;
        super::common::check_water(&current)?;
        for state in current.state()? {
            near(state.depth_m, 1.0, 1e-12);
            near(state.discharge_m3_per_s, 0.0, 1e-11);
        }
        let receipt = current.last_step()?.ok_or("Missing rest receipt")?;
        assert!(receipt.face_volume_m3.iter().all(|v| v.abs() <= 1e-11));
    }
    near(current.ledger().storage_m3, lengths.iter().sum(), 1e-11);
    Ok(())
}

#[test]
fn asymmetric_banks_share_inviscid_law_and_have_distinct_exact_friction() -> TestResult {
    let mut decayed = Vec::new();
    for (left, right) in [(1.0_f64, 1.0_f64), (0.0, 2.0)] {
        for velocity in [-1.0, 1.0] {
            for roughness in [0.0, 0.03] {
                let mut triangle = cell(0, 100.0);
                triangle.manning_n = roughness;
                let mut layout = definition(vec![triangle], 0.1);
                let ChannelGeometry::PrismaticTriangular {
                    left_bank_slope_h_per_v,
                    right_bank_slope_h_per_v,
                    ..
                } = &mut layout.geometry
                else {
                    return Err("Expected triangle".into());
                };
                *left_bank_slope_h_per_v = left;
                *right_bank_slope_h_per_v = right;
                let initial = reach(layout, &[1.0], &[velocity])?;
                let result = initial
                    .step(
                        0.1,
                        &forcing(0.0, 0.1, [exterior(1.0, velocity), exterior(1.0, velocity)]),
                    )?
                    .next;
                let perimeter = (1.0 + left * left).sqrt() + (1.0 + right * right).sqrt();
                let radius = 1.0 / perimeter;
                let expected = velocity
                    / (1.0
                        + GRAVITY * roughness * roughness * 0.1 * velocity.abs()
                            / radius.powf(4.0 / 3.0));
                let state = result.state()?;
                near(state[0].discharge_m3_per_s, expected, 1e-12);
                assert_eq!(state[0].volume_m3.to_bits(), 100.0_f64.to_bits());
                super::common::check_water(&result)?;
                if velocity > 0.0 && roughness > 0.0 {
                    decayed.push(state[0].discharge_m3_per_s);
                }
            }
        }
    }
    assert_eq!(decayed.len(), 2);
    assert!((decayed[0] - decayed[1]).abs() > 1e-6);
    Ok(())
}

#[test]
fn equal_area_asymmetric_banks_reproduce_nonuniform_inviscid_transport() -> TestResult {
    let mut results = Vec::new();
    for (left, right) in [(1.0, 1.0), (0.0, 2.0)] {
        let mut layout = definition((0..4).map(|i| cell(i, 2.0)).collect(), 0.005);
        if let ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v,
            right_bank_slope_h_per_v,
            ..
        } = &mut layout.geometry
        {
            *left_bank_slope_h_per_v = left;
            *right_bank_slope_h_per_v = right;
        }
        let result = advance(
            reach(layout, &[1.0, 0.5, 0.1, 0.0], &[0.2, -0.1, 0.0, 0.0])?,
            0.05,
            walls(),
        )?;
        results.push((
            result.state()?,
            result.ledger().clone(),
            result.last_step()?,
        ));
    }
    assert_eq!(results[0], results[1]);
    Ok(())
}
