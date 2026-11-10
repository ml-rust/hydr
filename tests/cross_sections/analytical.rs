use super::{Batch, Result, profiles::*};

#[test]
fn rectangular_vertical_walls_match_area_perimeter_and_hydrostatic_force() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    for h in [-1.0, 0.0, 0.125, 1.0, 4.0] {
        let output = batch.evaluate(&[h])?.properties()?;
        assert_eq!(output[0].section_id, "rectangle");
        assert_eq!(output[0].query_stage_m.to_bits(), h.to_bits());
        if h <= 0.0 {
            assert_dry(&output[0], h);
        } else {
            assert_wet(&output[0], rectangle_expected(h))?;
        }
    }
    Ok(())
}

#[test]
fn sloping_trapezoid_and_triangle_match_closed_form_geometry() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![trapezoid(), triangle()])?;
    for h in [0.5, 1.0, 2.0] {
        let outputs = batch.evaluate(&[h, h])?.properties()?;
        assert_wet(&outputs[0], trapezoid_expected(h))?;
        assert_wet(&outputs[1], triangle_expected(h))?;
    }
    Ok(())
}

#[test]
fn exact_horizontal_bench_stage_uses_left_limit_width_and_perimeter() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![bench()])?;
    for h in [1.0, 1.5] {
        let output = batch.evaluate(&[h])?.properties()?;
        assert_wet(&output[0], bench_expected(h))?;
    }
    let exact = batch.evaluate(&[1.0])?.properties()?;
    assert_eq!(exact[0].top_width_m, Some(2.0));
    assert_eq!(exact[0].wetted_perimeter_m, 4.0);
    let above = batch
        .evaluate(&[f64::from_bits(1.0_f64.to_bits() + 1)])?
        .properties()?;
    assert_eq!(above[0].top_width_m, Some(6.0));
    near(above[0].wetted_perimeter_m, 8.0);
    Ok(())
}

#[test]
fn separated_wet_portions_report_total_geometry_without_a_flow_connection_claim() -> Result {
    let profile = definition(
        "two-troughs",
        &[
            (-3.0, 2.0),
            (-2.0, 0.0),
            (-1.0, 2.0),
            (1.0, 2.0),
            (2.0, 0.0),
            (3.0, 2.0),
        ],
        2.0,
    );
    let batch = Batch::new(crate::runtime::client(), vec![profile])?;
    let output = batch.evaluate(&[1.0])?.properties()?;
    assert_wet(
        &output[0],
        Expected {
            area: 1.0,
            width: 2.0,
            perimeter: 2.0 * 5.0_f64.sqrt(),
            integral: 1.0 / 3.0,
        },
    )?;
    Ok(())
}
