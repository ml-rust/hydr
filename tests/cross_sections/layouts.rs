use super::{Batch, Result, profiles::*};

#[test]
fn mixed_profile_lengths_and_permutations_preserve_section_identity_and_values() -> Result {
    let definitions = all();
    let stages = [1.0, 0.5, 1.0, 1.5];
    let batch = Batch::new(crate::runtime::client(), definitions.clone())?;
    let first_query = batch.evaluate(&stages)?;
    let original = first_query.properties()?;
    let expected = [
        rectangle_expected(stages[0]),
        trapezoid_expected(stages[1]),
        triangle_expected(stages[2]),
        bench_expected(stages[3]),
    ];
    for (actual, expected) in original.iter().zip(expected) {
        assert_wet(actual, expected)?;
    }
    let order = [3, 1, 0, 2];
    let permuted = Batch::new(
        crate::runtime::client(),
        order.iter().map(|&i| definitions[i].clone()).collect(),
    )?;
    let values = permuted.evaluate(&order.map(|i| stages[i]))?.properties()?;
    for (actual, index) in values.iter().zip(order) {
        assert_same(actual, &original[index]);
    }
    let _other = batch.evaluate(&[2.0, 1.0, 0.5, 1.0])?.properties()?;
    let repeated = batch.evaluate(&stages)?.properties()?;
    let retained = first_query.properties()?;
    for ((a, b), c) in original.iter().zip(&repeated).zip(&retained) {
        assert_same(a, b);
        assert_same(a, c);
    }
    Ok(())
}

#[test]
fn exact_dyadic_station_and_datum_translations_preserve_geometry_bits() -> Result {
    let definitions = all();
    let stages = [0.125, 1.0, 0.5, 1.5];
    let original = Batch::new(crate::runtime::client(), definitions.clone())?
        .evaluate(&stages)?
        .properties()?;
    let translated = definitions
        .into_iter()
        .map(|mut profile| {
            for point in &mut profile.points {
                point.station_m += 32.0;
                point.elevation_m -= 64.0;
            }
            profile.maximum_stage_m -= 64.0;
            profile.vertical_datum = "Translated analytical datum".into();
            profile
        })
        .collect();
    let batch = Batch::new(crate::runtime::client(), translated)?;
    let actual = batch.evaluate(&stages.map(|h| h - 64.0))?.properties()?;
    for ((a, b), h) in actual.iter().zip(&original).zip(stages) {
        assert_geometry_bits(a, b);
        assert_eq!(a.query_stage_m.to_bits(), (h - 64.0).to_bits());
    }
    Ok(())
}

#[test]
fn geometric_scaling_preserves_normalized_properties_across_three_exponents() -> Result {
    let stages = [1.0, 1.0, 1.0, 1.5];
    let expected = [
        rectangle_expected(1.0),
        trapezoid_expected(1.0),
        triangle_expected(1.0),
        bench_expected(1.5),
    ];
    for exponent in [-100, 0, 100] {
        let scale = 2.0_f64.powi(exponent);
        let definitions = all()
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
        let batch = Batch::new(crate::runtime::client(), definitions)?;
        let outputs = batch.evaluate(&stages.map(|h| h * scale))?.properties()?;
        for (output, expected) in outputs.iter().zip(expected) {
            assert_scaled_wet(output, expected, scale)?;
        }
    }
    Ok(())
}

#[test]
fn collinear_subdivision_preserves_geometry_within_frozen_dimensional_gates() -> Result {
    let stages = [0.125, 1.0, 0.5, 1.5];
    let expected = [
        rectangle_expected(stages[0]),
        trapezoid_expected(stages[1]),
        triangle_expected(stages[2]),
        bench_expected(stages[3]),
    ];
    let subdivided = all()
        .into_iter()
        .map(|mut profile| {
            let mut points = Vec::with_capacity(profile.points.len() * 2 - 1);
            for pair in profile.points.windows(2) {
                points.push(pair[0]);
                points.push(hydr::channel::CrossSectionPoint {
                    station_m: (pair[0].station_m + pair[1].station_m) * 0.5,
                    elevation_m: (pair[0].elevation_m + pair[1].elevation_m) * 0.5,
                });
            }
            points.extend(profile.points.last().cloned());
            profile.points = points;
            profile
        })
        .collect();
    let outputs = Batch::new(crate::runtime::client(), subdivided)?
        .evaluate(&stages)?
        .properties()?;
    for (output, expected) in outputs.iter().zip(expected) {
        assert_wet(output, expected)?;
    }
    Ok(())
}
