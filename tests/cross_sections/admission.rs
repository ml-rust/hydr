use super::{Batch, Result, profiles::*};
use hydr::channel::CrossSectionError;

#[test]
fn profiles_require_finite_ordered_coordinates_and_declared_support() -> Result {
    assert!(Batch::new(crate::runtime::client(), vec![]).is_err());
    let mut cases = Vec::new();
    let mut one = rectangle();
    one.points.truncate(1);
    cases.push(("single-point", one));
    let mut duplicate_point = rectangle();
    duplicate_point.points.insert(2, duplicate_point.points[1]);
    cases.push(("duplicate-point", duplicate_point));
    let mut overhang = rectangle();
    overhang.points[2].station_m = -1.0;
    cases.push(("overhang", overhang));
    cases.push((
        "zero-width",
        definition("zero-width", &[(0.0, 2.0), (0.0, 0.0)], 1.0),
    ));
    cases.push((
        "vertical-retracing",
        definition(
            "retracing",
            &[(0.0, 4.0), (0.0, 1.0), (0.0, 2.0), (2.0, 0.0), (2.0, 4.0)],
            3.0,
        ),
    ));
    let mut above_banks = rectangle();
    above_banks.maximum_stage_m = 5.0;
    cases.push(("above-banks", above_banks));
    let mut bed_limit = rectangle();
    bed_limit.maximum_stage_m = 0.0;
    cases.push(("bed-limit", bed_limit));
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut station = rectangle();
        station.points[2].station_m = value;
        cases.push(("nonfinite-station", station));
        let mut elevation = rectangle();
        elevation.points[1].elevation_m = value;
        cases.push(("nonfinite-elevation", elevation));
        let mut cap = rectangle();
        cap.maximum_stage_m = value;
        cases.push(("nonfinite-cap", cap));
    }
    cases.push((
        "extent-overflow",
        definition(
            "extent-overflow",
            &[
                (-f64::MAX, 4.0),
                (-f64::MAX, 0.0),
                (f64::MAX, 0.0),
                (f64::MAX, 4.0),
            ],
            4.0,
        ),
    ));
    cases.push((
        "vertical-difference-overflow",
        definition(
            "vertical-overflow",
            &[
                (0.0, f64::MAX),
                (0.0, -f64::MAX),
                (2.0, -f64::MAX),
                (2.0, f64::MAX),
            ],
            0.0,
        ),
    ));
    for (name, profile) in cases {
        assert!(
            Batch::new(crate::runtime::client(), vec![profile]).is_err(),
            "Accepted {name}"
        );
    }
    Ok(())
}

#[test]
fn definitions_require_unique_nonempty_identities_and_provenance() -> Result {
    let original = rectangle();
    assert!(
        Batch::new(
            crate::runtime::client(),
            vec![original.clone(), original.clone()]
        )
        .is_err()
    );
    for field in 0..3 {
        for text in ["", " \t ", "unreadable\nidentity"] {
            let mut profile = original.clone();
            match field {
                0 => profile.id = text.into(),
                1 => profile.vertical_datum = text.into(),
                _ => profile.profile_provenance = text.into(),
            }
            assert!(Batch::new(crate::runtime::client(), vec![profile]).is_err());
        }
    }
    Ok(())
}

#[test]
fn invalid_queries_reject_without_changing_immutable_geometry_or_retained_queries() -> Result {
    let batch = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let definitions = batch.definitions().to_vec();
    let original_query = batch.evaluate(&[1.0])?;
    let original = original_query.properties()?;
    for stages in [
        vec![],
        vec![1.0, 1.0],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![f64::NEG_INFINITY],
    ] {
        assert!(batch.evaluate(&stages).is_err());
        assert_eq!(batch.definitions(), definitions.as_slice());
    }
    let error = batch
        .evaluate(&[4.125])
        .err()
        .ok_or("Above-cap stage was accepted")?;
    assert!(matches!(
        error,
        CrossSectionError::UnsupportedStage {
            section: 0,
            stage_m: 4.125,
            maximum_stage_m: 4.0
        }
    ));
    let tiny = batch
        .evaluate(&[f64::from_bits(1)])
        .err()
        .ok_or("Positive unresolved geometry became dry")?;
    assert!(matches!(
        tiny,
        CrossSectionError::Numerical {
            section: Some(0),
            ..
        }
    ));
    assert_same(&original_query.properties()?[0], &original[0]);
    assert_same(&batch.evaluate(&[1.0])?.properties()?[0], &original[0]);
    Ok(())
}

#[test]
fn dry_stages_preserve_query_bits_and_avoid_irrelevant_subtraction_overflow() -> Result {
    let ordinary = Batch::new(crate::runtime::client(), vec![rectangle()])?;
    let signed_zero = ordinary.evaluate(&[-0.0])?.properties()?;
    assert_dry(&signed_zero[0], -0.0);
    let extreme = definition(
        "extreme-datum",
        &[
            (0.0, 1.1e308),
            (0.0, 1e308),
            (1e-308, 1e308),
            (1e-308, 1.1e308),
        ],
        1.1e308,
    );
    let batch = Batch::new(crate::runtime::client(), vec![extreme])?;
    assert_dry(&batch.evaluate(&[-f64::MAX])?.properties()?[0], -f64::MAX);
    Ok(())
}
