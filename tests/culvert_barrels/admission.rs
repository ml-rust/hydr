//! Query and batch admission name the offending row, value and action.

use super::runtime::{Runtime, client};
use super::support::{TestResult, batch, tensor};
use hydr::structures::{
    CulvertBarrel, CulvertBarrelBatch, StructureError, StructureInputErrorKind,
};

fn rejection(
    result: Result<impl Sized, StructureError>,
) -> Option<(Option<usize>, StructureInputErrorKind, String)> {
    match result {
        Err(StructureError::InvalidInput {
            element,
            kind,
            context,
        }) => Some((element, kind, context)),
        _ => None,
    }
}

#[test]
fn depth_queries_reject_the_first_bad_row() -> TestResult {
    let barrels = batch(vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ])?;
    let cases = [
        (
            vec![0.5, -0.5],
            Some(1),
            StructureInputErrorKind::OperatingPoint,
            "depth_m=-0.5: supply a nonnegative depth above the invert in metres",
        ),
        (
            vec![f64::NAN, -1.0],
            Some(0),
            StructureInputErrorKind::NonfiniteInput,
            "depth_m=NaN: supply a finite depth above the invert in metres",
        ),
        (
            vec![0.5, f64::INFINITY],
            Some(1),
            StructureInputErrorKind::NonfiniteInput,
            "depth_m=inf: supply a finite depth",
        ),
        (
            vec![0.5, 0.5, 0.5],
            None,
            StructureInputErrorKind::OperatingPoint,
            "depth_m shape=[3],barrels=2: supply one value per barrel in a [2] tensor",
        ),
    ];
    for (depths, row, kind, message) in cases {
        let found = rejection(barrels.geometry(&tensor(&depths)?));
        assert!(
            matches!(&found, Some((element, k, context)) if *element == row && *k == kind && context.contains(message)),
            "{message}: {found:?}"
        );
    }
    Ok(())
}

#[test]
fn critical_queries_reject_bad_discharges_and_gravity() -> TestResult {
    let barrels = batch(vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ])?;
    let cases = [
        (
            vec![0.5, -0.5],
            9.81,
            Some(1),
            StructureInputErrorKind::OperatingPoint,
            "discharge_m3_per_s=-0.5: supply a nonnegative discharge per barrel in cubic metres per second",
        ),
        (
            vec![f64::NAN, 1.0],
            9.81,
            Some(0),
            StructureInputErrorKind::NonfiniteInput,
            "discharge_m3_per_s=NaN: supply a finite discharge per barrel",
        ),
        (
            vec![0.5, 0.5],
            0.0,
            None,
            StructureInputErrorKind::OperatingPoint,
            "gravity_m_per_s2=0.0: supply a finite positive gravity in metres per second squared",
        ),
        (
            vec![0.5, 0.5],
            f64::NAN,
            None,
            StructureInputErrorKind::OperatingPoint,
            "gravity_m_per_s2=NaN: supply a finite positive gravity",
        ),
        (
            vec![0.5],
            9.81,
            None,
            StructureInputErrorKind::OperatingPoint,
            "discharge_m3_per_s shape=[1],barrels=2: supply one value per barrel in a [2] tensor",
        ),
    ];
    for (discharges, gravity, row, kind, message) in cases {
        let found = rejection(barrels.critical_depth(&tensor(&discharges)?, gravity));
        assert!(
            matches!(&found, Some((element, k, context)) if *element == row && *k == kind && context.contains(message)),
            "{message}: {found:?}"
        );
    }
    Ok(())
}

#[test]
fn normal_queries_reject_bad_discharge_roughness_and_slope() -> TestResult {
    let barrels = batch(vec![
        CulvertBarrel::circular(1.0)?,
        CulvertBarrel::rectangular_box(2.0, 1.0)?,
    ])?;
    let good = [0.5, 0.5];
    let rough = [0.013, 0.013];
    let tilt = [0.01, -0.01];
    let cases = [
        (
            [&[0.5, -0.5][..], &rough, &tilt],
            Some(1),
            StructureInputErrorKind::OperatingPoint,
            "discharge_m3_per_s=-0.5: supply a nonnegative discharge per barrel",
        ),
        (
            [&good, &[0.013, 0.0][..], &tilt],
            Some(1),
            StructureInputErrorKind::OperatingPoint,
            "manning_n=0.0: supply a positive Manning n in s/m^(1/3)",
        ),
        (
            [&good, &[f64::NAN, -1.0][..], &tilt],
            Some(0),
            StructureInputErrorKind::NonfiniteInput,
            "manning_n=NaN: supply a finite Manning n",
        ),
        (
            [&good, &rough, &[0.01, f64::NEG_INFINITY][..]],
            Some(1),
            StructureInputErrorKind::NonfiniteInput,
            "slope=-inf: supply a finite barrel slope in metres per metre",
        ),
        (
            [&good, &rough, &[0.01][..]],
            None,
            StructureInputErrorKind::OperatingPoint,
            "slope shape=[1],barrels=2: supply one value per barrel in a [2] tensor",
        ),
    ];
    for ([discharges, roughness, slopes], row, kind, message) in cases {
        let found = rejection(barrels.normal_depth(
            &tensor(discharges)?,
            &tensor(roughness)?,
            &tensor(slopes)?,
        ));
        assert!(
            matches!(&found, Some((element, k, context)) if *element == row && *k == kind && context.contains(message)),
            "{message}: {found:?}"
        );
    }
    Ok(())
}

#[test]
fn empty_batches_reject_with_action() {
    let found = rejection(CulvertBarrelBatch::<Runtime>::new(client(), Vec::new()));
    assert!(matches!(
        &found,
        Some((None, StructureInputErrorKind::Definition, context))
            if context == "barrels=[]: supply at least one culvert barrel"
    ));
}
