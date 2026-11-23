//! Row and query admission: each rejection names its row, value and action.

use super::runtime::{Runtime, client};
use super::support::TestResult;
use hydr::structures::{
    CulvertBarrel, CulvertOutletBatch, CulvertOutletRow, OutletRegime, StructureError,
    StructureInputErrorKind as K,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

fn row(barrel: CulvertBarrel) -> CulvertOutletRow {
    CulvertOutletRow {
        barrel,
        length_m: 40.0,
        manning_n: 0.013,
        entrance_loss: 0.5,
        slope: 0.001,
        gravity_m_per_s2: 9.81,
        face_blockage: hydr::structures::BlockageFraction::NONE,
    }
}

fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

fn rejects<T>(
    result: Result<T, StructureError>,
    kind: K,
    element: Option<usize>,
    fragment: &str,
) -> TestResult {
    match result {
        Err(StructureError::InvalidInput {
            kind: found,
            element: at,
            context,
        }) if found == kind && at == element && context.contains(fragment) => Ok(()),
        Err(other) => Err(format!(
            "expected {kind:?} at {element:?} with {fragment:?}, found {other:?}"
        )
        .into()),
        Ok(_) => Err(format!("expected {kind:?} with {fragment:?}, found acceptance").into()),
    }
}

#[test]
fn row_admission_names_the_row_value_and_action() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let good = row(circle);
    let cases = [
        (
            CulvertOutletRow {
                length_m: 0.0,
                ..good
            },
            "length_m=0.0",
            "supply a finite positive barrel length in metres",
        ),
        (
            CulvertOutletRow {
                manning_n: f64::NAN,
                ..good
            },
            "manning_n=NaN",
            "supply a finite positive Manning n",
        ),
        (
            CulvertOutletRow {
                gravity_m_per_s2: -9.81,
                ..good
            },
            "gravity_m_per_s2=-9.81",
            "supply a finite positive gravity",
        ),
        (
            CulvertOutletRow {
                entrance_loss: -0.1,
                ..good
            },
            "entrance_loss=-0.1",
            "supply a finite nonnegative entrance loss coefficient",
        ),
        (
            CulvertOutletRow {
                slope: f64::INFINITY,
                ..good
            },
            "slope=inf",
            "supply a finite barrel slope",
        ),
        (
            CulvertOutletRow {
                slope: -0.016,
                ..good
            },
            "rise_m=1.2",
            "Supply an adverse fall of at most rise/2",
        ),
    ];
    for (bad, value, action) in cases {
        let result = CulvertOutletBatch::<Runtime>::new(client(), vec![good, bad]);
        rejects(result, K::Definition, Some(1), value)?;
        rejects(
            CulvertOutletBatch::<Runtime>::new(client(), vec![good, bad]),
            K::Definition,
            Some(1),
            action,
        )?;
    }
    rejects(
        CulvertOutletBatch::<Runtime>::new(client(), Vec::new()),
        K::Definition,
        None,
        "supply at least one culvert outlet row",
    )?;
    // An adverse fall below half the rise is admitted.
    let edge = CulvertOutletRow {
        slope: -0.0125,
        ..good
    };
    CulvertOutletBatch::<Runtime>::new(client(), vec![edge])?;
    Ok(())
}

#[test]
fn query_admission_names_the_row_and_zero_flow_holds_a_level_pool() -> TestResult {
    let circle = CulvertBarrel::circular(1.2)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.5)?;
    let batch = CulvertOutletBatch::<Runtime>::new(client(), vec![row(circle), row(rectangle)])?;
    rejects(
        batch.headwater(&tensor(&[0.5, -0.5])?, &tensor(&[0.0, 0.0])?),
        K::OperatingPoint,
        Some(1),
        "supply a nonnegative discharge",
    )?;
    rejects(
        batch.headwater(&tensor(&[0.5, 0.5])?, &tensor(&[f64::NAN, 0.0])?),
        K::NonfiniteInput,
        Some(0),
        "tailwater_m",
    )?;
    rejects(
        batch.headwater(&tensor(&[0.5])?, &tensor(&[0.0, 0.0])?),
        K::OperatingPoint,
        None,
        "supply one value per barrel",
    )?;
    // Zero flow: the still level TW - S L above the inlet invert, or a dry inlet.
    let rows = batch
        .headwater(&tensor(&[0.0, 0.0])?, &tensor(&[1.0, 0.02])?)?
        .project()?;
    let still = 1.0 - 0.001 * 40.0;
    if let [first, second] = rows.as_slice() {
        let holds = first.regime == OutletRegime::NoFlow
            && (first.headwater_m.0 - still).abs() < 1e-15
            && (first.headwater_m.1 - still).abs() < 1e-15
            && second.regime == OutletRegime::NoFlow
            && second.headwater_m == (0.0, 0.0);
        if !holds {
            return Err(format!("zero-flow rows {first:?} and {second:?}").into());
        }
    } else {
        return Err(format!("{} zero-flow rows", rows.len()).into());
    }
    Ok(())
}
