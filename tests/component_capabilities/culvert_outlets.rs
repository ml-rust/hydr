//! Culvert outlet declaration checked against its admission code.

use crate::runtime::{Runtime, client};
use hydr::capability::{ArithmeticRequirement as A, UnsupportedReason};
use hydr::structures::{
    CulvertBarrel, CulvertOutletBatch, CulvertOutletRow, StructureError,
    StructureInputErrorKind as K, culvert_outlet_capabilities,
};
use numr::runtime::RuntimeClient;
use numr::tensor::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// One rejection: the result, its expected input kind and a message fragment.
type Rejection = (Result<(), StructureError>, K, &'static str);

fn tensor(values: &[f64]) -> Result<Tensor<Runtime>, numr::error::Error> {
    Tensor::from_slice(values, &[values.len()], client().device())
}

fn dropped<T>(result: Result<T, StructureError>) -> Result<(), StructureError> {
    result.map(|_| ())
}

fn row(barrel: CulvertBarrel) -> CulvertOutletRow {
    CulvertOutletRow {
        barrel,
        length_m: 40.0,
        manning_n: 0.013,
        entrance_loss: 0.5,
        slope: 0.001,
        gravity_m_per_s2: 9.81,
    }
}

fn batch(rows: Vec<CulvertOutletRow>) -> Result<(), StructureError> {
    dropped(CulvertOutletBatch::<Runtime>::new(client(), rows))
}

/// Cases for each `RejectedByAdmission` entry, or `None` for an unknown id.
fn cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let circle = CulvertBarrel::circular(1.0)?;
    let good = row(circle);
    Ok(Some(match id {
        "invalid-outlet-row" => vec![
            (
                batch(Vec::new()),
                K::Definition,
                "supply at least one culvert outlet row",
            ),
            (
                batch(vec![CulvertOutletRow {
                    length_m: f64::INFINITY,
                    ..good
                }]),
                K::Definition,
                "supply a finite positive barrel length",
            ),
            (
                batch(vec![CulvertOutletRow {
                    entrance_loss: f64::NAN,
                    ..good
                }]),
                K::Definition,
                "supply a finite nonnegative entrance loss coefficient",
            ),
            (
                batch(vec![CulvertOutletRow {
                    slope: f64::NAN,
                    ..good
                }]),
                K::Definition,
                "supply a finite barrel slope",
            ),
            (
                batch(vec![row(CulvertBarrel::vertical_ellipse(
                    0.9652,
                    1.524,
                    0.419_475_92,
                )?)]),
                K::Definition,
                "arc barrel geometry not yet available",
            ),
        ],
        "adverse-beyond-range" => vec![(
            batch(vec![CulvertOutletRow {
                slope: -0.02,
                ..good
            }]),
            K::Definition,
            "more than half the rise above the inlet invert",
        )],
        "invalid-query-input" => {
            let outlets = CulvertOutletBatch::<Runtime>::new(client(), vec![good, good])?;
            vec![
                (
                    dropped(outlets.headwater(&tensor(&[0.5, -0.5])?, &tensor(&[0.0, 0.0])?)),
                    K::OperatingPoint,
                    "supply a nonnegative discharge",
                ),
                (
                    dropped(outlets.headwater(&tensor(&[0.5, 0.5])?, &tensor(&[0.0, -1.0])?)),
                    K::OperatingPoint,
                    "tailwater_m",
                ),
                (
                    dropped(outlets.jump(&tensor(&[f64::NAN, 0.5])?, &tensor(&[0.0, 0.0])?)),
                    K::NonfiniteInput,
                    "supply a finite discharge",
                ),
                (
                    dropped(outlets.headwater(&tensor(&[0.5])?, &tensor(&[0.0, 0.0])?)),
                    K::OperatingPoint,
                    "supply one value per barrel",
                ),
            ]
        }
        _ => return Ok(None),
    }))
}

#[test]
fn culvert_outlet_attaches_only_to_the_culvert_law_and_declares_its_arithmetic() {
    let capabilities = culvert_outlet_capabilities();
    assert!(capabilities.endpoints.is_empty());
    assert!(capabilities.regimes.is_empty());
    assert!(capabilities.sources.is_empty());
    assert!(capabilities.options.is_empty());
    assert!(capabilities.nested.is_empty());
    assert_eq!(
        capabilities.attachable_to,
        [hydr::capability::ComponentId::Culvert]
    );
    assert_eq!(
        capabilities.arithmetic,
        [
            A::F64Ieee754Basic,
            A::NearestEvenGradualUnderflow,
            A::RoundToNearestSqrt,
            A::FusedMulAdd,
            A::CorrectlyRoundedElementary,
        ]
    );
}

#[test]
fn culvert_outlet_admission_rejects_every_declared_case() -> TestResult {
    let mut errors = Vec::new();
    for entry in culvert_outlet_capabilities().unsupported {
        if entry.reason != UnsupportedReason::RejectedByAdmission {
            continue;
        }
        let Some(rejections) = cases(entry.id)? else {
            errors.push(format!("`{}` has no rejecting case", entry.id));
            continue;
        };
        for (result, kind, fragment) in rejections {
            match result {
                Err(StructureError::InvalidInput {
                    kind: found,
                    context,
                    ..
                }) if found == kind && context.contains(fragment) => {}
                other => errors.push(format!(
                    "`{}` expected {kind:?} containing {fragment:?}, found {other:?}",
                    entry.id
                )),
            }
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors.join("\n").into())
    }
}
