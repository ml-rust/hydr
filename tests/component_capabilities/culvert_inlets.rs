//! Culvert inlet declaration checked against its admission code.

use crate::runtime::{Runtime, client};
use hydr::capability::{ArithmeticRequirement as A, UnsupportedReason};
use hydr::structures::{
    CulvertBarrel, CulvertInlet, CulvertInletBatch, CulvertInletForm, CulvertInletRow,
    StructureError, StructureInputErrorKind as K, culvert_inlet_capabilities,
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

fn row(barrel: CulvertBarrel, inlet: CulvertInlet, slope: f64, gravity: f64) -> CulvertInletRow {
    CulvertInletRow {
        barrel,
        inlet,
        slope,
        gravity_m_per_s2: gravity,
    }
}

fn batch(rows: Vec<CulvertInletRow>) -> Result<(), StructureError> {
    dropped(CulvertInletBatch::<Runtime>::new(client(), rows))
}

/// Cases for each `RejectedByAdmission` entry, or `None` for an unknown id.
fn cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let circle = CulvertBarrel::circular(1.0)?;
    let rectangle = CulvertBarrel::rectangular_box(2.0, 1.0)?;
    let coded = CulvertInlet::hds5(1)?;
    Ok(Some(match id {
        "unadmitted-inlet-definition" => vec![
            (
                dropped(CulvertInlet::hds5(40)),
                K::Definition,
                "supply an HDS-5 code from 1 to 26",
            ),
            (
                dropped(CulvertInlet::explicit(
                    CulvertInletForm::Two,
                    [0.5, 0.0, 0.03, 0.8],
                    false,
                )),
                K::Definition,
                "supply a finite positive inlet coefficient M",
            ),
            (
                dropped(CulvertInlet::explicit(
                    CulvertInletForm::One,
                    [0.5, 0.6, 0.03, f64::NAN],
                    false,
                )),
                K::Definition,
                "supply a finite inlet coefficient Y",
            ),
        ],
        "invalid-inlet-row" => vec![
            (
                batch(Vec::new()),
                K::Definition,
                "supply at least one culvert inlet row",
            ),
            (
                batch(vec![row(circle, coded, f64::INFINITY, 9.81)]),
                K::Definition,
                "supply a finite barrel slope",
            ),
            (
                batch(vec![row(circle, coded, 0.01, 0.0)]),
                K::Definition,
                "supply a finite positive gravity",
            ),
            (
                batch(vec![row(rectangle, coded, 0.01, 9.81)]),
                K::Definition,
                "the code is for Circular barrels",
            ),
            (
                batch(vec![row(
                    CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
                    CulvertInlet::explicit(
                        CulvertInletForm::One,
                        [0.0083, 2.0, 0.0379, 0.69],
                        false,
                    )?,
                    0.01,
                    9.81,
                )]),
                K::Definition,
                "arc barrel geometry not yet available",
            ),
        ],
        "falling-transition" => vec![
            (
                batch(vec![row(
                    rectangle,
                    CulvertInlet::explicit(CulvertInletForm::Two, [1.0, 1.0, 0.01, 0.5], false)?,
                    0.0,
                    9.81,
                )]),
                K::Definition,
                "the headwater falls across the transition",
            ),
            (
                batch(vec![row(rectangle, CulvertInlet::hds5(9)?, 0.0, 1.0)]),
                K::Definition,
                "is not below the crown",
            ),
        ],
        "invalid-query-input" => {
            let inlets = CulvertInletBatch::<Runtime>::new(
                client(),
                vec![
                    row(circle, coded, 0.01, 9.81),
                    row(rectangle, CulvertInlet::hds5(10)?, 0.01, 9.81),
                ],
            )?;
            vec![
                (
                    dropped(inlets.headwater(&tensor(&[0.5, -0.5])?)),
                    K::OperatingPoint,
                    "supply a nonnegative discharge",
                ),
                (
                    dropped(inlets.headwater(&tensor(&[f64::NAN, 0.5])?)),
                    K::NonfiniteInput,
                    "supply a finite discharge",
                ),
                (
                    dropped(inlets.headwater(&tensor(&[0.5])?)),
                    K::OperatingPoint,
                    "supply one value per barrel",
                ),
            ]
        }
        _ => return Ok(None),
    }))
}

#[test]
fn culvert_inlet_attaches_only_to_the_culvert_law_and_declares_its_arithmetic() {
    let capabilities = culvert_inlet_capabilities();
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
fn culvert_inlet_admission_rejects_every_declared_case() -> TestResult {
    let mut errors = Vec::new();
    for entry in culvert_inlet_capabilities().unsupported {
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
