//! Culvert barrel declaration checked against its admission code.

use crate::runtime::{Runtime, client};
use hydr::capability::{ArithmeticRequirement as A, UnsupportedReason};
use hydr::structures::{
    CulvertBarrel, CulvertBarrelBatch, StructureError, StructureInputErrorKind as K,
    culvert_barrel_capabilities,
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

/// A circle and a box of unit rise.
fn pair() -> Result<CulvertBarrelBatch<Runtime>, StructureError> {
    CulvertBarrelBatch::new(
        client(),
        vec![
            CulvertBarrel::circular(1.0)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
        ],
    )
}

/// Cases for each `RejectedByAdmission` entry, or `None` for an unknown id.
fn cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let batch = pair()?;
    let flows = tensor(&[0.5, 0.5])?;
    let roughness = tensor(&[0.013, 0.013])?;
    let slopes = tensor(&[0.01, 0.01])?;
    Ok(Some(match id {
        "unrepresentable-barrel-dimensions" => vec![
            (
                dropped(CulvertBarrel::circular(-1.0)),
                K::Definition,
                "supply a finite positive barrel diameter",
            ),
            (
                dropped(CulvertBarrel::rectangular_box(1.0, f64::NAN)),
                K::Definition,
                "supply a finite positive barrel rise",
            ),
            (
                dropped(CulvertBarrel::circular(1e-300)),
                K::Definition,
                "rise times 2^-30 is subnormal",
            ),
            (
                dropped(CulvertBarrel::circular(1e103)),
                K::Definition,
                "times 64 overflows F64",
            ),
        ],
        "arc-barrel-dimensions" => vec![
            (
                dropped(CulvertBarrel::horizontal_ellipse(1.2, 1.5, 0.3)),
                K::Definition,
                "a horizontal ellipse needs span above rise",
            ),
            (
                dropped(CulvertBarrel::vertical_ellipse(1.0, 2.0, 0.5)),
                K::Definition,
                "a vertical ellipse needs an end radius below half the span",
            ),
            (
                dropped(CulvertBarrel::pipe_arch(2.0, 1.8, 0.5, 2.0)),
                K::Definition,
                "a pipe-arch needs the rise above the corner centres",
            ),
            (
                dropped(CulvertBarrel::arch(2.0, 0.4)),
                K::Definition,
                "a flat-floor arch needs a rise of at least a quarter of the span",
            ),
        ],
        "arc-barrel-certificate" => {
            // Admitted in metres, but its invert radius over the rise has a cube past F64.
            let flat = CulvertBarrel::horizontal_ellipse(2e-5, 1e-100, 2.5e-101)?;
            vec![(
                dropped(CulvertBarrelBatch::<Runtime>::new(
                    client(),
                    vec![CulvertBarrel::circular(1.0)?, flat],
                )),
                K::Definition,
                "the barrel context scales the barrel by a power of two to a rise in [1, 2), and the scaled barrel fails admission",
            )]
        }
        "empty-barrel-batch" => vec![(
            dropped(CulvertBarrelBatch::<Runtime>::new(client(), Vec::new())),
            K::Definition,
            "supply at least one culvert barrel",
        )],
        "invalid-query-input" => vec![
            (
                dropped(batch.geometry(&tensor(&[0.5, -0.5])?)),
                K::OperatingPoint,
                "supply a nonnegative depth",
            ),
            (
                dropped(batch.critical_depth(&tensor(&[f64::NAN, 0.5])?, 9.81)),
                K::NonfiniteInput,
                "supply a finite discharge",
            ),
            (
                dropped(batch.critical_depth(&flows, 0.0)),
                K::OperatingPoint,
                "supply a finite positive gravity",
            ),
            (
                dropped(batch.normal_depth(&flows, &tensor(&[0.013, 0.0])?, &slopes)),
                K::OperatingPoint,
                "supply a positive Manning n",
            ),
            (
                dropped(batch.normal_depth(&flows, &roughness, &tensor(&[0.01, f64::INFINITY])?)),
                K::NonfiniteInput,
                "supply a finite barrel slope",
            ),
            (
                dropped(batch.geometry(&tensor(&[0.5])?)),
                K::OperatingPoint,
                "supply one value per barrel",
            ),
        ],
        "range-reaching-crown" => vec![
            (
                dropped(batch.geometry_over(&tensor(&[0.2, 0.6])?, &tensor(&[0.4, 0.5])?)),
                K::OperatingPoint,
                "supply lower_m <= upper_m",
            ),
            (
                dropped(batch.geometry_over(&tensor(&[0.2, 0.2])?, &tensor(&[0.4, 1.0])?)),
                K::OperatingPoint,
                "the range reaches the crown",
            ),
        ],
        _ => return Ok(None),
    }))
}

#[test]
fn culvert_barrel_declares_no_attachments_and_its_arithmetic() {
    let capabilities = culvert_barrel_capabilities();
    assert!(capabilities.endpoints.is_empty());
    assert!(capabilities.regimes.is_empty());
    assert!(capabilities.sources.is_empty());
    assert!(capabilities.options.is_empty());
    assert!(capabilities.nested.is_empty());
    assert!(capabilities.attachable_to.is_empty());
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
    let limits: Vec<&str> = capabilities.limits.iter().map(|limit| limit.id).collect();
    for id in ["culvert-floor-roughness", "scaled-values-outside-f64"] {
        assert!(limits.contains(&id), "{id} missing from {limits:?}");
    }
    let other = capabilities
        .unsupported
        .iter()
        .find(|entry| entry.id == "other-barrel-shapes")
        .map(|entry| entry.summary);
    assert_eq!(
        other,
        Some(
            "No corrugated metal box, long-span structural plate, true ellipse or tapered sections."
        )
    );
}

#[test]
fn culvert_barrel_admission_rejects_every_declared_case() -> TestResult {
    let mut errors = Vec::new();
    for entry in culvert_barrel_capabilities().unsupported {
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
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    Ok(())
}
