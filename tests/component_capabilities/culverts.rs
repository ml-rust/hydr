//! Culvert structure-law declaration checked against its admission code.

use crate::runtime::{Runtime, client};
use hydr::capability::{ArithmeticRequirement as A, ComponentId, UnsupportedReason};
use hydr::structures::{
    BlockageFraction, CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertBlockage,
    CulvertEnd, CulvertEnds, CulvertInlet, CulvertInletForm, CulvertParameters, DirectionPolicy,
    EndpointHead, HeadConvention, HeadSupport, StructureBatch, StructureBlockage,
    StructureCondition, StructureElement, StructureError, StructureHeads,
    StructureInputErrorKind as K, StructureLaw, culvert_capabilities,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

/// One rejection: the result, its expected input kind and a message fragment.
type Rejection = (Result<(), StructureError>, K, &'static str);

fn dropped<T>(result: Result<T, StructureError>) -> Result<(), StructureError> {
    result.map(|_| ())
}

fn evidence() -> CulvertApplicability {
    CulvertApplicability {
        coefficient_provenance: "HDS-5 Table A.1".into(),
        barrel_condition_assertion: "Clean synthetic barrel".into(),
    }
}

fn level() -> CulvertAlignment {
    CulvertAlignment {
        length_m: 30.0,
        invert_a_m: 0.0,
        invert_b_m: 0.0,
    }
}

fn parameters(
    alignment: CulvertAlignment,
    count: u32,
    ends: CulvertEnds,
) -> Result<CulvertParameters, StructureError> {
    CulvertParameters::new(
        CulvertBarrel::circular(1.0)?,
        alignment,
        0.013,
        count,
        ends,
        9.81,
        evidence(),
    )
}

fn element(law: CulvertParameters, direction: DirectionPolicy) -> StructureElement {
    StructureElement {
        id: "culvert".into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: "datum".into(),
        direction,
        law: StructureLaw::Culvert(law),
        condition: StructureCondition::AsBuilt,
    }
}

fn with_culverts(element: StructureElement) -> Result<(), StructureError> {
    dropped(StructureBatch::<Runtime>::with_culverts(
        client(),
        vec![element],
    ))
}

fn packet(a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: "culvert".into(),
        endpoint_a: EndpointHead {
            id: "a".into(),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: "b".into(),
            support: b,
        },
        datum: "datum".into(),
        convention: HeadConvention::QuiescentStage,
    }
}

/// Cases for each `RejectedByAdmission` entry, or `None` for an unknown id.
fn cases(id: &str) -> Result<Option<Vec<Rejection>>, Box<dyn std::error::Error>> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    let one = CulvertEnds {
        a: Some(end),
        b: None,
    };
    let both = CulvertEnds {
        a: Some(end),
        b: Some(end),
    };
    Ok(Some(match id {
        "invalid-culvert-definition" => vec![
            (
                dropped(parameters(level(), 0, one)),
                K::Definition,
                "supply at least one barrel",
            ),
            (
                dropped(parameters(level(), 1, CulvertEnds { a: None, b: None })),
                K::Definition,
                "supply the culvert end of at least one flow direction",
            ),
            (
                dropped(parameters(
                    CulvertAlignment {
                        invert_a_m: f64::INFINITY,
                        ..level()
                    },
                    1,
                    one,
                )),
                K::Definition,
                "supply finite invert elevations",
            ),
            (
                dropped(CulvertEnd::new(CulvertInlet::hds5(1)?, -1.0)),
                K::Definition,
                "supply a finite nonnegative entrance loss coefficient",
            ),
            (
                dropped(parameters(
                    level(),
                    1,
                    CulvertEnds {
                        a: Some(CulvertEnd::new(CulvertInlet::hds5(12)?, 0.5)?),
                        b: None,
                    },
                )),
                K::Definition,
                "the code is for Box barrels",
            ),
            (
                dropped(parameters(
                    CulvertAlignment {
                        invert_b_m: 0.6,
                        ..level()
                    },
                    1,
                    one,
                )),
                K::Definition,
                "Supply an adverse fall of at most rise/2",
            ),
        ],
        "culvert-end-mismatch" => vec![
            (
                with_culverts(element(
                    parameters(level(), 1, both)?,
                    DirectionPolicy::AtoBOnly,
                )),
                K::Definition,
                "declare the culvert end of each admitted direction and no other",
            ),
            (
                with_culverts(element(
                    parameters(level(), 1, one)?,
                    DirectionPolicy::Bidirectional,
                )),
                K::Definition,
                "Bidirectional needs both ends",
            ),
        ],
        "culvert-falling-transition" => {
            let falling =
                CulvertInlet::explicit(CulvertInletForm::Two, [1.0, 1.0, 0.01, 0.0], false)?;
            let ends = CulvertEnds {
                a: Some(CulvertEnd::new(falling, 0.5)?),
                b: None,
            };
            vec![(
                with_culverts(element(
                    parameters(level(), 1, ends)?,
                    DirectionPolicy::AtoBOnly,
                )),
                K::Definition,
                "the headwater falls across the transition",
            )]
        }
        "culvert-without-certified-client" => vec![(
            dropped(StructureBatch::<Runtime>::new(
                client(),
                vec![element(
                    parameters(level(), 1, one)?,
                    DirectionPolicy::AtoBOnly,
                )],
            )),
            K::Definition,
            "Build the batch with StructureBatch::with_culverts",
        )],
        "buried-dry-end" => {
            let batch = StructureBatch::<Runtime>::with_culverts(
                client(),
                vec![element(
                    parameters(level(), 1, one)?,
                    DirectionPolicy::AtoBOnly,
                )],
            )?;
            vec![(
                dropped(batch.propose(&[packet(
                    HeadSupport::Wet { stage_m: 0.8 },
                    HeadSupport::Dry {
                        bed_elevation_m: 0.2,
                    },
                )])),
                K::UnsupportedRegime,
                "a known dry culvert end needs its bed at or below its invert",
            )]
        }
        "culvert-exit-face-blockage" => {
            let blocked_a = StructureCondition::Blocked(StructureBlockage::culvert(
                CulvertBlockage {
                    face_a: BlockageFraction::new(0.25)?,
                    ..CulvertBlockage::NONE
                },
                "Debris survey".into(),
            )?);
            let blocked_b = StructureCondition::Blocked(StructureBlockage::culvert(
                CulvertBlockage {
                    face_b: BlockageFraction::new(0.25)?,
                    ..CulvertBlockage::NONE
                },
                "Debris survey".into(),
            )?);
            let mut two_way = element(
                parameters(level(), 1, both)?,
                DirectionPolicy::Bidirectional,
            );
            two_way.condition = blocked_a;
            let mut one_way = element(parameters(level(), 1, one)?, DirectionPolicy::AtoBOnly);
            one_way.condition = blocked_b;
            vec![
                (with_culverts(two_way), K::Definition, "exit-face blockage"),
                (with_culverts(one_way), K::Definition, "exit-face blockage"),
            ]
        }
        _ => return Ok(None),
    }))
}

#[test]
fn culvert_law_nests_its_queries_and_declares_their_arithmetic() {
    let capabilities = culvert_capabilities();
    assert!(capabilities.endpoints.is_empty());
    assert!(capabilities.regimes.is_empty());
    assert!(capabilities.sources.is_empty());
    assert!(capabilities.options.is_empty());
    assert_eq!(
        capabilities.attachable_to,
        [ComponentId::SurfaceChannels, ComponentId::CoupledLevelPools]
    );
    assert_eq!(
        capabilities.nested,
        [ComponentId::CulvertInlet, ComponentId::CulvertOutlet]
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
    for id in [
        "culvert-barrel-storage",
        "culvert-composite-roughness",
        "culvert-uniform-sediment",
        "culvert-buried-inlet-coefficients",
        "culvert-face-loss",
        "culvert-metre-products",
    ] {
        assert!(
            capabilities.limits.iter().any(|limit| limit.id == id),
            "{id}"
        );
    }
    for removed in ["culvert-arc-barrel", "culvert-blockage"] {
        assert!(
            capabilities
                .unsupported
                .iter()
                .all(|entry| entry.id != removed),
            "{removed}"
        );
    }
    assert!(
        capabilities
            .limits
            .iter()
            .all(|limit| limit.id != "culvert-floor-roughness")
    );
}

#[test]
fn culvert_law_admission_rejects_every_declared_case() -> TestResult {
    let mut errors = Vec::new();
    for entry in culvert_capabilities().unsupported {
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
