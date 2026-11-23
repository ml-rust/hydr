//! Test culverts and centred-difference checks of the culvert sensitivity.

use super::CAP;
use crate::structures::numerics::sensitivity::fixture::{
    Lane, MARGIN, STEP as DIFFERENCE, TestResult, check_bound, culvert_structures, evaluate,
};
use crate::structures::tensor::codes;
use crate::structures::{
    CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds, CulvertInlet,
    CulvertInletForm, CulvertParameters, DirectionPolicy, StructureCondition, StructureElement,
    StructureLaw, StructureOperatingPoint, StructureOperation,
};

/// Barrel, length, roughness and A and B inverts of a test culvert.
pub(super) struct Spec {
    pub(super) barrel: CulvertBarrel,
    pub(super) length_m: f64,
    pub(super) manning_n: f64,
    pub(super) inverts: [f64; 2],
}

/// A culvert at zero slope with both inverts at zero.
pub(super) fn level(barrel: CulvertBarrel, (length_m, manning_n): (f64, f64)) -> Spec {
    Spec {
        barrel,
        length_m,
        manning_n,
        inverts: [0.0, 0.0],
    }
}

/// An explicit inlet so weak that the outlet governs at every head.
pub(super) fn weak() -> TestResult<CulvertInlet> {
    Ok(CulvertInlet::explicit(
        CulvertInletForm::Two,
        [1e-3, 1.0, 1e-4, 0.05],
        false,
    )?)
}

/// A one-barrel culvert with `inlet` at end A, and at end B too when `both`.
pub(super) fn culvert(spec: Spec, inlet: CulvertInlet, both: bool) -> TestResult<StructureLaw> {
    let Spec {
        barrel,
        length_m,
        manning_n,
        inverts: [invert_a_m, invert_b_m],
    } = spec;
    let end = CulvertEnd::new(inlet, 0.5)?;
    Ok(StructureLaw::Culvert(CulvertParameters::new(
        barrel,
        CulvertAlignment {
            length_m,
            invert_a_m,
            invert_b_m,
        },
        manning_n,
        1,
        CulvertEnds {
            a: Some(end),
            b: both.then_some(end),
        },
        9.81,
        CulvertApplicability {
            coefficient_provenance: "Analytical coefficients".into(),
            barrel_condition_assertion: "Clean analytical barrel".into(),
        },
    )?))
}

/// Checks the bound of one copy of `law` per lane. Returns the number of flowing lanes.
pub(super) fn check(
    law: &StructureLaw,
    direction: DirectionPolicy,
    cases: &[Lane],
) -> TestResult<usize> {
    let structures = culvert_structures(
        |_| (law.clone(), direction, StructureOperatingPoint::Passive),
        cases.len(),
    )?;
    check_bound(&structures, cases)
}

/// Wet lanes over `upstream x downstream` that keep `MARGIN` from each `boundary` and from
/// level heads.
pub(super) fn pairs(upstream: &[f64], downstream: &[f64], boundaries: &[f64]) -> Vec<Lane> {
    let mut cases = Vec::new();
    for &a in upstream {
        for &b in downstream {
            let distance = boundaries
                .iter()
                .map(|z| (a - z).abs().min((b - z).abs()))
                .fold((a - b).abs(), f64::min);
            if distance >= MARGIN {
                cases.push(Lane {
                    stage: [a, b],
                    wet: [true, true],
                    step: DIFFERENCE * distance,
                });
            }
        }
    }
    cases
}

/// Checks the bound of one copy of `law` with `condition` per lane. Returns outlet-controlled
/// flowing lanes.
pub(super) fn certified_with_condition(
    law: StructureLaw,
    direction: DirectionPolicy,
    condition: StructureCondition,
    cases: &[Lane],
) -> TestResult<usize> {
    use crate::structures::coupling::CouplingStructures;
    use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
    let elements: Vec<StructureElement> = (0..cases.len())
        .map(|index| StructureElement {
            id: format!("c{index}"),
            endpoint_a: format!("a{index}"),
            endpoint_b: format!("b{index}"),
            datum: "analytical".into(),
            direction,
            law: law.clone(),
            condition: condition.clone(),
        })
        .collect();
    let operations: Vec<StructureOperation> = elements
        .iter()
        .map(|e| StructureOperation {
            structure_id: e.id.clone(),
            operating_point: StructureOperatingPoint::Passive,
        })
        .collect();
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let structures = CouplingStructures::with_culverts(client, elements, operations)?;
    check_bound(&structures, cases)?;
    let sample = evaluate(&structures, cases, None)?;
    let mut outlet = 0;
    for (index, (&q, &s)) in sample.discharge.iter().zip(&sample.sensitivity).enumerate() {
        assert!(
            q == 0.0 || s < CAP,
            "lane {index}: stages={:?} takes the cap",
            cases[index].stage
        );
        if q != 0.0 && sample.regime[index] == codes::CULVERT_OUTLET_CONTROL {
            outlet += 1;
        }
    }
    Ok(outlet)
}

/// Checks the bound against centred differences over `cases`. Every flowing lane must hold a
/// bound below the cap. Returns the number of outlet-controlled lanes.
pub(super) fn certified(spec: Spec, inlet: CulvertInlet, cases: &[Lane]) -> TestResult<usize> {
    let law = culvert(spec, inlet, false)?;
    let structures = culvert_structures(
        |_| {
            (
                law.clone(),
                DirectionPolicy::AtoBOnly,
                StructureOperatingPoint::Passive,
            )
        },
        cases.len(),
    )?;
    check_bound(&structures, cases)?;
    let sample = evaluate(&structures, cases, None)?;
    let mut outlet = 0;
    for (index, (&q, &s)) in sample.discharge.iter().zip(&sample.sensitivity).enumerate() {
        assert!(
            q == 0.0 || s < CAP,
            "lane {index}: stages={:?} takes the cap",
            cases[index].stage
        );
        if q != 0.0 && sample.regime[index] == codes::CULVERT_OUTLET_CONTROL {
            outlet += 1;
        }
    }
    Ok(outlet)
}
