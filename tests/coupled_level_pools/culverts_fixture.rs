//! Culvert cases of the coupled level pools: constant-area pools joined by circular concrete
//! culverts and a roadway weir.
//!
//! Each case returns its setup and initial state, so the CPU and CUDA runs share it.

use super::fixture::{DATUM, Result, config};
use hydr::coupling::{
    CoupledLevelPoolInitialState, CoupledLevelPoolSetup, CoupledPoolAttachment,
    CoupledPoolConnection, CoupledPoolInflowBinding, CoupledPoolSupport,
};
use hydr::storage::{LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState};
use hydr::structures::{
    CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds, CulvertInlet,
    CulvertParameters, DirectionPolicy, StructureCondition, StructureElement, StructureLaw,
    StructureOperatingPoint, StructureOperation, SuppressedRectangularWeirParameters,
    WeirApplicability,
};

/// Height of every pool curve above its origin at zero, in m.
const HEIGHT_M: f64 = 3.0;

/// Gravity of every culvert, bit-identical to the drain reference.
const GRAVITY: f64 = 9.81;

/// Constant-area pool `id` of `area` m² on origin zero.
fn pool(id: &str, area: f64) -> Result<LevelPoolElement> {
    Ok(LevelPoolElement {
        id: id.into(),
        curve: LevelPoolCurve::new(
            vec![
                LevelPoolKnot {
                    volume_m3: 0.0,
                    stage_m: 0.0,
                },
                LevelPoolKnot {
                    volume_m3: area * HEIGHT_M,
                    stage_m: HEIGHT_M,
                },
            ],
            DATUM.into(),
            "Exact synthetic constant plan area".into(),
        )?,
        connected_pool_assertion: "Separate connected, approximately level synthetic pool".into(),
        storage_support_identity: format!("synthetic-storage-{id}"),
    })
}

/// Circular concrete culvert `id` of `rise` m from pool `a` to pool `b`.
///
/// It runs `length` m between `inverts` `[A, B]`. Each end has a square-edge headwall: HDS-5
/// code 1 and entrance loss `0.5`. Manning `n` is `0.013`. `two_way` declares the B end too,
/// else only A-to-B flow has an end.
pub fn culvert(
    id: &str,
    [a, b]: [&str; 2],
    (rise, length): (f64, f64),
    inverts: [f64; 2],
    two_way: bool,
) -> Result<StructureElement> {
    let end = CulvertEnd::new(CulvertInlet::hds5(1)?, 0.5)?;
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: a.into(),
        endpoint_b: b.into(),
        datum: DATUM.into(),
        direction: if two_way {
            DirectionPolicy::Bidirectional
        } else {
            DirectionPolicy::AtoBOnly
        },
        law: StructureLaw::Culvert(CulvertParameters::new(
            CulvertBarrel::circular(rise)?,
            CulvertAlignment {
                length_m: length,
                invert_a_m: inverts[0],
                invert_b_m: inverts[1],
            },
            0.013,
            1,
            CulvertEnds {
                a: Some(end),
                b: two_way.then_some(end),
            },
            GRAVITY,
            CulvertApplicability {
                coefficient_provenance: "HDS-5 Table A.1 code 1".into(),
                barrel_condition_assertion: "Clean synthetic concrete barrel".into(),
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

/// Road weir `id` 2 m long from pool `a` to pool `b`, crest at `crest`.
pub fn road(id: &str, [a, b]: [&str; 2], crest: f64) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: a.into(),
        endpoint_b: b.into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            2.0,
            crest,
            1.7,
            WeirApplicability {
                coefficient_provenance: "Declared roadway weir coefficient".into(),
                free_nappe_and_approach_assertion: "Suppressed road crest, quiescent approach"
                    .into(),
                submergence_correction_assertion: "Villemonte correction adopted".into(),
                symmetric_reversal: true,
            },
        )?),
        condition: StructureCondition::AsBuilt,
    })
}

/// External inflow binding `boundary` into pool `pool`.
pub fn inflow(boundary: &str, pool: &str) -> CoupledPoolInflowBinding {
    CoupledPoolInflowBinding {
        boundary_id: boundary.into(),
        pool_id: pool.into(),
        input_id: format!("selected-input-{boundary}"),
        physical_support_id: format!("external-support-{boundary}"),
        boundary_assertion: "Prescribed synthetic inflow enters the named receiving pool".into(),
        ownership_assertion: "External origin excludes represented pool storage and local rainfall"
            .into(),
    }
}

/// Pools `(id, area, stage)` joined by `structures`, every attachment on its pool origin.
pub fn system(
    pools: &[(&str, f64, f64)],
    structures: Vec<StructureElement>,
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    let elements = pools
        .iter()
        .map(|&(id, area, _)| pool(id, area))
        .collect::<Result<Vec<_>>>()?;
    let attachment = |id: &str| CoupledPoolAttachment {
        pool_id: id.into(),
        bed_elevation_m: 0.0,
        physical_connection_assertion: "The structure opens onto the named synthetic pool".into(),
    };
    let connections = structures
        .iter()
        .map(|element| CoupledPoolConnection {
            structure_id: element.id.clone(),
            endpoint_a: attachment(&element.endpoint_a),
            endpoint_b: attachment(&element.endpoint_b),
        })
        .collect();
    let operations = structures
        .iter()
        .map(|element| StructureOperation {
            structure_id: element.id.clone(),
            operating_point: StructureOperatingPoint::Passive,
        })
        .collect();
    Ok((
        CoupledLevelPoolSetup {
            source_policies: None,
            external_inflows: Vec::new(),
            pool_support: pools
                .iter()
                .map(|&(id, _, _)| CoupledPoolSupport {
                    pool_id: id.into(),
                    zero_volume_stage_m: 0.0,
                    zero_volume_limit_assertion: "The synthetic pool empties at its origin".into(),
                })
                .collect(),
            pools: elements,
            structures,
            operations,
            connections,
            config: config(),
        },
        CoupledLevelPoolInitialState {
            time_s: 0.0,
            states: pools
                .iter()
                .map(|&(_, area, stage)| LevelPoolState {
                    volume_m3: area * stage,
                })
                .collect(),
        },
    ))
}

/// A lagoon at 1.4 m and a sea at 1 m joined by a two-way culvert `tide` and a one-end culvert
/// `flap` from the lagoon. A boundary `rise` feeds the sea.
pub fn reversal() -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    let (mut setup, initial) = system(
        &[("lagoon", 8.0, 1.4), ("sea", 4.0, 1.0)],
        vec![
            culvert("tide", ["lagoon", "sea"], (0.3, 10.0), [0.0, 0.0], true)?,
            culvert("flap", ["lagoon", "sea"], (0.3, 10.0), [0.0, 0.0], false)?,
        ],
    )?;
    setup.external_inflows = vec![inflow("rise", "sea")];
    Ok((setup, initial))
}

/// An upper pool at 1.8 m over a road crest at 1.5 m, a lower pool at 1 m, a drowned culvert
/// under the road.
pub fn roadway() -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    system(
        &[("lower", 30.0, 1.0), ("upper", 10.0, 1.8)],
        vec![
            culvert(
                "culvert",
                ["upper", "lower"],
                (0.3, 12.0),
                [0.05, 0.0],
                true,
            )?,
            road("road", ["upper", "lower"], 1.5)?,
        ],
    )
}
