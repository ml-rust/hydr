use hydr::coupling::{
    CoupledLevelPoolConfig, CoupledLevelPoolInitialState, CoupledLevelPoolSetup,
    CoupledPoolAttachment, CoupledPoolConnection, CoupledPoolSupport,
};
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};
use hydr::structures::{
    DirectionPolicy, OrificeApplicability, StructureElement, StructureLaw, StructureOperatingPoint,
    StructureOperation, SubmergedOrificeParameters,
};

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub const DATUM: &str = "Synthetic common vertical datum";
pub const VOLUME_TOLERANCE: f64 = 4e-10;

pub fn config() -> CoupledLevelPoolConfig {
    CoupledLevelPoolConfig {
        storage: LevelPoolConfig::default(),
        head_absolute_m: 1e-10,
        transfer_absolute_m3: 1e-10,
        mass_absolute_m3: 1e-9,
        mass_relative: 1e-12,
        minimum_step_s: 1e-8,
        maximum_step_s: 1.0,
        iteration_limit: 128,
        retry_limit: 16,
    }
}

pub fn assemble(
    pools: &[(&str, f64)],
    volumes: &[f64],
    structures: Vec<StructureElement>,
    operations: Vec<StructureOperation>,
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    let elements = pools
        .iter()
        .map(|&(id, origin)| {
            Ok(LevelPoolElement {
                id: id.into(),
                curve: LevelPoolCurve::new(
                    vec![
                        LevelPoolKnot {
                            volume_m3: 0.0,
                            stage_m: origin,
                        },
                        LevelPoolKnot {
                            volume_m3: 32.0,
                            stage_m: origin + 32.0,
                        },
                    ],
                    DATUM.into(),
                    "Exact synthetic constant area of one square metre".into(),
                )?,
                connected_pool_assertion: "Separate connected, approximately level synthetic pool"
                    .into(),
                storage_support_identity: format!("synthetic-storage-{id}"),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let attachment = |id: &str| -> Result<CoupledPoolAttachment> {
        let (_, origin) = pools
            .iter()
            .find(|(pool, _)| *pool == id)
            .ok_or("Fixture attachment has no pool")?;
        Ok(CoupledPoolAttachment {
            pool_id: id.into(),
            bed_elevation_m: *origin,
            physical_connection_assertion:
                "Declared unobstructed attachment to the named synthetic pool".into(),
        })
    };
    let connections = structures
        .iter()
        .map(|element| {
            Ok(CoupledPoolConnection {
                structure_id: element.id.clone(),
                endpoint_a: attachment(&element.endpoint_a)?,
                endpoint_b: attachment(&element.endpoint_b)?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((
        CoupledLevelPoolSetup {
            source_policies: None,
            external_inflows: vec![],
            pools: elements,
            pool_support: pools
                .iter()
                .map(|&(id, origin)| CoupledPoolSupport {
                    pool_id: id.into(),
                    zero_volume_stage_m: origin,
                    zero_volume_limit_assertion:
                        "Explicit physical dry limit of the synthetic constant-area basin".into(),
                })
                .collect(),
            structures,
            operations,
            connections,
            config: config(),
        },
        CoupledLevelPoolInitialState {
            time_s: 0.0,
            states: volumes
                .iter()
                .map(|&volume_m3| LevelPoolState { volume_m3 })
                .collect(),
        },
    ))
}

pub fn orifice_case(
    volumes: [f64; 2],
) -> Result<(CoupledLevelPoolSetup, CoupledLevelPoolInitialState)> {
    let element = StructureElement {
        id: "opening".into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
            1.0,
            0.0,
            1.0,
            1.0,
            0.5,
            OrificeApplicability {
                coefficient_provenance:
                    "Synthetic unit coefficient with explicitly artificial gravity 0.5 m/s²".into(),
                fully_submerged_applicability:
                    "Both synthetic stages remain strictly above the one-metre crown".into(),
                symmetric_reversal: true,
            },
        )?),
    };
    assemble(
        &[("a", 0.0), ("b", 0.0)],
        &volumes,
        vec![element],
        vec![StructureOperation {
            structure_id: "opening".into(),
            operating_point: StructureOperatingPoint::Passive,
        }],
    )
}

pub fn near(actual: f64, expected: f64) {
    assert!(actual.is_finite(), "Nonfinite value {actual}");
    assert!(
        (actual - expected).abs() <= VOLUME_TOLERANCE,
        "value={actual:.17e}, expected={expected:.17e}, tolerance={VOLUME_TOLERANCE:.17e}"
    );
}
