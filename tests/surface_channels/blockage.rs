//! Surface/channel coordinator blockage checks: sediment bed stops, closed links, scenario sets,
//! geometry reports, and checkpoint restart with forged-condition rejection.

use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, culvert_law, manual, pond, pool, step, structure, transfer, weir_law,
};
use crate::runtime::{Runtime, client};
use hydr::coupling::{StructureScenario, SurfaceChannelBatch};
use hydr::structures::{
    BlockageFraction, CulvertBlockage, CulvertSediment, DirectionPolicy, StructureBlockage,
    StructureCondition, StructureConditionKind, StructureGeometry, StructureOperatingPoint,
    StructureRegime,
};

const DT: f64 = 0.5;
const ALLOWANCE: f64 = 1e-12;

/// A small pond drains through a sedimented culvert (invert 0.2, rise 0.6, sediment 0.24 -> bed
/// 0.44) into a dry basin, stopping at the bed stop and never below the bed elevation.
#[test]
fn a_draining_sedimented_culvert_stops_at_its_bed_elevation() -> Result {
    let rise = 0.6;
    let invert = 0.2;
    let open = rise - 0.24_f64;
    let admitted_depth = rise - open;
    let bed = invert + admitted_depth;
    let area = 0.01;
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("basin", 0.0, 40.0, 3.0)?, 0.0),
        (pond("pond", 0.0, area, 3.0)?, (bed + 0.1) * area),
    ];
    let mut link = structure(
        "culvert",
        culvert_law(rise, 4.0, [invert, 0.0], false)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("pond"), 0.0),
        (pool("basin"), 0.0),
    );
    link.element.condition = StructureCondition::Blocked(StructureBlockage::culvert(
        CulvertBlockage {
            face_a: BlockageFraction::NONE,
            face_b: BlockageFraction::NONE,
            sediment: Some(CulvertSediment {
                depth_m: 0.24,
                bed_manning_n: 0.030,
            }),
            closed_barrels: 0,
        },
        "Sediment bed survey".into(),
    )?);
    case.structures = vec![link];
    let mut batch = case.build()?;
    let mut capped = false;
    let mut rested = false;
    for index in 0..3 {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
        let stage = pool_stage(&batch, "pond")?;
        assert!(
            stage >= bed,
            "interval {index}: pond stage={stage:e} < {bed:e}"
        );
        let link = transfer(&batch, "culvert")?;
        if capped {
            assert_eq!(link.signed_volume_m3.to_bits(), 0.0_f64.to_bits());
            assert_eq!(link.cap_factor.to_bits(), 0.0_f64.to_bits());
            rested = true;
        }
        capped |= link.cap_factor < 1.0;
    }
    assert!(capped, "the bed cap never bound");
    assert!(rested, "no interval followed the binding cap");
    let band = 2.0 * (ALLOWANCE + ALLOWANCE * bed);
    assert!(pool_stage(&batch, "pond")? <= bed + band);
    Ok(())
}

/// A `Closed` weir between two ponds at different stages leaves both volumes bit-identical and
/// writes zero ledger volume.
#[test]
fn closed_weir_leaves_pool_volumes_bit_identical_with_zero_ledger() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("east", 0.0, 10.0, 3.0)?, 6.0),
        (pond("west", 0.0, 10.0, 3.0)?, 18.0),
    ];
    let mut link = structure(
        "weir",
        weir_law(0.5, 2.0)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("west"), 0.0),
        (pool("east"), 0.0),
    );
    link.element.condition = StructureCondition::Closed {
        evidence: "Stop logs installed".into(),
    };
    case.structures = vec![link];
    let mut batch = case.build()?;
    let initial_volumes: Vec<u64> = batch
        .pools()
        .ok_or("pools")?
        .states()?
        .iter()
        .map(|s| s.volume_m3.to_bits())
        .collect();
    for _ in 0..4 {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
        let link = transfer(&batch, "weir")?;
        assert_eq!(link.regime, StructureRegime::Closed);
        assert_eq!(link.signed_volume_m3.to_bits(), 0.0_f64.to_bits());
    }
    let final_volumes: Vec<u64> = batch
        .pools()
        .ok_or("pools")?
        .states()?
        .iter()
        .map(|s| s.volume_m3.to_bits())
        .collect();
    assert_eq!(final_volumes, initial_volumes);
    let ledgers = batch.structure_ledgers()?;
    assert_eq!(ledgers[0].a_to_b_m3.to_bits(), 0.0_f64.to_bits());
    assert_eq!(ledgers[0].b_to_a_m3.to_bits(), 0.0_f64.to_bits());
    let report = batch.structure_geometry();
    assert_eq!(report.len(), 1);
    assert_eq!(report[0].condition, StructureConditionKind::Closed);
    assert_eq!(report[0].effective, None);
    Ok(())
}

/// One batch per scenario (as built, face 50%, sediment 0.4 D, Closed), each reporting its own
/// geometry and passing mass checks independently.
#[test]
fn scenario_set_runs_independently_and_reports_structure_geometry() -> Result {
    let rise = 0.6;
    let scenarios = [
        (
            "as-built",
            StructureCondition::AsBuilt,
            StructureConditionKind::AsBuilt,
        ),
        (
            "face-50",
            StructureCondition::Blocked(StructureBlockage::culvert(
                CulvertBlockage {
                    face_a: BlockageFraction::new(0.5)?,
                    ..CulvertBlockage::NONE
                },
                "Half-blocked trash rack".into(),
            )?),
            StructureConditionKind::Blocked,
        ),
        (
            "sediment-40",
            StructureCondition::Blocked(StructureBlockage::culvert(
                CulvertBlockage {
                    sediment: Some(CulvertSediment {
                        depth_m: 0.4 * rise,
                        bed_manning_n: 0.030,
                    }),
                    ..CulvertBlockage::NONE
                },
                "40% sediment fill".into(),
            )?),
            StructureConditionKind::Blocked,
        ),
        (
            "closed",
            StructureCondition::Closed {
                evidence: "Bulkhead".into(),
            },
            StructureConditionKind::Closed,
        ),
    ];
    for (id, condition, kind) in scenarios {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("lower", 0.0, 20.0, 3.0)?, 4.0),
            (pond("upper", 0.0, 20.0, 3.0)?, 24.0),
        ];
        let mut link = structure(
            "culvert",
            culvert_law(rise, 10.0, [0.1, 0.0], false)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        );
        link.scenario = StructureScenario {
            id: id.into(),
            evidence: format!("Scenario {id}"),
        };
        link.element.condition = condition;
        case.structures = vec![link];
        let mut batch = case.build()?;
        let reports = batch.structure_geometry();
        assert_eq!(reports.len(), 1);
        assert_eq!(reports[0].condition, kind);
        if kind == StructureConditionKind::Closed {
            assert_eq!(reports[0].effective, None);
        } else {
            assert!(matches!(
                reports[0].effective,
                Some(StructureGeometry::Culvert { .. })
            ));
        }
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
    }
    Ok(())
}

/// Checkpoint, restore, import, and resume bit for bit with blocked and closed structures, and
/// reject a forged condition at import.
#[test]
fn blocked_and_closed_structures_restart_bit_for_bit_and_reject_forged_condition() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 20.0, 3.0)?, 4.0),
        (pond("upper", 0.0, 20.0, 3.0)?, 24.0),
    ];
    let mut blocked = structure(
        "culvert",
        culvert_law(0.6, 12.0, [0.1, 0.0], false)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    );
    blocked.element.condition = StructureCondition::Blocked(StructureBlockage::culvert(
        CulvertBlockage {
            face_a: BlockageFraction::new(0.25)?,
            face_b: BlockageFraction::NONE,
            sediment: Some(CulvertSediment {
                depth_m: 0.15,
                bed_manning_n: 0.030,
            }),
            closed_barrels: 0,
        },
        "Survey".into(),
    )?);
    let mut shut = structure(
        "weir",
        weir_law(0.5, 2.0)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    );
    shut.element.condition = StructureCondition::Closed {
        evidence: "Stop logs".into(),
    };
    case.structures = vec![blocked, shut];

    let origin = case.build()?;
    let mid = step(&origin, DT, &PoolForcing::default())?;
    let snapshot = mid.snapshot();
    let package = mid.snapshot_data()?;
    let direct = step(&mid, DT, &PoolForcing::default())?.snapshot_data()?;

    let restored = case.build()?.restore(&snapshot)?;
    let after_restore = step(&restored, DT, &PoolForcing::default())?.snapshot_data()?;
    assert_eq!(format!("{after_restore:?}"), format!("{direct:?}"));

    let destination = case.build()?;
    let imported = destination.import_snapshot(&package)?;
    let after_import = step(&imported, DT, &PoolForcing::default())?.snapshot_data()?;
    assert_eq!(format!("{after_import:?}"), format!("{direct:?}"));

    let resumed = SurfaceChannelBatch::<Runtime>::resume(
        client(),
        &package,
        origin.setup().surface.forcing.clone(),
    )?;
    let after_resume = step(&resumed, DT, &PoolForcing::default())?.snapshot_data()?;
    assert_eq!(format!("{after_resume:?}"), format!("{direct:?}"));

    let mut forged = package;
    forged.setup.structures[0].element.condition = StructureCondition::AsBuilt;
    let err = destination
        .import_snapshot(&forged)
        .err()
        .ok_or("forged condition was admitted")?;
    assert!(!err.to_string().is_empty());
    Ok(())
}
