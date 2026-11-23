//! Coupled level pool blockage checks: zero-fraction bit identity, closed links, geometry
//! reports, and checkpoint restart with forged-condition rejection.

use crate::culverts_fixture::blocked_culvert;
use crate::fixture::{Result, orifice_case};
use crate::runtime::{Runtime, client};
use hydr::coupling::CoupledLevelPoolBatch;
use hydr::structures::{
    BlockageFraction, StructureBlockage, StructureCondition, StructureConditionKind,
    StructureGeometry,
};

#[test]
fn zero_fraction_blockage_matches_as_built_stripped_package_text() -> Result {
    let (as_built_setup, initial) = orifice_case([4.0, 2.0])?;
    let mut zero_setup = as_built_setup.clone();
    zero_setup.structures[0].condition = StructureCondition::Blocked(StructureBlockage::orifice(
        BlockageFraction::NONE,
        "Zero blockage".into(),
    )?);

    let a = CoupledLevelPoolBatch::<Runtime>::new(client(), as_built_setup, initial.clone())?
        .trial_to(0.5)?
        .into_next()
        .snapshot_data()?;
    let b = CoupledLevelPoolBatch::<Runtime>::new(client(), zero_setup, initial)?
        .trial_to(0.5)?
        .into_next()
        .snapshot_data()?;

    let strip = |text: String| -> String {
        text.replace(
            "condition: Blocked(StructureBlockage { geometry: Orifice { area: BlockageFraction(0.0) }, applicability_evidence: \"Zero blockage\" })",
            "condition: AsBuilt",
        )
    };
    assert_eq!(strip(format!("{b:?}")), format!("{a:?}"));
    Ok(())
}

#[test]
fn closed_structure_leaves_pool_volumes_bit_identical_and_reports_none() -> Result {
    let (mut setup, initial) = orifice_case([4.0, 2.0])?;
    setup.structures[0].condition = StructureCondition::Closed {
        evidence: "Gate sealed".into(),
    };
    let mut batch = CoupledLevelPoolBatch::<Runtime>::new(client(), setup, initial)?;
    let reports = batch.structure_geometry();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].condition, StructureConditionKind::Closed);
    assert_eq!(reports[0].effective, None);

    let initial_bits: Vec<u64> = batch
        .pools()
        .states()?
        .iter()
        .map(|s| s.volume_m3.to_bits())
        .collect();
    for step in 1..=4 {
        batch = batch.trial_to(0.25 * f64::from(step))?.into_next();
        let diag = batch.last_step()?.ok_or("missing last step")?;
        assert_eq!(
            diag.transfers[0].signed_volume_m3.to_bits(),
            0.0_f64.to_bits()
        );
    }
    let final_bits: Vec<u64> = batch
        .pools()
        .states()?
        .iter()
        .map(|s| s.volume_m3.to_bits())
        .collect();
    assert_eq!(final_bits, initial_bits);
    Ok(())
}

#[test]
fn blocked_culvert_reports_geometry_restarts_bit_for_bit_and_rejects_forged_condition() -> Result {
    let (setup, initial) = blocked_culvert()?;
    let origin = CoupledLevelPoolBatch::<Runtime>::new(client(), setup, initial)?;
    let reports = origin.structure_geometry();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].condition, StructureConditionKind::Blocked);
    assert!(matches!(
        reports[0].effective,
        Some(StructureGeometry::Culvert { .. })
    ));

    let mid = origin.trial_to(0.5)?.into_next();
    let snapshot = mid.snapshot();
    let package = mid.snapshot_data()?;
    let direct = mid.trial_to(1.0)?.into_next().snapshot_data()?;

    let mut restored = origin.clone();
    restored.restore(&snapshot)?;
    let after_restore = restored.trial_to(1.0)?.into_next().snapshot_data()?;
    assert_eq!(format!("{after_restore:?}"), format!("{direct:?}"));

    let mut imported = origin;
    imported.restore_data(&package)?;
    let after_import = imported.trial_to(1.0)?.into_next().snapshot_data()?;
    assert_eq!(format!("{after_import:?}"), format!("{direct:?}"));

    let mut forged = package;
    forged.setup.structures[0].condition = StructureCondition::AsBuilt;
    let err = imported
        .restore_data(&forged)
        .err()
        .ok_or("forged condition was admitted")?;
    assert!(!err.to_string().is_empty());
    Ok(())
}
