//! Closed volume-authoritative storage with explicit original interval exchanges.

use hydr::storage::{
    LevelPoolBatch, LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolInput,
    LevelPoolKnot, LevelPoolState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 10.0,
            },
            LevelPoolKnot {
                volume_m3: 4.0,
                stage_m: 12.0,
            },
        ],
        "synthetic vertical datum".into(),
        "explicit constant-area tank, area 2 m2".into(),
    )?;
    let batch = LevelPoolBatch::new(
        vec![LevelPoolElement {
            id: "tank".into(),
            curve,
            connected_pool_assertion: "one connected tank with approximately level water".into(),
            storage_support_identity: "synthetic-tank-interior".into(),
        }],
        vec![LevelPoolState { volume_m3: 1.0 }],
        0.0,
        LevelPoolConfig::default(),
    )?;
    let trial = batch.trial_to(
        60.0,
        &[LevelPoolInput {
            incoming_volume_m3: 0.2,
            outgoing_volume_m3: 0.1,
        }],
    )?;
    let diagnostics = trial.diagnostics()?;
    println!("candidate={:?}", trial.outputs()?);
    println!(
        "original_incoming_m3={}, original_outgoing_m3={}, residual_m3={}",
        diagnostics.incoming_volume_m3,
        diagnostics.outgoing_volume_m3,
        diagnostics.water_residual_m3
    );
    let accepted = trial.into_next();
    println!(
        "accepted_time_s={}, totals={:?}",
        accepted.time(),
        accepted.totals()?
    );
    let mut restored = batch;
    restored.restore_data(&accepted.snapshot_data()?)?;
    println!("restored={:?}", restored.outputs()?);
    Ok(())
}
