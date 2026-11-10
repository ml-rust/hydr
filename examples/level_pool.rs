//! Closed volume-authoritative storage with explicit original interval exchanges.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

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
    let batch = LevelPoolBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
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
    let enabled = LevelPoolBatch::<crate::runtime::Runtime>::new_with_sources(
        crate::runtime::client(),
        restored.elements().to_vec(),
        restored.states()?,
        restored.time(),
        restored.config().clone(),
        vec![hydr::storage::LevelPoolSourcePolicy {
            dry_basal_footprint_assertion: "the first segment is the tank's rain-receiving floor"
                .into(),
            bottom_loss_provenance:
                "explicit synthetic potential bottom demand, without soil memory".into(),
        }],
    )?;
    let sources = enabled.trial_with_sources_to(
        70.0,
        &[LevelPoolInput {
            incoming_volume_m3: 0.0,
            outgoing_volume_m3: 0.05,
        }],
        &[hydr::storage::LevelPoolSources {
            rain_rate_m_per_s: 0.01,
            potential_evaporation_rate_m_per_s: 0.002,
            bottom_loss_rate_m3_per_s: 0.001,
        }],
    )?;
    println!("source_candidate={:?}", sources.outputs()?);
    println!("source_receipts={:?}", sources.diagnostics()?.receipts);
    Ok(())
}
