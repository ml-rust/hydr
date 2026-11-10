//! Advances explicit instantaneous overflow and reports conservative interval volumes.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

use hydr::runoff::{
    ReservoirBatch, ReservoirConfig, ReservoirElement, ReservoirInput, ReservoirParameters,
    ReservoirState,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parameters =
        ReservoirParameters::from_native(1024.0, 1024.0, 32.0, 1.0 / 64.0, 0.0, 0.125)?;
    let accepted = ReservoirBatch::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        vec![ReservoirElement {
            id: "storage-a".into(),
            parameters,
        }],
        vec![ReservoirState { depth_m: 0.0 }],
        0.0,
        ReservoirConfig::default(),
    )?;
    let trial = accepted.trial_to(
        256.0,
        &[ReservoirInput {
            incoming_rate_m_per_s: 0.3048 / 1024.0,
        }],
    )?;
    let output =
        trial.outputs()?.into_iter().next().ok_or(
            "reservoir output missing for storage-a: require one output per declared element",
        )?;
    let next = trial.into_next();
    let ledger = next.ledger();
    println!("Accepted time: {} s", next.time());
    println!("Incoming volume: {} m³", ledger.incoming_volume_m3);
    println!("Storage: {} m³", ledger.storage_m3);
    println!("Export volume: {} m³", ledger.export_volume_m3);
    println!("Water residual: {} m³", ledger.residual_m3);
    println!(
        "Endpoint discharge (m³/s): {:?}",
        output.endpoint_discharge_m3_per_s
    );
    Ok(())
}
