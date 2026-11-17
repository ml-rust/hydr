//! Runs the native worker lifecycle and prints each labelled result.
//!
//! Steps: capability check, cancellation, checkpoint, restore, forecast cycle, and water budget.
//! The results are a numerical demonstration on a synthetic basin.

#[allow(dead_code)]
#[path = "../tests/support/runtime.rs"]
mod runtime;

#[path = "../tests/support/impervious.rs"]
mod impervious;

#[path = "native_worker/mod.rs"]
mod worker;

use worker::{RunRecord, WorkerReport};

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn verdict(left: &RunRecord, right: &RunRecord) -> &'static str {
    if left.same_bits(right) {
        "bitwise equal"
    } else {
        "DIFFERENT"
    }
}

fn print(report: &WorkerReport) {
    let caps = &report.capabilities;
    println!("Label: numerical demonstration");
    println!(
        "Capabilities: admitted boundaries {:?}, sources {:?}",
        caps.boundaries, caps.sources
    );
    println!(
        "Capabilities: cpu {}, contract {}, schema {}",
        caps.cpu_status, caps.contract, caps.schema
    );
    println!("Capabilities: limit accepted: {}", caps.port_limit);

    println!(
        "Cancel: status {:?} at {} s after {} accepted steps",
        report.cancelled.status, report.cancelled.accepted_time, report.cancelled.accepted_steps
    );
    if let Some(last) = report
        .records
        .get(report.cancelled.accepted_steps.saturating_sub(1))
    {
        println!(
            "Observer: step {} at {} s, step residual {:e} m³, max depth {} m",
            last.step, last.time, last.residual, last.max_depth
        );
    }
    println!(
        "Resume: status {:?} at {} s after {} more steps, {} records",
        report.resumed.status,
        report.resumed.accepted_time,
        report.resumed.accepted_steps,
        report.records.len()
    );

    let package = &report.cancel_package;
    println!(
        "Checkpoint at cancel: schema {}, contract {}, time {} s, steps {}",
        package.schema, package.contract, package.accepted.time, package.accepted_steps
    );
    let boundary = &report.worker.data;
    let prefix = &boundary.forcing_prefix;
    println!(
        "Checkpoint at {} s: forcing prefix {} intervals [{}, {}) s, digest {}",
        boundary.accepted.time,
        prefix.intervals,
        prefix.start,
        prefix.end,
        hex(&prefix.digest)
    );
    println!(
        "Checkpoint at {} s: window {} intervals",
        boundary.accepted.time,
        boundary.setup.forcing.intervals().len()
    );

    for (label, record) in [
        ("cancelled then resumed", &report.worker),
        ("restore_data", &report.restored),
        ("Solver::resume", &report.resumed_package),
    ] {
        println!(
            "Restart ({label}) vs uninterrupted: {}",
            verdict(record, &report.uninterrupted)
        );
    }
    for (label, record) in &report.forecasts {
        println!(
            "Forecast ({label}) vs single spliced run: {}",
            verdict(record, &report.single_forecast)
        );
    }

    let budget = &report.budget;
    println!(
        "Water budget: storage change {:e} m³",
        budget.storage_change
    );
    println!(
        "Water budget: inputs {:e} m³, outputs {:e} m³, adjustment {:e} m³",
        budget.inputs, budget.outputs, budget.adjustment
    );
    println!(
        "Water budget: residual {:e} m³, bound {:e} m³",
        budget.residual, budget.residual_bound
    );
    println!(
        "Water budget: adjustment budget used {:e} of {:e} m³",
        budget.adjustment_used, budget.adjustment_limit
    );
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let report = worker::run()?;
    print(&report);
    let restarts = [&report.worker, &report.restored, &report.resumed_package];
    if restarts
        .iter()
        .any(|record| !record.same_bits(&report.uninterrupted))
        || report
            .forecasts
            .iter()
            .any(|(_, record)| !record.same_bits(&report.single_forecast))
    {
        return Err(
            "a restarted or continued run differs from its reference: inspect the step partition"
                .into(),
        );
    }
    if report.budget.residual.abs() > report.budget.residual_bound {
        return Err("water residual exceeds its bound: inspect sources and port data".into());
    }
    Ok(())
}
