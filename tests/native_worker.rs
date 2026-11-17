//! Pins the native worker lifecycle that `examples/native_worker.rs` prints.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/impervious.rs"]
mod impervious;

#[path = "../examples/native_worker/mod.rs"]
mod worker;

use hydr::capability::{BoundaryKind, RuntimeStatus, SourceKind};
use hydr::solver::{AdvanceStatus, CONTRACT_VERSION, SNAPSHOT_SCHEMA};
use worker::{CANCEL_AFTER, EDGES};

#[test]
fn native_worker_restart_matches_uninterrupted_bits() -> Result<(), Box<dyn std::error::Error>> {
    let report = worker::run()?;

    let caps = &report.capabilities;
    assert_eq!(
        caps.boundaries,
        [BoundaryKind::Wall.id(), BoundaryKind::FluxPort.id()]
    );
    assert_eq!(
        caps.sources,
        [SourceKind::Rain.id(), SourceKind::ImperviousLand.id()]
    );
    assert_eq!(caps.cpu_status, RuntimeStatus::Reference.id());
    assert!(!caps.port_limit.is_empty());
    assert_eq!(caps.contract, CONTRACT_VERSION);
    assert_eq!(caps.schema, SNAPSHOT_SCHEMA);

    assert_eq!(report.cancelled.status, AdvanceStatus::Cancelled);
    assert_eq!(report.cancelled.accepted_steps, CANCEL_AFTER);
    assert!(report.cancelled.accepted_time > EDGES[0]);
    assert!(report.cancelled.accepted_time < EDGES[2]);
    assert_eq!(
        report.cancel_package.accepted.time.to_bits(),
        report.cancelled.accepted_time.to_bits()
    );
    assert_eq!(report.cancel_package.accepted_steps, CANCEL_AFTER);
    assert_eq!(report.cancel_package.schema, SNAPSHOT_SCHEMA);
    assert_eq!(report.cancel_package.contract, CONTRACT_VERSION);

    assert_eq!(report.resumed.status, AdvanceStatus::ReachedTarget);
    assert_eq!(report.resumed.accepted_time.to_bits(), EDGES[2].to_bits());
    let total = CANCEL_AFTER + report.resumed.accepted_steps;
    assert_eq!(report.records.len(), total);
    assert_eq!(report.worker.data.accepted_steps, total);
    for (index, record) in report.records.iter().enumerate() {
        assert_eq!(record.step, index + 1);
    }
    let last = report.records.last().ok_or("observer recorded no step")?;
    let last_step = report
        .worker
        .last_step
        .as_ref()
        .ok_or("worker has no last step")?;
    assert_eq!(last.time.to_bits(), last_step.end.to_bits());
    assert_eq!(last.residual.to_bits(), last_step.residual.to_bits());

    for record in [&report.worker, &report.restored, &report.resumed_package] {
        assert_eq!(record, &report.uninterrupted);
        assert!(record.same_bits(&report.uninterrupted));
    }
    let prefix = &report.worker.data.forcing_prefix;
    let window = report.worker.data.setup.forcing.intervals();
    assert_eq!(
        prefix.end.to_bits(),
        window.first().ok_or("empty window")?.start.to_bits()
    );

    assert_eq!(report.forecasts.len(), 5);
    for (label, record) in &report.forecasts {
        assert_eq!(record, &report.single_forecast, "{label}");
        assert!(record.same_bits(&report.single_forecast), "{label}");
    }
    assert_eq!(
        report.single_forecast.data.accepted.time.to_bits(),
        EDGES[3].to_bits()
    );

    let budget = &report.budget;
    assert!(budget.residual.abs() <= budget.residual_bound);
    assert!(budget.adjustment_used <= budget.adjustment_limit);
    assert!(budget.inputs > 0.0);
    Ok(())
}
