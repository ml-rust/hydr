//! Reach-end port intervals through the public trial: port-only setup and flux regimes.

use crate::fixture::Result;
use crate::ports_fixture::{
    Batch, CELLS, Reach, Scenario, balanced, east, last, step, still, transfer, west,
};
use hydr::coupling::{SurfaceChannelPortRegime, SurfaceChannelPortTransfer};
use hydr::model::CellState;

/// Requested duration of every test interval, in s. The CFL limit shortens it.
const DT: f64 = 0.1;

#[test]
fn a_port_only_coordinator_builds_and_steps() -> Result {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    case.link = false;
    case.alpha = Reach::rest(0.5, 1.8);
    let mut batch = case.build()?;
    assert!(batch.definition().surface_cells().is_empty());
    for _ in 0..5 {
        batch = step(&batch, DT)?;
        // The higher reach drains onto the surface.
        assert!(transfer(&batch)?.signed_volume_m3 < 0.0);
        balanced(&batch)?;
    }
    let diagnostics = last(&batch)?;
    assert!(diagnostics.links.is_empty());
    assert_eq!(diagnostics.iterations, 0);
    assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
    assert!(diagnostics.port_paired_bound_m3 > 0.0);
    assert!(batch.link_ledgers()?.is_empty());
    let ledgers = batch.port_ledgers()?;
    assert_eq!(ledgers.len(), 1);
    assert_eq!(ledgers[0].port_id, "p");
    assert!(ledgers[0].channel_to_surface_m3 > 0.0 && ledgers[0].surface_to_channel_m3 == 0.0);
    // The surface ledger counts port water as exchange. The composite link totals leave it out.
    assert!(batch.surface().ledger().exchange_in_m3 > 0.0);
    assert!(batch.ledger()?.surface_exchange_in_m3.abs() <= 1e-12);
    Ok(())
}

/// Every surface cell at depth `h` with x discharge per width `qx`.
fn moving(h: f64, qx: f64) -> Vec<CellState> {
    vec![CellState { h, qx, qy: 0.0 }; CELLS]
}

/// `alpha` in state `alpha` against `surface` through the east port, on a flat 1 m bed.
fn east_case(alpha: Reach, surface: Vec<CellState>) -> Scenario {
    let mut case = Scenario::rest(1.0, 1.5, 1.0, vec![east("p")]);
    case.surface = surface;
    case.alpha = alpha;
    case.link = false;
    case
}

/// One interval toward `dt`, with the batch and its port receipt.
fn east_step(case: &Scenario, dt: f64) -> Result<(Batch, SurfaceChannelPortTransfer)> {
    let batch = step(&case.build()?, dt)?;
    let transfer = transfer(&batch)?;
    Ok((batch, transfer))
}

#[test]
fn a_sonic_point_and_a_dry_surface_take_their_regimes() -> Result {
    // A subcritical reach end beside fast shallow surface water: the 1-fan is sonic.
    let sonic = east_case(Reach::uniform(1.0, 1.0, 1.0), moving(0.1, -0.2));
    assert_eq!(
        east_step(&sonic, 0.01)?.1.regime,
        SurfaceChannelPortRegime::Transcritical
    );
    // A wet reach end against a dry surface drains across a wet/dry front.
    let front = east_case(Reach::uniform(1.0, 0.5, 0.0), vec![still(0.0); CELLS]);
    let transfer = east_step(&front, 0.01)?.1;
    assert_eq!(transfer.regime, SurfaceChannelPortRegime::WetDryFront);
    assert!(transfer.signed_volume_m3 < 0.0);
    Ok(())
}
