//! Reach-end port intervals through the public trial: rest, dry beds, drains, flux regimes,
//! pairing, and reversal.

use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, BETA, Batch, CELLS, GRAVITY, REACH_WIDTH, Reach, Scenario, balanced, east, last,
    reach_cells, step, still, west,
};
use hydr::coupling::{SurfaceChannelPortRegime, SurfaceChannelPortTransfer};
use hydr::model::CellState;

/// Requested duration of every test interval, in s. The CFL limit shortens it.
const DT: f64 = 0.1;

/// Accepted intervals of the seiche run.
const SEICHE_INTERVALS: usize = 120;

/// The only port receipt of the last interval.
fn transfer(batch: &Batch) -> Result<SurfaceChannelPortTransfer> {
    Ok(last(batch)?
        .ports
        .first()
        .ok_or("the port receipt")?
        .clone())
}

#[test]
fn a_lake_at_rest_through_ports_with_a_bed_step_stays_at_rest() -> Result {
    // Frozen criterion: after 50 intervals every |q|, |Q| and stage change is at most 1e-12.
    const TOLERANCE: f64 = 1e-12;
    let stage = 1.6;
    // The reach bed lies below, then above, the 1 m surface bed, under the 1.2 m bank crest.
    for alpha_bed in [0.5, 1.125] {
        let ports = vec![west("p-west"), east("p-east")];
        let mut batch = Scenario::rest(1.0, stage, alpha_bed, ports).build()?;
        for _ in 0..50 {
            batch = step(&batch, DT)?;
            let interval = last(&batch)?;
            assert_eq!(interval.ports.len(), 2);
            for transfer in &interval.ports {
                assert_eq!(transfer.regime, SurfaceChannelPortRegime::Subcritical);
                assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
            }
        }
        assert_eq!(batch.accepted_steps(), 50);
        for cell in batch.surface().state()?.cells() {
            assert!(
                cell.qx.abs() <= TOLERANCE && cell.qy.abs() <= TOLERANCE,
                "{cell:?}"
            );
            assert!((cell.h + 1.0 - stage).abs() <= TOLERANCE, "{cell:?}");
        }
        for id in [ALPHA, BETA] {
            for cell in reach_cells(&batch, id)? {
                assert!(cell.discharge_m3_per_s.abs() <= TOLERANCE, "{id}: {cell:?}");
                let moved = (cell.stage_m.ok_or("the reach is wet")? - stage).abs();
                assert!(moved <= TOLERANCE, "{id}: {cell:?}");
            }
        }
        balanced(&batch)?;
    }
    Ok(())
}

#[test]
fn water_below_the_higher_bed_moves_nothing() -> Result {
    // The reach stage of 0.8 m lies below the 1 m surface bed, and the surface is dry.
    let mut case = Scenario::rest(1.0, 0.8, 0.5, vec![west("p")]);
    case.link = false;
    let batch = step(&case.build()?, DT)?;
    let transfer = transfer(&batch)?;
    assert_eq!(transfer.regime, SurfaceChannelPortRegime::Dry);
    assert_eq!(transfer.signed_volume_m3.to_bits(), 0.0f64.to_bits());
    assert_eq!(
        batch.surface().ledger().exchange_in_m3.to_bits(),
        0.0f64.to_bits()
    );
    Ok(())
}

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

#[test]
fn a_draining_end_cell_shortens_the_attempt_and_keeps_its_water_nonnegative() -> Result {
    // Only the reach end cell holds water, and the floodplain is dry.
    let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
    case.link = false;
    case.alpha = Reach {
        bed: 1.0,
        depth: [0.05, 0.0, 0.0],
        discharge_per_width: [0.0; 3],
    };
    let start = case.build()?;
    let held = reach_cells(&start, ALPHA)?[0].volume_m3;
    let mut batch = step(&start, 1.0)?;
    assert!(
        last(&batch)?.retries >= 1,
        "the requested second must shorten"
    );
    // The first interval drains across the front. Later the floodplain cell is wet.
    assert_eq!(
        transfer(&batch)?.regime,
        SurfaceChannelPortRegime::WetDryFront
    );
    let mut drained = 0.0;
    for _ in 0..10 {
        drained -= transfer(&batch)?.signed_volume_m3;
        let reach = batch.reach(ALPHA).ok_or("alpha")?;
        let limits = reach.last_step()?.ok_or("a reach receipt")?;
        // Port export enters the face depletion limit, which covers the interval.
        let dt = limits.end_s - limits.start_s;
        assert!(limits.depletion_limit_s.is_finite() && dt <= limits.depletion_limit_s);
        assert!(dt <= limits.cfl_limit_s);
        assert!(
            reach_cells(&batch, ALPHA)?
                .iter()
                .all(|c| c.volume_m3 >= 0.0)
        );
        assert!(batch.surface().state()?.cells().iter().all(|c| c.h >= 0.0));
        balanced(&batch)?;
        batch = step(&batch, 1.0)?;
    }
    assert!(drained > 0.0 && drained <= held);
    Ok(())
}

#[test]
fn a_dam_break_onto_a_dry_floodplain_runs_and_balances() -> Result {
    // Ritter smoke case: a still reach releases onto a dry floodplain through its port.
    let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
    case.link = false;
    case.alpha = Reach::uniform(1.0, 0.5, 0.0);
    let mut batch = case.build()?;
    let held: f64 = reach_cells(&batch, ALPHA)?
        .iter()
        .map(|c| c.volume_m3)
        .sum();
    let mut released = 0.0;
    for index in 0..20 {
        batch = step(&batch, DT)?;
        let transfer = transfer(&batch)?;
        if index == 0 {
            assert_eq!(transfer.regime, SurfaceChannelPortRegime::WetDryFront);
        }
        assert!(transfer.signed_volume_m3 <= 0.0);
        released -= transfer.signed_volume_m3;
        assert!(
            reach_cells(&batch, ALPHA)?
                .iter()
                .all(|c| c.volume_m3 >= 0.0)
        );
        assert!(batch.surface().state()?.cells().iter().all(|c| c.h >= 0.0));
        balanced(&batch)?;
    }
    assert!(released > 0.0 && released < held);
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

fn relative(actual: f64, expected: f64) -> f64 {
    (actual - expected).abs() / expected.abs()
}

#[test]
fn uniform_supercritical_flow_takes_the_upwind_flux_both_ways() -> Result {
    // Frozen criterion: the reach state's own flux, relative change at most 1e-12.
    const TOLERANCE: f64 = 1e-12;
    let h = 0.5;
    let q = h * 2.0 * (GRAVITY * h).sqrt();
    let flux = REACH_WIDTH * q;
    let momentum = REACH_WIDTH * (q * q / h + 0.5 * GRAVITY * h * h);
    // Into the surface is -x at the east edge. The reach flows toward its right end.
    let out = east_case(Reach::uniform(1.0, h, q), moving(h, -q));
    let (batch, transfer) = east_step(&out, 0.01)?;
    let dt = batch.time();
    assert_eq!(
        transfer.regime,
        SurfaceChannelPortRegime::SupercriticalToSurface
    );
    assert!(relative(-transfer.signed_volume_m3, flux * dt) <= TOLERANCE);
    assert!(relative(transfer.reach_momentum_impulse_m4_per_s, momentum * dt) <= TOLERANCE);
    // The same state running into the reach takes the surface state's own flux.
    let into = east_case(Reach::uniform(1.0, h, -q), moving(h, q));
    let (batch, transfer) = east_step(&into, 0.01)?;
    let dt = batch.time();
    assert_eq!(
        transfer.regime,
        SurfaceChannelPortRegime::SupercriticalToReach
    );
    assert!(relative(transfer.signed_volume_m3, flux * dt) <= TOLERANCE);
    assert!(relative(transfer.reach_momentum_impulse_m4_per_s, momentum * dt) <= TOLERANCE);
    Ok(())
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

/// The reach starts above the surface. Water runs out through the port and the bank link,
/// then back.
///
/// Every interval checks the step mass bound and the lifetime composite balance.
/// Returns the port receipts of every interval and the last batch.
pub fn seiche(intervals: usize) -> Result<(Vec<SurfaceChannelPortTransfer>, Batch)> {
    let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    case.alpha = Reach::rest(0.5, 1.75);
    let mut batch = case.build()?;
    let mut run = Vec::with_capacity(intervals);
    for _ in 0..intervals {
        batch = step(&batch, 0.25)?;
        let diagnostics = last(&batch)?;
        assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
        balanced(&batch)?;
        run.push(transfer(&batch)?);
    }
    Ok((run, batch))
}

#[test]
fn pairing_and_composite_balance_hold_through_inflow_and_outflow() -> Result {
    let (run, last) = seiche(SEICHE_INTERVALS)?;
    let (mut inflow, mut outflow) = (false, false);
    for transfer in &run {
        assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
        inflow |= transfer.signed_volume_m3 > 0.0;
        outflow |= transfer.signed_volume_m3 < 0.0;
    }
    assert!(inflow && outflow, "the seiche must run both ways");
    let ledger = &last.port_ledgers()?[0];
    assert!(ledger.surface_to_channel_m3 > 0.0 && ledger.channel_to_surface_m3 > 0.0);
    Ok(())
}

#[test]
fn a_reversal_sets_the_reversed_flag() -> Result {
    let (run, _) = seiche(SEICHE_INTERVALS)?;
    let mut previous: Option<f64> = None;
    let mut reversals = 0;
    for transfer in &run {
        let signed = transfer.signed_volume_m3;
        let flipped =
            previous.is_some_and(|p| (p > 0.0 && signed < 0.0) || (p < 0.0 && signed > 0.0));
        assert_eq!(transfer.reversed, flipped);
        reversals += usize::from(flipped);
        previous = Some(signed);
    }
    assert!(reversals >= 1, "the seiche must reverse the port");
    Ok(())
}
