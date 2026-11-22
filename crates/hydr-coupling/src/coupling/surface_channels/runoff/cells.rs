//! Runoff into surface cells: delivery, spans, momentum, links, tailwater, and shortening.

use super::SurfaceChannelRunoffShare;
use super::attempt::split_bound;
use super::basin::{
    Basin, CELL_AREA, REACH, Receiver, advance, balanced, forcing, isolated, surface_volume,
    transfers,
};
use super::scenario::{input_message, share, starts};
use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult, client};
use crate::coupling::surface_channels::{SurfaceChannelBatch, SurfaceChannelError};

/// The default basin without the pond, draining into the surface `cells`.
fn surface(cells: Vec<SurfaceChannelRunoffShare>) -> Basin {
    let mut basin = Basin::into(Receiver::Surface(cells));
    basin.pond = false;
    basin
}

/// Bits of every surface cell's depth and discharges.
fn surface_bits(batch: &Batch) -> TestResult<Vec<[u64; 3]>> {
    Ok(batch
        .surface()
        .state()?
        .cells()
        .iter()
        .map(|cell| [cell.h, cell.qx, cell.qy].map(f64::to_bits))
        .collect())
}

/// Bits of every reach cell's volume and momentum.
fn reach_bits(batch: &Batch) -> TestResult<Vec<[u64; 2]>> {
    Ok(batch
        .reach(REACH)
        .ok_or("the reach")?
        .state()?
        .iter()
        .map(|cell| [cell.volume_m3, cell.longitudinal_momentum_m4_per_s].map(f64::to_bits))
        .collect())
}

/// Runoff into surface cell 1 of a coordinator without links.
///
/// Frozen criteria, fixed before the first run:
/// - The first delivered volume equals the runoff child trialled alone, bit for bit.
/// - Over four intervals the lane ledger holds the running sum of the delivered volumes bit
///   for bit, and the surface gains that sum within the composite bound.
/// - The surface exchange totals of each interval and of the lifetime hold no runoff: each
///   stays within its pairing or mass bound of zero.
/// - `runoff_delivered_m3` equals the ledger. Every interval balances.
#[test]
fn runoff_into_one_cell_steps_without_links() -> TestResult {
    let basin = surface(vec![share(1, 1.0)]);
    let mut batch = basin.build()?;
    assert!(batch.context.links.is_none() && batch.context.link_count == 0);
    let alone = isolated(&batch, 0.25)?;
    let mut total = 0.0;
    for step in 0..4 {
        batch = advance(&batch, 0.25, 0.0)?;
        let [transfer] = transfers(&batch)?.try_into().map_err(|_| "one lane")?;
        if step == 0 {
            assert_eq!(transfer.volume_m3.to_bits(), alone[0].to_bits());
        }
        assert!(transfer.volume_m3 > 0.0);
        total += transfer.volume_m3;
        let ledgers = batch.runoff_ledgers()?;
        assert_eq!(ledgers[0].delivered_volume_m3.to_bits(), total.to_bits());
        let diagnostics = batch.last_step()?.ok_or("an accepted interval")?;
        assert!(
            diagnostics.surface_exchange_in_m3.abs() <= diagnostics.paired_bound_m3,
            "{diagnostics:?}"
        );
        let balance = batch.ledger()?;
        assert_eq!(balance.runoff_delivered_m3.to_bits(), total.to_bits());
        assert!(balance.surface_exchange_in_m3.abs() <= balance.bound_m3);
        assert!((surface_volume(&batch)? - total).abs() <= balance.bound_m3);
        balanced(&batch)?;
    }
    Ok(())
}

/// One outlet spans surface cells 0 and 1 with weights 1 and 3.
///
/// Frozen criteria, fixed before the first run:
/// - The two parts sum to the export within the split bound and keep the 1:3 ratio within
///   four ulps.
/// - Each lane ledger holds its part after the first interval.
/// - Every interval of four balances, and the surface holds the delivered water.
#[test]
fn span_parts_on_cells_sum_to_their_export_within_the_split_bound() -> TestResult {
    let batch = surface(vec![share(0, 1.0), share(1, 3.0)]).build()?;
    let export = isolated(&batch, 0.25)?[0];
    let mut next = advance(&batch, 0.25, 0.0)?;
    let parts: Vec<f64> = transfers(&next)?
        .iter()
        .map(|transfer| transfer.volume_m3)
        .collect();
    let [low, high] = parts.as_slice() else {
        return Err("two delivery lanes".into());
    };
    assert!((low + high - export).abs() <= split_bound(2, export));
    assert!(
        (high / low - 3.0).abs() <= 4.0 * 3.0 * f64::EPSILON,
        "{parts:?}"
    );
    for (ledger, part) in next.runoff_ledgers()?.iter().zip(&parts) {
        assert_eq!(ledger.delivered_volume_m3.to_bits(), part.to_bits());
    }
    for _ in 0..3 {
        next = advance(&next, 0.25, 0.0)?;
        balanced(&next)?;
    }
    let delivered: f64 = next
        .runoff_ledgers()?
        .iter()
        .map(|ledger| ledger.delivered_volume_m3)
        .sum();
    assert!((surface_volume(&next)? - delivered).abs() <= next.ledger()?.bound_m3);
    Ok(())
}

/// Runoff into a moving surface cell against the same coordinator whose runoff exports
/// nothing.
///
/// Frozen criteria, fixed before the first run: after one interval every discharge bit of both
/// cells and the depth bits of cell 0 agree, and cell 1's depth rises by the delivered volume
/// over its area within two ulps of its depth.
#[test]
fn runoff_enters_a_moving_cell_without_momentum() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.surface_depth_m = 0.02;
    basin.surface_qx = 0.01;
    let wet = advance(&basin.build()?, 0.25, 0.0)?;
    basin.depth_m = 0.0;
    let dry = advance(&basin.build()?, 0.25, 0.0)?;
    let volume = transfers(&wet)?.first().ok_or("one lane")?.volume_m3;
    assert!(volume > 0.0);
    assert_eq!(transfers(&dry)?.first().ok_or("one lane")?.volume_m3, 0.0);
    let (wet_bits, dry_bits) = (surface_bits(&wet)?, surface_bits(&dry)?);
    assert_eq!(wet_bits[0], dry_bits[0]);
    assert_eq!(wet_bits[1][1..], dry_bits[1][1..]);
    let depth = |bits: &[[u64; 3]]| f64::from_bits(bits[1][0]);
    let rise = depth(&wet_bits) - depth(&dry_bits);
    assert!(
        (rise - volume / CELL_AREA).abs() <= 2.0 * f64::EPSILON * depth(&wet_bits),
        "rise={rise}, volume={volume}"
    );
    Ok(())
}

/// A bank weir drains the moving reach onto surface cell 1, which also takes runoff that
/// exports nothing.
///
/// Frozen criteria, fixed before the first run: over two intervals every surface, reach, and
/// link bit equals the same coordinator without runoff.
#[test]
fn zero_runoff_on_a_linked_cell_keeps_every_bit() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.bank = true;
    basin.depth_m = 0.0;
    basin.reach_momentum = 0.05;
    let (mut setup, mut initial) = basin.parts()?;
    setup.runoff = None;
    initial.runoff.clear();
    let mut plain = SurfaceChannelBatch::new(client(), setup, initial)?;
    let mut runoff = basin.build()?;
    for _ in 0..2 {
        runoff = advance(&runoff, 0.25, 0.0)?;
        let mut held = forcing(&plain, 0.25, 0.0);
        held.runoff.clear();
        plain = plain.trial_to(held.end_s, &held)?.into_next();
        assert_eq!(runoff.time().to_bits(), plain.time().to_bits());
        assert_eq!(surface_bits(&runoff)?, surface_bits(&plain)?);
        assert_eq!(reach_bits(&runoff)?, reach_bits(&plain)?);
        let links = |batch: &Batch| -> TestResult<Vec<u64>> {
            Ok(batch
                .last_step()?
                .ok_or("an accepted interval")?
                .links
                .iter()
                .map(|link| link.signed_volume_m3.to_bits())
                .collect())
        };
        assert_eq!(links(&runoff)?, links(&plain)?);
    }
    Ok(())
}

/// Runoff into surface cell 1 while a bank weir drains the moving reach onto it.
///
/// Frozen criteria, fixed before the first run: over four intervals the weir carries water
/// onto the surface, runoff arrives, the surface exchange totals stay within the pairing bound
/// of the link water, and every interval balances.
#[test]
fn runoff_into_a_linked_cell_pairs_with_the_link_water() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.bank = true;
    basin.reach_momentum = 0.05;
    let mut batch = basin.build()?;
    assert_eq!(batch.context.link_count, 1);
    for _ in 0..4 {
        batch = advance(&batch, 0.25, 0.0)?;
        let diagnostics = batch.last_step()?.ok_or("an accepted interval")?;
        let link = diagnostics.links.first().ok_or("the bank link")?;
        assert!(link.signed_volume_m3 < 0.0, "{link:?}");
        assert!(diagnostics.runoff.first().ok_or("one lane")?.volume_m3 > 0.0);
        let gap = diagnostics.surface_exchange_in_m3 + link.signed_volume_m3;
        assert!(gap.abs() <= diagnostics.paired_bound_m3, "{diagnostics:?}");
        balanced(&batch)?;
    }
    Ok(())
}

#[test]
fn a_submerged_cell_at_the_start_rejects_without_retry() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.surface_depth_m = 0.5;
    basin.invert_m = 0.25;
    let batch = basin.build()?;
    let message = input_message(batch.trial_to(0.5, &forcing(&batch, 0.5, 0.0)))?;
    starts(&message, "runoff-outlet-submerged");
    assert!(message.contains("surface-cell:1"), "{message}");
    Ok(())
}

/// The surface starts dry below an invert that half an interval of runoff crosses.
///
/// Frozen criteria, fixed before the first run:
/// - The trial shortens below the requested end, and the accepted end stage lies at or below
///   the invert.
/// - The delivered volume equals the runoff child trialled alone to the accepted end, bit for
///   bit.
/// - Without retries the same trial is `Unresolved`, naming `runoff-outlet-submerged`.
#[test]
fn a_cell_crossing_the_invert_at_the_end_shortens_the_interval() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.invert_m = 3e-4;
    let batch = basin.build()?;
    let next = advance(&batch, 0.5, 0.0)?;
    let end = next.time();
    assert!(end < 0.5, "end={end}");
    let transfer = transfers(&next)?
        .into_iter()
        .next()
        .ok_or("one delivery lane")?;
    assert!(transfer.end_stage_m <= transfer.invert_elevation_m);
    assert_eq!(
        transfer.volume_m3.to_bits(),
        isolated(&batch, end)?[0].to_bits()
    );
    balanced(&next)?;
    basin.retry_limit = 0;
    let batch = basin.build()?;
    match batch.trial_to(0.5, &forcing(&batch, 0.5, 0.0)) {
        Err(SurfaceChannelError::Unresolved { reason, .. }) => {
            assert!(reason.contains("runoff-outlet-submerged"), "{reason}");
            Ok(())
        }
        other => Err(format!("expected an unresolved trial, got {other:?}").into()),
    }
}

/// Deep surface water limits the interval by its CFL number, after the runoff admits its
/// rates.
///
/// Frozen criteria, fixed before the first run:
/// - The accepted end lies before the requested end.
/// - The delivered volume equals the runoff child trialled alone to the accepted end, bit for
///   bit, and differs from the full-interval volume scaled by the duration ratio.
#[test]
fn a_shortened_attempt_recomputes_surface_runoff() -> TestResult {
    let mut basin = surface(vec![share(1, 1.0)]);
    basin.surface_depth_m = 2.0;
    let batch = basin.build()?;
    let next = advance(&batch, 1.0, 0.0)?;
    let end = next.time();
    assert!(end < 1.0, "end={end}");
    let volume = transfers(&next)?.first().ok_or("one lane")?.volume_m3;
    assert_eq!(volume.to_bits(), isolated(&batch, end)?[0].to_bits());
    let scaled = isolated(&batch, 1.0)?[0] * end;
    assert!(
        (scaled - volume).abs() > 1e-3 * volume,
        "scaled={scaled}, delivered={volume}"
    );
    balanced(&next)
}
