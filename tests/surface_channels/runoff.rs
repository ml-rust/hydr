//! Native runoff through the public coordinator API: interval receipts, balance totals, and
//! restart from a package taken mid-run.

use crate::pools_fixture::{Batch, Result};
use crate::runoff_fixture::{OUTLET, RECEIVERS, Receiver, forcing, parts};
use hydr::coupling::{SurfaceChannelNode, SurfaceChannelSnapshotData};

/// Requested interval, in s.
const DT: f64 = 0.25;
/// Rain on the subcatchment, in m/s.
const RAIN: f64 = 2e-5;

/// One accepted interval of `batch` under the fixture forcing.
fn advance(batch: &Batch) -> Result<Batch> {
    let end = batch.time() + DT;
    Ok(batch
        .trial_to(end, &forcing(batch.time(), end, RAIN))?
        .into_next())
}

/// Debug text of the package, the composite ledger, and the lane ledgers.
///
/// Debug text shows every bit of finite values, signed zero included.
fn state(batch: &Batch) -> Result<String> {
    Ok(format!(
        "{} {} {:?} {:?} {:?}",
        batch.time(),
        batch.accepted_steps(),
        batch.snapshot_data()?,
        batch.ledger()?,
        batch.runoff_ledgers()?
    ))
}

/// Four intervals into `receiver`.
///
/// Frozen criteria, fixed before the first run:
/// - Every interval reports one positive transfer to the receiver at its full length.
/// - The lane ledger holds the running sum of the transfers bit for bit.
/// - The balance reports the runoff child's storage bit for bit and the lane ledger's total
///   within four ulps, and it closes.
#[test]
fn runoff_reports_its_transfers_ledgers_and_balance_totals() -> Result {
    for receiver in RECEIVERS {
        let (setup, initial) = parts(receiver)?;
        let mut batch = Batch::new(crate::runtime::client(), setup, initial)?;
        let mut total = 0.0_f64;
        for _ in 0..4 {
            batch = advance(&batch)?;
            let step = batch.last_step()?.ok_or("an accepted interval")?;
            assert_eq!((step.end_s - step.start_s).to_bits(), DT.to_bits());
            let [transfer] = step.runoff.as_slice() else {
                return Err("one delivery lane".into());
            };
            assert_eq!(transfer.outlet_id, OUTLET);
            assert!(transfer.volume_m3 > 0.0, "{transfer:?}");
            let matched = match transfer.target {
                SurfaceChannelNode::Pool(_) => receiver == Receiver::Pool,
                SurfaceChannelNode::ChannelCell { .. } => receiver == Receiver::Reach,
                SurfaceChannelNode::SurfaceCell(_) => {
                    matches!(receiver, Receiver::Surface | Receiver::UnlinkedSurface)
                }
            };
            assert!(matched, "{receiver:?}: {transfer:?}");
            total += transfer.volume_m3;
            let ledgers = batch.runoff_ledgers()?;
            let [ledger] = ledgers.as_slice() else {
                return Err("one lane ledger".into());
            };
            assert_eq!(ledger.delivered_volume_m3.to_bits(), total.to_bits());
            let balance = batch.ledger()?;
            let child = batch.runoff().ok_or("the runoff child")?.ledger();
            assert_eq!(
                balance.runoff_storage_m3.to_bits(),
                child.storage_m3.to_bits()
            );
            assert!(
                (balance.runoff_delivered_m3 - total).abs() <= 4.0 * f64::EPSILON * total,
                "{balance:?}"
            );
            assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
        }
    }
    Ok(())
}

/// A package taken after two intervals resumes on a new client.
///
/// Frozen criterion, fixed before the first run: two more intervals from `import_snapshot` and
/// from `resume` give state text equal to the uninterrupted run after every interval.
#[test]
fn a_package_taken_mid_run_imports_and_resumes_bit_for_bit() -> Result {
    for receiver in RECEIVERS {
        let build = || -> Result<Batch> {
            let (setup, initial) = parts(receiver)?;
            Ok(Batch::new(crate::runtime::client(), setup, initial)?)
        };
        let mut whole = Vec::new();
        let mut batch = build()?;
        let mut data: Option<SurfaceChannelSnapshotData> = None;
        for index in 0..4 {
            batch = advance(&batch)?;
            whole.push(state(&batch)?);
            if index == 1 {
                data = Some(batch.snapshot_data()?);
            }
        }
        let data = data.ok_or("the mid-run package")?;
        let imported = build()?.import_snapshot(&data)?;
        let resumed = Batch::resume(
            crate::runtime::client(),
            &data,
            data.surface.setup.forcing.clone(),
        )?;
        for mut restarted in [imported, resumed] {
            for expected in &whole[2..] {
                restarted = advance(&restarted)?;
                assert_eq!(&state(&restarted)?, expected, "{receiver:?}");
            }
        }
    }
    Ok(())
}
