//! Runoff child restoration from checkpoint packages.

use crate::coupling::surface_channels::frames::rejected;
use crate::coupling::surface_channels::{SurfaceChannelError, SurfaceChannelInitialState};
use crate::runoff::{SubcatchmentBatch, SubcatchmentSnapshotData};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

/// Restores the runoff package of one frame at `prefix` into a copy of `base`.
///
/// `base` holds the immutable runoff model: the destination's child, or a fresh child of a
/// resumed definition. Its ledger holds the original runoff storage. The package must record
/// that storage bit for bit. A zero-history package must hold `initial.runoff` as its states.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming `{prefix}runoff` for a presence that differs
/// from `base`, a foreign epoch, or a package the runoff child rejects.
pub(in super::super) fn restored_child<R>(
    base: Option<&SubcatchmentBatch<R>>,
    initial: &SurfaceChannelInitialState,
    prefix: &str,
    package: Option<&SubcatchmentSnapshotData>,
) -> Result<Option<SubcatchmentBatch<R>>, SurfaceChannelError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let path = format!("{prefix}runoff");
    let (base, package) = match (base, package) {
        (Some(base), Some(package)) => (base, package),
        (None, None) => return Ok(None),
        _ => {
            return Err(rejected(
                &path,
                "presence differs from the destination children",
            ));
        }
    };
    let original = base.ledger().initial_storage_m3;
    let recorded = package.ledger.initial_storage_m3;
    if original.to_bits() != recorded.to_bits() {
        return Err(rejected(
            &format!("{path}.ledger.initial_storage_m3"),
            &format!("={recorded} differs from the storage of initial.runoff={original}"),
        ));
    }
    // Debug text separates signed zeros, which equality does not.
    let same_states = package.states == initial.runoff
        && format!("{:?}", package.states) == format!("{:?}", initial.runoff);
    if package.accepted_steps == 0 && !same_states {
        return Err(rejected(
            &format!("{path}.states"),
            "differs from initial.runoff in a zero-history package",
        ));
    }
    let mut child = base.clone();
    child
        .restore_data(package)
        .map_err(|error| rejected(&path, &format!("rejects: {error}")))?;
    Ok(Some(child))
}

#[cfg(test)]
mod tests {
    use super::super::basin::{Basin, Receiver, advance};
    use super::super::scenario::{input_message, share};
    use crate::coupling::surface_channels::SurfaceChannelSnapshotData;
    use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult, client};

    /// Rain on the subcatchment, in m/s.
    const RAIN: f64 = 2e-5;
    /// Requested interval, in s.
    const DT: f64 = 0.25;

    /// Debug text of the package, the composite ledger, the lane ledgers, and the runoff ledger.
    ///
    /// Debug text shows every bit of finite values, signed zero included.
    fn state(batch: &Batch) -> TestResult<String> {
        Ok(format!(
            "{} {} {:?} {:?} {:?} {:?}",
            batch.time(),
            batch.accepted_steps(),
            batch.snapshot_data()?,
            batch.ledger()?,
            batch.runoff_ledgers()?,
            batch.runoff().ok_or("the runoff child")?.ledger()
        ))
    }

    /// `batch` after `count` accepted intervals, with the state text after each one.
    fn run(mut batch: Batch, count: usize) -> TestResult<(Batch, Vec<String>)> {
        let mut states = Vec::new();
        for _ in 0..count {
            batch = advance(&batch, DT, RAIN)?;
            states.push(state(&batch)?);
        }
        Ok((batch, states))
    }

    /// The pond, a reach cell, and a span over both surface cells of a basin without the pond,
    /// each receiving the outlet.
    fn basins() -> [Basin; 3] {
        let mut cells = Basin::into(Receiver::Surface(vec![share(0, 1.0), share(1, 3.0)]));
        cells.pond = false;
        [
            Basin::into(Receiver::Pool),
            Basin::into(Receiver::Reach(vec![share(1, 1.0)])),
            cells,
        ]
    }

    /// Resumes `data` on a new client with its own surface window.
    fn resumed(data: &SurfaceChannelSnapshotData) -> TestResult<Batch> {
        Ok(Batch::resume(
            client(),
            data,
            data.surface.setup.forcing.clone(),
        )?)
    }

    /// Three intervals, then a trusted snapshot and a package taken mid-run.
    ///
    /// Frozen criteria, fixed before the first run: for every basin, three more intervals from
    /// `restore`, `import_snapshot`, and `resume` give state text equal to the uninterrupted
    /// run after every interval. Equal text means equal bits.
    #[test]
    fn restore_import_and_resume_continue_bit_for_bit() -> TestResult {
        for basin in basins() {
            let (middle, _) = run(basin.build()?, 3)?;
            let (_, mut whole) = run(basin.build()?, 6)?;
            let whole = whole.split_off(3);
            let snapshot = middle.snapshot();
            let data = middle.snapshot_data()?;
            assert!(data.runoff.is_some() && data.last_interval.is_some());
            let fresh = basin.build()?;
            for (path, restarted) in [
                ("restore", fresh.restore(&snapshot)?),
                ("import", fresh.import_snapshot(&data)?),
                ("resume", resumed(&data)?),
            ] {
                assert_eq!(state(&restarted)?, state(&middle)?, "{path}");
                let (_, states) = run(restarted, 3)?;
                assert_eq!(states, whole, "{path}: {:?}", basin.receiver);
            }
        }
        Ok(())
    }

    /// A zero-history package imports and resumes into the constructed state.
    ///
    /// Frozen criterion, fixed before the first run: equal state text.
    #[test]
    fn zero_history_packages_restart_at_the_epoch() -> TestResult {
        for basin in basins() {
            let built = basin.build()?;
            let data = built.snapshot_data()?;
            assert!(data.last_interval.is_none());
            let imported = basin.build()?.import_snapshot(&data)?;
            assert_eq!(state(&imported)?, state(&built)?);
            assert_eq!(state(&resumed(&data)?)?, state(&built)?);
        }
        Ok(())
    }

    /// Fails unless import and resume both reject `data` with a message naming `path`.
    fn rejects(basin: &Basin, data: &SurfaceChannelSnapshotData, path: &str) -> TestResult {
        let imported = input_message(basin.build()?.import_snapshot(data))?;
        let resumed = input_message(Batch::resume(
            client(),
            data,
            data.surface.setup.forcing.clone(),
        ))?;
        for message in [imported, resumed] {
            assert!(message.contains(path), "{path}: {message}");
        }
        Ok(())
    }

    /// One flipped runoff bit rejects with its field path.
    ///
    /// Frozen criteria, fixed before the first run: each edit below rejects at import and at
    /// resume with an input error naming the edited field.
    #[test]
    fn a_flipped_runoff_bit_rejects_with_its_field_path() -> TestResult {
        type Edit = fn(&mut SurfaceChannelSnapshotData) -> TestResult;
        fn next(value: &mut f64) {
            *value = value.next_up();
        }
        let edits: [(&str, Edit); 5] = [
            ("checkpoint runoff ", |data| {
                let runoff = data.runoff.as_mut().ok_or("runoff")?;
                next(&mut runoff.ledger.outlet_volume_m3);
                Ok(())
            }),
            ("checkpoint counters.runoff", |data| {
                let ledger = data.counters.runoff.ledgers.first_mut().ok_or("ledger")?;
                next(&mut ledger.delivered_volume_m3);
                Ok(())
            }),
            ("checkpoint last_interval.previous.runoff", |data| {
                let last = data.last_interval.as_mut().ok_or("last interval")?;
                let runoff = last.previous.runoff.as_mut().ok_or("runoff")?;
                next(&mut runoff.states[0].depths_m[0]);
                Ok(())
            }),
            ("checkpoint last_interval.diagnostics.runoff", |data| {
                let last = data.last_interval.as_mut().ok_or("last interval")?;
                let transfer = last.diagnostics.runoff.first_mut().ok_or("transfer")?;
                next(&mut transfer.volume_m3);
                Ok(())
            }),
            ("checkpoint runoff ", |data| {
                data.runoff = None;
                Ok(())
            }),
        ];
        for basin in basins() {
            let (middle, _) = run(basin.build()?, 3)?;
            let data = middle.snapshot_data()?;
            for (path, edit) in edits {
                let mut forged = data.clone();
                edit(&mut forged)?;
                rejects(&basin, &forged, path)?;
            }
        }
        Ok(())
    }
}
