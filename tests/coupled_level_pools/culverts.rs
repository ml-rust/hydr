//! Culverts between coupled level pools: a drain against the independent reference, a reversal
//! through a two-way and a one-end culvert, a culvert beside a road weir, and restart.

use super::culverts_fixture::{culvert, reversal, roadway, system};
use super::drain_rows::{
    AREA_DOWN, AREA_UP, COUNTS, END_DROP_UP, END_S, FINEST_RELATIVE, INVERT_DOWN, INVERT_UP,
    LENGTH, ORDER, RISE, STAGE_DOWN, STAGE_UP,
};
use super::fixture::Result;
use hydr::coupling::{
    CoupledLevelPoolBatch, CoupledLevelPoolForcingInterval, CoupledLevelPoolInitialState,
    CoupledLevelPoolSetup, CoupledLevelPoolStepDiagnostics, CoupledPoolInflow,
    CoupledPoolInflowValue,
};

type Batch = CoupledLevelPoolBatch<crate::runtime::Runtime>;

fn bits(value: u64) -> f64 {
    f64::from_bits(value)
}

fn build(parts: (CoupledLevelPoolSetup, CoupledLevelPoolInitialState)) -> Result<Batch> {
    let (setup, initial) = parts;
    Ok(Batch::new(crate::runtime::client(), setup, initial)?)
}

/// Accepted stage of pool `index` in setup order, in m.
fn stage(batch: &Batch, index: usize) -> Result<f64> {
    Ok(batch
        .pools()
        .outputs()?
        .get(index)
        .and_then(|output| output.stage_m)
        .ok_or("a wet pool")?)
}

/// The receipt of `batch`'s last interval, which must end at `end`, after its budget checks.
fn checked(batch: &Batch, end: f64) -> Result<CoupledLevelPoolStepDiagnostics> {
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    assert_eq!(step.end_s.to_bits(), end.to_bits(), "{step:?}");
    let config = &batch.setup().config;
    assert!(step.head_residual_m <= config.head_absolute_m, "{step:?}");
    assert!(
        step.transfer_residual_m3 <= config.transfer_absolute_m3,
        "{step:?}"
    );
    assert!(
        step.mass_residual_m3.abs() <= config.mass_absolute_m3,
        "{step:?}"
    );
    Ok(step)
}

/// Signed volume of structure `index` over the last interval.
fn signed(step: &CoupledLevelPoolStepDiagnostics, index: usize) -> Result<f64> {
    Ok(step
        .transfers
        .get(index)
        .ok_or("the structure receipt")?
        .signed_volume_m3)
}

/// Relative error of the upper pool's drop at `END_S` after `n` intervals.
///
/// Frozen criteria, fixed before the first run, in `resource/evidence/culverts/drain.json`.
fn drain_error(n: usize) -> Result<f64> {
    let mut batch = build(system(
        &[
            ("down", bits(AREA_DOWN), bits(STAGE_DOWN)),
            ("up", bits(AREA_UP), bits(STAGE_UP)),
        ],
        vec![culvert(
            "culvert",
            ["up", "down"],
            (bits(RISE), bits(LENGTH)),
            [bits(INVERT_UP), bits(INVERT_DOWN)],
            true,
        )?],
    )?)?;
    let dt = bits(END_S) / n as f64;
    for _ in 0..n {
        let end = batch.time() + dt;
        batch = batch.trial_to(end)?.into_next();
        let step = checked(&batch, end)?;
        assert!(signed(&step, 0)? > 0.0, "{step:?}");
    }
    let drop = bits(STAGE_UP) - stage(&batch, 1)?;
    let expected = 0.5 * (bits(END_DROP_UP[0]) + bits(END_DROP_UP[1]));
    Ok((drop - expected).abs() / expected)
}

#[test]
fn coupled_pools_drain_through_a_drowned_culvert_at_second_order() -> Result {
    let errors = COUNTS
        .map(drain_error)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let orders: Vec<f64> = errors.windows(2).map(|w| (w[0] / w[1]).log2()).collect();
    println!("coupled culvert drain: errors={errors:?}, orders={orders:?}");
    assert!(errors[errors.len() - 1] <= FINEST_RELATIVE, "{errors:?}");
    assert!(orders.iter().all(|order| *order >= ORDER), "{orders:?}");
    Ok(())
}

/// Requested duration of every reversal interval, in s.
const REVERSAL_DT: f64 = 0.5;

/// One reversal interval: the boundary `rise` feeds the sea at 0.2 m³/s.
fn rise(batch: &Batch) -> Result<Batch> {
    let start = batch.time();
    let end = start + REVERSAL_DT;
    let forcing = CoupledLevelPoolForcingInterval {
        start_s: start,
        end_s: end,
        sources: None,
        inflows: vec![CoupledPoolInflow {
            boundary_id: "rise".into(),
            input_id: "selected-input-rise".into(),
            value: CoupledPoolInflowValue::Rate { rate_m3_per_s: 0.2 },
        }],
    };
    let next = batch.trial_with_forcing_to(end, &forcing)?.into_next();
    checked(&next, end)?;
    Ok(next)
}

/// Frozen criteria, fixed before the first run:
/// - Every interval takes its full duration within its head, transfer, and mass budgets.
/// - The two-way culvert drains the lagoon first, then the fed sea rises above it and the
///   culvert carries water back into the lagoon.
/// - While the two-way culvert drains the lagoon, the one-end culvert does too. Once the flow
///   reverses, the one-end culvert moves `0.0` bit for bit.
#[test]
fn a_rising_sea_reverses_a_two_way_culvert_and_closes_a_one_end_culvert() -> Result {
    let mut batch = build(reversal()?)?;
    let (mut drained, mut reversed) = (false, false);
    for _ in 0..24 {
        batch = rise(&batch)?;
        let step = batch.last_step()?.ok_or("an accepted interval")?;
        let (tide, flap) = (signed(&step, 0)?, signed(&step, 1)?);
        if tide > 0.0 {
            assert!(!reversed, "the flow returned to the sea: {step:?}");
            drained = true;
            assert!(flap > 0.0, "{step:?}");
        }
        if tide < 0.0 {
            reversed = true;
            assert_eq!(flap.to_bits(), 0.0f64.to_bits(), "{step:?}");
        }
    }
    assert!(drained && reversed, "the culvert never reversed");
    Ok(())
}

/// Requested duration of every roadway interval, in s.
const ROAD_DT: f64 = 0.5;

/// `batch` after `count` checked roadway intervals.
fn road_run(mut batch: Batch, count: usize) -> Result<Batch> {
    for _ in 0..count {
        let end = batch.time() + ROAD_DT;
        batch = batch.trial_to(end)?.into_next();
        checked(&batch, end)?;
    }
    Ok(batch)
}

/// Debug text of the package and the live pools. Debug text shows every bit of finite values.
fn text(batch: &Batch) -> Result<String> {
    Ok(format!(
        "{:?} {:?}",
        batch.snapshot_data()?,
        batch.pools().states()?
    ))
}

/// Frozen criteria, fixed before the first run:
/// - The road weir and the culvert both carry water from the upper pool while it stands above
///   the crest.
/// - Snapshot and restore, and owned data and restore, each match the checkpoint bit for bit,
///   and match the uninterrupted run bit for bit after four more intervals.
#[test]
fn a_culvert_beside_a_road_weir_restarts_bit_for_bit() -> Result {
    let checkpoint = road_run(build(roadway()?)?, 4)?;
    let step = checkpoint.last_step()?.ok_or("an accepted interval")?;
    assert!(
        signed(&step, 0)? > 0.0 && signed(&step, 1)? > 0.0,
        "{step:?}"
    );
    let expected = text(&checkpoint)?;
    let data = checkpoint.snapshot_data()?;
    let mut restored = build(roadway()?)?;
    restored.restore(&checkpoint.snapshot())?;
    let mut imported = build(roadway()?)?;
    imported.restore_data(&data)?;
    let uninterrupted = text(&road_run(checkpoint, 4)?)?;
    for (path, batch) in [("restore", restored), ("data", imported)] {
        assert_eq!(text(&batch)?, expected, "{path} at the checkpoint");
        assert_eq!(text(&road_run(batch, 4)?)?, uninterrupted, "{path} after");
    }
    Ok(())
}
