//! A controlled pump that starts and stops on the stage of its sump.
//!
//! Frozen criterion, fixed before the first run (plan §5, "Controlled pump"):
//! `ControlledPumpBatch` starts and stops on sump stage. Commands switch at the exact decision
//! times. Flat-curve volume criterion per running interval: `|V - Q0 dt| <= 8 ε Q0 dt`, `ε` the
//! `f64` machine epsilon, and `|ΣV - Q0 T| <= 8 ε N Q0 T` over the `N` running intervals of
//! total duration `T`.
//!
//! The sump is a constant-area pond fed by a pipe. A relay starts the pump when a decision
//! observes the sump at or above the upper threshold, and stops it when a decision observes the
//! sump at or below the lower threshold. A decision takes effect on the interval that starts at
//! it. An independent plain-`f64` replay of the relay gives the exact decision stages and
//! switch times. A stopped interval moves `0.0` bit for bit. Common controls:
//! `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts zero. Every
//! interval checks the mass bound, the pairing, and the composite balance.

use crate::pools_evidence::{advance_to, build};
use crate::pools_fixture::laws::pump_on;
use crate::pools_fixture::{
    Case, PoolForcing, Result, controlled_pump, inflow_binding, inflow_input, operating, pond,
    pool, signal, structure,
};
use hydr::coupling::{
    SurfaceChannelOperating, SurfaceChannelOperationOwner, SurfaceChannelOperationReceipt,
};
use hydr::forcing::WaterAmount;
use hydr::structures::{ControlCommand, ControlLatch, DirectionPolicy, StructureRegime};

/// Plateau discharge of the pump, in m³/s.
const PLATEAU: f64 = 0.12;

/// Pipe inflow into the sump, in m³/s.
const FEED: f64 = 0.08;

/// Plan area of the sump, in m².
const SUMP_AREA_M2: f64 = 4.0;

/// Initial sump stage, in m.
const START_M: f64 = 1.0;

/// Lower and upper relay thresholds, in m.
const THRESHOLDS_M: [f64; 2] = [0.945, 1.05];

/// Decision times, one per second, in s.
const DECISIONS_S: [f64; 24] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0,
    18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
];

/// Run end, in s.
const END_S: f64 = 24.0;

/// Frozen rounding multiple of the flat-curve volume.
const ROUNDING: f64 = 8.0;

fn case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.1);
    case.ponds = vec![
        (pond("basin", 0.0, 100.0, 4.0)?, 100.0 * 1.0),
        (
            pond("sump", 0.0, SUMP_AREA_M2, 4.0)?,
            SUMP_AREA_M2 * START_M,
        ),
    ];
    case.inflows = vec![inflow_binding("feed", "sump")];
    case.structures = vec![structure(
        "lift",
        pump_on(&[(-1.0, PLATEAU), (3.0, PLATEAU)], 0.1, None)?,
        DirectionPolicy::AtoBOnly,
        SurfaceChannelOperationOwner::ControlledPump,
        (pool("sump"), 0.0),
        (pool("basin"), 0.0),
    )];
    case.operating = Some(SurfaceChannelOperating {
        pumps: Some(controlled_pump(
            "lift",
            THRESHOLDS_M,
            [
                ControlCommand::PumpStoppedIsolated,
                ControlCommand::PumpRunning { speed_ratio: 1.0 },
            ],
            (0.0, &DECISIONS_S[..], 30.0),
        )),
        ..operating(vec![signal("lift", pool("sump"))])
    });
    Ok(case)
}

/// What the relay decides at one decision time, replayed in plain `f64`.
#[derive(Clone, Copy, Debug)]
struct Decision {
    time: f64,
    stage: f64,
    /// Latch after the decision.
    latch: ControlLatch,
    transitioned: bool,
}

/// The decisions of the relay replayed with each interval of one second moving the sump by the
/// pipe, less the plateau while the latch stood high.
fn replay() -> Vec<Decision> {
    let (mut stage, mut latch) = (START_M, ControlLatch::Low);
    DECISIONS_S
        .iter()
        .map(|&time| {
            let pumping = if latch == ControlLatch::High {
                PLATEAU
            } else {
                0.0
            };
            stage += (FEED - pumping) / SUMP_AREA_M2;
            let before = latch;
            latch = match latch {
                ControlLatch::Low if stage >= THRESHOLDS_M[1] => ControlLatch::High,
                ControlLatch::High if stage <= THRESHOLDS_M[0] => ControlLatch::Low,
                held => held,
            };
            Decision {
                time,
                stage,
                latch,
                transitioned: latch != before,
            }
        })
        .collect()
}

#[test]
fn a_pump_starts_and_stops_at_the_exact_decision_times() -> Result {
    let expected = replay();
    // The replay itself must start, stop, and start again, and keep every decision clear of the
    // threshold its latch answers to: the upper while low, the lower while high.
    assert_eq!(expected.iter().filter(|d| d.transitioned).count(), 3);
    let mut before = ControlLatch::Low;
    for decision in &expected {
        let threshold = match before {
            ControlLatch::Low => THRESHOLDS_M[1],
            ControlLatch::High => THRESHOLDS_M[0],
        };
        assert!((decision.stage - threshold).abs() >= 0.004, "{decision:?}");
        before = decision.latch;
    }
    let forcing = PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate {
                rate_m3_per_s: FEED,
            },
        )],
    };
    for dt in [1.0, 0.5, 0.25] {
        let mut batch = build(&case()?)?;
        let (mut seen, mut running, mut total, mut length) = (Vec::new(), 0, 0.0, 0.0);
        for index in 1..=(END_S / dt).round() as usize {
            let end = index as f64 * dt;
            let (next, steps) = advance_to(batch, end, &forcing, &[0.0, FEED])?;
            batch = next;
            for step in &steps {
                let transfer = &step.structures[0];
                let SurfaceChannelOperationReceipt::ControlledPump { latch, .. } =
                    &transfer.receipt
                else {
                    return Err(format!("expected a pump receipt: {transfer:?}").into());
                };
                // The latch at the start of the interval is the one the last decision left.
                let time = step.start_s;
                let held = expected
                    .iter()
                    .rev()
                    .find(|d| d.time <= time)
                    .map_or(ControlLatch::Low, |d| d.latch);
                assert_eq!(*latch, held, "dt={dt}: interval from {time} s");
                let span = step.end_s - step.start_s;
                if held == ControlLatch::High {
                    let gap = (transfer.signed_volume_m3 - PLATEAU * span).abs();
                    assert!(
                        gap <= ROUNDING * f64::EPSILON * PLATEAU * span,
                        "dt={dt}: interval from {time} s: volume={:e}, gap={gap:e}",
                        transfer.signed_volume_m3
                    );
                    assert_eq!(transfer.regime, StructureRegime::PumpCurve);
                    running += 1;
                    total += transfer.signed_volume_m3;
                    length += span;
                } else {
                    assert_eq!(
                        transfer.signed_volume_m3.to_bits(),
                        0.0_f64.to_bits(),
                        "dt={dt}: interval from {time} s: {transfer:?}"
                    );
                    assert_eq!(transfer.regime, StructureRegime::PumpIsolated);
                }
            }
            let controls = batch
                .controlled_pumps()
                .ok_or("the pump controls")?
                .controls();
            let control = controls.last_step()?.ok_or("a control receipt")?;
            for decision in &control.decisions {
                seen.push((end, decision.sample.stage_m, decision.transitioned));
            }
        }
        assert_eq!(seen.len(), expected.len(), "dt={dt}");
        for (seen, expected) in seen.iter().zip(&expected) {
            assert_eq!(seen.0.to_bits(), expected.time.to_bits(), "dt={dt}");
            assert!(
                (seen.1 - expected.stage).abs() <= 1e-12,
                "dt={dt}: {seen:?} {expected:?}"
            );
            assert_eq!(
                seen.2, expected.transitioned,
                "dt={dt}: {seen:?} {expected:?}"
            );
        }
        let allowance = ROUNDING * f64::EPSILON * f64::from(running) * PLATEAU * length;
        assert!(
            (total - PLATEAU * length).abs() <= allowance,
            "dt={dt}: total={total:e}, expected={:e}",
            PLATEAU * length
        );
    }
    Ok(())
}
