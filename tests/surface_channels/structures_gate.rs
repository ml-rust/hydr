//! Gates between ponds: a prescribed opening ramp and an actuated stage rule.
//!
//! Frozen criteria, fixed before the first run (plan §5):
//!
//! | Claim | Frozen criterion |
//! | --- | --- |
//! | Prescribed gate ramp: linear opening ramp between two tanks, free aperture | Self-refinement order `>= 1.8` for the final stage. Closed intervals move `0.0` bit for bit. |
//! | Actuated gate on a stage rule: `ControlledGateBatch` opens a pond outlet when the pond stage reaches its upper threshold. Every decision stage keeps at least `0.01 m` from each threshold. | Switch at the first decision with stage `>=` threshold. Order `>= 1.8` for the final stage. Receipts hold the midpoint opening and its error. |
//!
//! Self-refinement compares each final stage with the one from half the duration. The two
//! orders of the three differences are the two finest pairs. Common controls:
//! `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts zero. Four
//! durations, each half the last. Every interval checks the mass bound, the pairing, and the
//! composite balance.

use crate::pools_evidence::{advance, advance_to, build, pool_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, controlled_gate, gate_law, inflow_binding, inflow_input, operating,
    pond, pool, signal, structure, trajectory,
};
use hydr::coupling::{SurfaceChannelOperating, SurfaceChannelOperationOwner};
use hydr::coupling::{SurfaceChannelOperationReceipt, SurfaceChannelStepDiagnostics};
use hydr::forcing::WaterAmount;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Frozen order floor.
const ORDER: f64 = 1.8;

/// Interval durations, each half the last, in s. Every one divides the decision spacing.
const DURATIONS_S: [f64; 4] = [0.5, 0.25, 0.125, 0.0625];

/// Stage of the reach cell no structure touches, in m. It is shallow enough that the reach
/// accepts every requested duration.
const UNUSED_REACH_M: f64 = 0.1;

/// Gate sill, in m. The downstream pond stays below it, so the aperture runs free.
const SILL_M: f64 = 0.6;

/// Fails unless the orders of the self-refinement differences of `finals` reach `ORDER` on
/// both pairs.
fn assert_self_refinement(finals: &[f64]) {
    let differences: Vec<f64> = finals.windows(2).map(|w| (w[0] - w[1]).abs()).collect();
    let orders: Vec<f64> = differences
        .windows(2)
        .map(|w| (w[0] / w[1]).log2())
        .collect();
    println!("finals={finals:?}, differences={differences:?}, orders={orders:?}");
    assert!(
        orders.iter().all(|order| *order >= ORDER),
        "finals={finals:?}, differences={differences:?}, orders={orders:?}"
    );
}

/// Run end of the ramp case, in s.
const RAMP_END_S: f64 = 6.0;

/// Time the ramp starts and the time it ends, in s, and its final opening, in m.
const RAMP: [f64; 3] = [1.0, 3.0, 0.4];

/// Two ponds joined by a gate whose opening follows a prescribed ramp.
fn ramp_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, UNUSED_REACH_M);
    case.ponds = vec![
        (pond("down", 0.0, 16.0, 4.0)?, 16.0 * 0.2),
        (pond("up", 0.0, 16.0, 4.0)?, 16.0 * 2.0),
    ];
    case.structures = vec![structure(
        "gate",
        gate_law(SILL_M)?,
        DirectionPolicy::Bidirectional,
        SurfaceChannelOperationOwner::Prescribed,
        (pool("up"), 0.0),
        (pool("down"), 0.0),
    )];
    case.operating = Some(SurfaceChannelOperating {
        trajectories: vec![trajectory(
            "gate",
            &[
                (0.0, 0.0),
                (RAMP[0], 0.0),
                (RAMP[1], RAMP[2]),
                (100.0, RAMP[2]),
            ],
        )?],
        ..operating(Vec::new())
    });
    Ok(case)
}

/// Final stage of the downstream pond after intervals of `dt`.
fn ramp_final(dt: f64) -> Result<f64> {
    let mut batch = build(&ramp_case()?)?;
    let mut closed = 0;
    while batch.time() < RAMP_END_S {
        let (next, step) = advance(&batch, dt, &PoolForcing::default(), &[])?;
        batch = next;
        let transfer = &step.structures[0];
        let SurfaceChannelOperationReceipt::PrescribedGate {
            closed: is_closed, ..
        } = transfer.receipt
        else {
            return Err(format!("expected a prescribed receipt: {transfer:?}").into());
        };
        if step.end_s <= RAMP[0] {
            assert!(is_closed, "{transfer:?}");
        }
        if is_closed {
            closed += 1;
            assert_eq!(
                transfer.signed_volume_m3.to_bits(),
                0.0_f64.to_bits(),
                "{transfer:?}"
            );
            assert_eq!(transfer.regime, StructureRegime::GateClosed, "{transfer:?}");
        } else {
            assert_eq!(
                transfer.regime,
                StructureRegime::FreeGateAperture,
                "{transfer:?}"
            );
            assert!(transfer.signed_volume_m3 > 0.0, "{transfer:?}");
        }
    }
    assert_eq!(closed, (RAMP[0] / dt).round() as usize, "dt={dt}");
    pool_stage(&batch, "down")
}

#[test]
fn a_prescribed_gate_ramp_refines_at_second_order() -> Result {
    let finals = DURATIONS_S
        .map(ramp_final)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    assert_self_refinement(&finals);
    Ok(())
}

/// Decision times of the stage rule, in s.
const DECISIONS_S: [f64; 12] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
];

/// Run end of the stage-rule case, in s.
const RULE_END_S: f64 = 12.0;

/// Thresholds of the relay, in m.
const THRESHOLDS_M: [f64; 2] = [0.7, 1.125];

/// Target opening of the outlet, in m.
const TARGET_M: f64 = 0.2;

/// Actuator rate of the stage-rule case, in m/s. The outlet reaches `0.18 m` at the run end:
/// the run holds no actuator arrival.
const RATE_M_PER_S: f64 = 0.02;

/// Actuator rate of the arrival case, in m/s. The outlet arrives at its target inside the run.
const ARRIVING_RATE_M_PER_S: f64 = 0.06;

/// Pipe inflow into the pond, in m³/s, and the pond area, in m².
const FEED: (f64, f64) = (0.2, 4.0);

/// A pond filling from a pipe, with an actuated outlet gate onto a large downstream basin.
fn rule_case(rate: f64) -> Result<Case> {
    let mut case = Case::tanks(0.0, UNUSED_REACH_M);
    case.ponds = vec![
        (pond("basin", 0.0, 100.0, 4.0)?, 100.0 * 0.2),
        (pond("pond", 0.0, FEED.1, 4.0)?, FEED.1 * 1.0),
    ];
    case.inflows = vec![inflow_binding("feed", "pond")];
    case.structures = vec![structure(
        "outlet",
        gate_law(SILL_M)?,
        DirectionPolicy::Bidirectional,
        SurfaceChannelOperationOwner::Actuated,
        (pool("pond"), 0.0),
        (pool("basin"), 0.0),
    )];
    case.operating = Some(SurfaceChannelOperating {
        gates: Some(controlled_gate(
            "outlet",
            THRESHOLDS_M,
            ([0.0, TARGET_M], rate),
            (0.0, &DECISIONS_S[..], 20.0),
        )),
        ..operating(vec![signal("outlet", pool("pond"))])
    });
    Ok(case)
}

/// What one run of the stage rule recorded.
struct Rule {
    /// Final stage of the pond.
    last: f64,
    /// Decision time, pond stage observed, and whether the latch changed, per decision.
    decisions: Vec<(f64, f64, bool)>,
    /// The receipts of every interval.
    steps: Vec<SurfaceChannelStepDiagnostics>,
}

/// The stage-rule case with actuator `rate` after intervals of `dt`.
fn rule_run(dt: f64, rate: f64) -> Result<Rule> {
    let forcing = PoolForcing {
        sources: None,
        inflows: vec![inflow_input(
            "feed",
            WaterAmount::Rate {
                rate_m3_per_s: FEED.0,
            },
        )],
    };
    let mut batch = build(&rule_case(rate)?)?;
    let (mut decisions, mut steps) = (Vec::new(), Vec::new());
    let count = (RULE_END_S / dt).round() as usize;
    for index in 1..=count {
        // An interval can end early, at an actuator arrival bound. Each decision is read from
        // the interval that ends at it, so every sub-interval is read in turn.
        let end = index as f64 * dt;
        let (next, taken) = advance_to(batch, end, &forcing, &[0.0, FEED.0])?;
        batch = next;
        let controls = batch
            .controlled_gates()
            .ok_or("the gate controls")?
            .controls();
        let control = controls.last_step()?.ok_or("a control receipt")?;
        for decision in &control.decisions {
            decisions.push((end, decision.sample.stage_m, decision.transitioned));
        }
        steps.extend(taken);
    }
    Ok(Rule {
        last: pool_stage(&batch, "pond")?,
        decisions,
        steps,
    })
}

/// Ideal actuator opening at `t`, in m, for a high command at `switch` and actuator `rate`.
fn ideal_opening(t: f64, switch: f64, rate: f64) -> f64 {
    ((t - switch).max(0.0) * rate).min(TARGET_M)
}

/// Fails unless `rule` switched at the first decision with stage at or above the upper
/// threshold, kept every decision 0.01 m from both thresholds, and carried the exact interval
/// opening with its error in every receipt.
fn verify(rule: &Rule, dt: f64, rate: f64) -> Result {
    {
        assert_eq!(rule.decisions.len(), DECISIONS_S.len(), "dt={dt}");
        // Every decision stage keeps 0.01 m from each threshold.
        for &(time, stage, _) in &rule.decisions {
            for threshold in THRESHOLDS_M {
                assert!(
                    (stage - threshold).abs() >= 0.01,
                    "dt={dt}: decision at {time} s observed {stage} m against {threshold} m"
                );
            }
        }
        // The first decision with stage at or above the upper threshold switches. No other does.
        let first = rule
            .decisions
            .iter()
            .position(|d| d.1 >= THRESHOLDS_M[1])
            .ok_or("a decision above the upper threshold")?;
        for (index, &(time, stage, transitioned)) in rule.decisions.iter().enumerate() {
            assert_eq!(transitioned, index == first, "dt={dt}: {time} s, {stage} m");
        }
        let switch = rule.decisions[first].0;
        // Independent of the code: until the switch only the pipe moves the pond, by
        // `FEED.0 / FEED.1` per s from its initial stage of 1 m.
        assert_eq!(switch, DECISIONS_S[first]);
        for &(time, stage, _) in &rule.decisions[..=first] {
            let exact = 1.0 + FEED.0 / FEED.1 * time;
            assert!(
                (stage - exact).abs() <= 1e-12,
                "dt={dt}: {time} s, {stage} m"
            );
        }
        assert!(first > 0 && rule.decisions[first - 1].1 < THRESHOLDS_M[1]);
        // The receipts hold the interval opening and its error.
        for step in &rule.steps {
            let transfer = &step.structures[0];
            let SurfaceChannelOperationReceipt::ActuatedGate {
                opening_m,
                lower_opening_m,
                upper_opening_m,
                absolute_error_m,
                midpoint_rounding_m,
                ..
            } = &transfer.receipt
            else {
                return Err(format!("expected an actuated receipt: {transfer:?}").into());
            };
            let exact = 0.5
                * (ideal_opening(step.start_s, switch, rate)
                    + ideal_opening(step.end_s, switch, rate));
            assert!(
                (opening_m - exact).abs() <= *absolute_error_m,
                "dt={dt}: opening={opening_m}, exact={exact}, error={absolute_error_m}"
            );
            assert!(
                lower_opening_m <= opening_m && opening_m <= upper_opening_m,
                "dt={dt}: {transfer:?}"
            );
            assert!(
                *midpoint_rounding_m <= *absolute_error_m,
                "dt={dt}: {transfer:?}"
            );
            assert_eq!(
                transfer.operation,
                StructureOperatingPoint::GateOpening {
                    opening_m: *opening_m
                }
            );
        }
    }
    Ok(())
}

#[test]
fn an_actuated_gate_switches_at_the_first_decision_above_its_threshold() -> Result {
    assert!(ideal_opening(RULE_END_S, 3.0, RATE_M_PER_S) < TARGET_M);
    let mut finals = Vec::new();
    for dt in DURATIONS_S {
        let rule = rule_run(dt, RATE_M_PER_S)?;
        verify(&rule, dt, RATE_M_PER_S)?;
        finals.push(rule.last);
    }
    assert_self_refinement(&finals);
    Ok(())
}

/// The same rule with an outlet that arrives at its target inside the run. The actuator brackets
/// its arrival between a lower and an upper boundary a few ulps apart, and the coordinator ends
/// an interval at each.
#[test]
fn an_actuated_gate_arrives_at_its_target_inside_the_run() -> Result {
    let rule = rule_run(0.5, ARRIVING_RATE_M_PER_S)?;
    verify(&rule, 0.5, ARRIVING_RATE_M_PER_S)?;
    let last = rule.steps.last().ok_or("an accepted interval")?;
    let SurfaceChannelOperationReceipt::ActuatedGate { opening_m, .. } =
        &last.structures[0].receipt
    else {
        return Err("expected an actuated receipt".into());
    };
    assert_eq!(opening_m.to_bits(), TARGET_M.to_bits());
    Ok(())
}
