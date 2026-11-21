//! A surveyed reach end into a pool: filling, then reversing, on three profiles.
//!
//! Frozen criterion, "Surface pool end"): every `pools_reach_end.rs`
//! criterion for the same case holds on the rectangle, the triangle, and three bands.
//! Pool-end pairing is exact.
//!
//! - Full span: the frozen references of `CASES` on the surveyed rectangle and triangle, and
//!   the six three-band references, at both reach ends after one interval of `0.125 s`:
//!   reservoir trace, transfer, pool head, pool volume, reach volume, and reach momentum, each
//!   within `1e-10` absolute plus `1e-10` relative. The same reach under a fixed head takes a
//!   different discharge by more than `1e-7`.
//! - Lake at rest: depth `1 m` within `1e-12`, discharge and momentum at most `1e-11`,
//!   storage change at most `1e-11`, transfer at most `1e-11`, for 16 intervals of `0.125 s`,
//!   at both ends with entrance loss `0` and `1`.
//! - Reversal: a ten-cell reach on a pool of `1 m²` flows into the pool and then back, both
//!   transfers beyond `1e-6 m³`, and both pool-end ledger counters positive.
//! - Pairing: the pool-end volume, the reach face volume, and the pool's incoming and outgoing
//!   volume agree in every bit, in every interval.

use crate::pools_evidence::analytic_controls;
use crate::pools_fixture::{Batch, Case, REACH, Result, pond};
use crate::pools_reach_end::{SPAN_S, check_pairing, end_of, near, normal, side};
use crate::surveyed_references::reference::CASES;
use crate::surveyed_references::surveyed_profiles::{RECTANGLE_1, TRIANGLE_1};
use crate::surveyed_references::{
    SurveyedCase, THREE_BANDS_CASES, definition, rest_depth, surveyed_case,
};
use hydr::channel::{
    ChannelBoundary, ChannelEnd, ChannelForcingInterval, ChannelInitialState, ChannelReach,
    ChannelReachDefinition,
};
use hydr::coupling::{SurfaceChannelForcingInterval, SurfaceChannelReachForcing};

/// Frozen allowance of the lake at rest.
const LAKE: f64 = 1e-11;

/// The coordinator of `case` with its reach end `end` on a pool, and the reach definition.
fn build(
    case: &SurveyedCase,
    end: ChannelEnd,
    cells: (usize, f64),
) -> Result<(Batch, ChannelReachDefinition)> {
    let mut base = Case::tanks(0.0, 0.5);
    base.ponds = vec![(
        pond("pool", 0.0, case.pool_area, 4.0)?,
        case.pool_area * case.elevation,
    )];
    base.pool_ends = vec![end_of(end, case.loss)];
    let (mut setup, mut initial) = base.parts()?;
    analytic_controls(&mut setup);
    setup.surface.config.gravity = case.gravity;
    let (count, length) = cells;
    let reach = definition(case.profile, case.gravity, count, length);
    setup.reaches[0].definition = reach.clone();
    initial.reaches[0].1 = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![length * case.interior_area_m2; count],
        longitudinal_momentum_m4_per_s: vec![0.0; count],
    };
    Ok((Batch::new(crate::runtime::client(), setup, initial)?, reach))
}

/// Forcing toward `end_s`: the pool owns `owned`. The other end holds water at rest at `depth`,
/// or a wall when `wall`.
fn forcing(
    batch: &Batch,
    end_s: f64,
    owned: ChannelEnd,
    wall: bool,
    depth: f64,
) -> SurfaceChannelForcingInterval {
    let outer = if wall {
        ChannelBoundary::Wall
    } else {
        ChannelBoundary::Exterior {
            depth_m: depth,
            velocity_m_per_s: 0.0,
        }
    };
    let held = |end: ChannelEnd| (end != owned).then(|| outer.clone());
    SurfaceChannelForcingInterval {
        start_s: batch.time(),
        end_s,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: held(ChannelEnd::Left),
            right: held(ChannelEnd::Right),
            lateral_inputs: Vec::new(),
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}

/// One frozen full-span reference at one reach end.
fn check_reference(case: &SurveyedCase, end: ChannelEnd) -> Result {
    let label = format!(
        "{} {} area {} {end:?}",
        case.name, case.profile.name, case.pool_area
    );
    let rest = rest_depth(case);
    let (origin, reach) = build(case, end, (1, case.cell_length_m))?;
    let before = format!("{:?}", origin.snapshot_data()?);
    let held = forcing(&origin, SPAN_S, end, false, rest);
    let next = origin.trial_to(SPAN_S, &held)?.into_next();
    assert_eq!(next.time().to_bits(), SPAN_S.to_bits(), "{label}");
    let step = next.last_step()?.ok_or("an accepted interval")?;
    assert_eq!(step.retries, 0, "{label}");
    let child = next.reach(REACH).ok_or("the reach")?;
    let receipt = child.last_step()?.ok_or("the reach receipt")?;
    let trace = receipt.boundary_traces[side(end)]
        .as_ref()
        .ok_or("the reservoir trace")?;
    assert!(trace.reservoir.is_some(), "{label}");
    let reference = case.expected;
    near(trace.depth_m, reference.depth);
    near(trace.outward_velocity_m_per_s, reference.velocity);
    near(trace.outward_discharge_m3_per_s, reference.discharge);
    near(trace.momentum_flux_m4_per_s2, reference.momentum_flux);
    let transfer = &step.pool_ends[0];
    near(transfer.signed_volume_m3, reference.transfer);
    near(transfer.reservoir_head_m, reference.head);
    near(step.pools[0].volume_m3, reference.pool_volume);
    let state = child.state()?;
    near(state[0].volume_m3, reference.channel_volume);
    near(
        state[0].longitudinal_momentum_m4_per_s,
        normal(end) * reference.right_momentum,
    );
    assert_ne!(
        transfer.reservoir_head_m.to_bits(),
        case.elevation.to_bits(),
        "{label}"
    );
    check_pairing(&next, end, side(end))?;
    // The same reach under the initial head instead of the pool's midpoint head moves a
    // different discharge: the pool feeds back.
    let fixed = ChannelBoundary::QuiescentReservoir {
        elevation_m: case.elevation,
        entrance_loss_coefficient: case.loss,
    };
    let outer = ChannelBoundary::Exterior {
        depth_m: rest,
        velocity_m_per_s: 0.0,
    };
    let (left, right) = match end {
        ChannelEnd::Left => (fixed, outer),
        ChannelEnd::Right => (outer, fixed),
    };
    let fixed_reach = ChannelReach::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        reach,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![case.cell_length_m * case.interior_area_m2],
            longitudinal_momentum_m4_per_s: vec![0.0],
        },
    )?;
    let fixed_next = fixed_reach
        .step(
            SPAN_S,
            &ChannelForcingInterval {
                start_s: 0.0,
                end_s: SPAN_S,
                left,
                right,
                lateral_inputs: Vec::new(),
            },
        )?
        .next;
    let fixed_receipt = fixed_next.last_step()?.ok_or("the held-head receipt")?;
    let fixed_trace = fixed_receipt.boundary_traces[side(end)]
        .as_ref()
        .ok_or("the held-head trace")?;
    assert!(
        (fixed_trace.outward_discharge_m3_per_s - trace.outward_discharge_m3_per_s).abs() > 1e-7,
        "{label}"
    );
    assert_eq!(format!("{:?}", origin.snapshot_data()?), before, "{label}");
    Ok(())
}

/// The rectangular cases when `triangle` is false, otherwise the triangular cases.
fn check_profile(triangle: bool, end: ChannelEnd) -> Result {
    for case in CASES.iter().map(surveyed_case) {
        if (case.profile.name == TRIANGLE_1.name) == triangle {
            check_reference(&case, end)?;
        }
    }
    Ok(())
}

fn check_bands(end: ChannelEnd) -> Result {
    for case in THREE_BANDS_CASES {
        check_reference(&case, end)?;
    }
    Ok(())
}

#[test]
fn surveyed_rectangle_frozen_references_hold_at_the_left_end() -> Result {
    check_profile(false, ChannelEnd::Left)
}

#[test]
fn surveyed_rectangle_frozen_references_hold_at_the_right_end() -> Result {
    check_profile(false, ChannelEnd::Right)
}

#[test]
fn surveyed_triangle_frozen_references_hold_at_the_left_end() -> Result {
    check_profile(true, ChannelEnd::Left)
}

#[test]
fn surveyed_triangle_frozen_references_hold_at_the_right_end() -> Result {
    check_profile(true, ChannelEnd::Right)
}

#[test]
fn surveyed_three_band_frozen_references_hold_at_the_left_end() -> Result {
    check_bands(ChannelEnd::Left)
}

#[test]
fn surveyed_three_band_frozen_references_hold_at_the_right_end() -> Result {
    check_bands(ChannelEnd::Right)
}

#[test]
fn surveyed_lake_at_rest_keeps_its_depth_at_both_ends_and_both_losses() -> Result {
    let mut bands = THREE_BANDS_CASES[0];
    // Three bands hold 3 m² at depth 1 m, a knot depth.
    bands.interior_area_m2 = 3.0;
    for mut case in [surveyed_case(&CASES[0]), surveyed_case(&CASES[3]), bands] {
        case.elevation = 1.0;
        for end in [ChannelEnd::Left, ChannelEnd::Right] {
            for loss in [0.0, 1.0] {
                case.loss = loss;
                let (origin, _) = build(&case, end, (1, case.cell_length_m))?;
                let storage = origin.ledger()?.storage_m3;
                let mut current = origin;
                for index in 0..16 {
                    let target = (index + 1) as f64 / 8.0;
                    let held = forcing(&current, target, end, false, 1.0);
                    current = current.trial_to(target, &held)?.into_next();
                    assert_eq!(current.time().to_bits(), target.to_bits());
                    let state = current.reach(REACH).ok_or("the reach")?.state()?;
                    assert!((state[0].depth_m - 1.0).abs() <= 1e-12);
                    assert!(state[0].discharge_m3_per_s.abs() <= LAKE);
                    assert!(state[0].longitudinal_momentum_m4_per_s.abs() <= LAKE);
                    assert!((current.ledger()?.storage_m3 - storage).abs() <= LAKE);
                    let moved = check_pairing(&current, end, side(end))?;
                    assert!(
                        moved.abs() <= LAKE,
                        "{} end={end:?}, loss={loss}: {moved:e}",
                        case.profile.name
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn surveyed_filling_pool_reverses_without_a_policy_switch_or_transfer_scaling() -> Result {
    let mut case = surveyed_case(&CASES[0]);
    case.profile = RECTANGLE_1;
    case.interior_area_m2 = 1.0;
    case.gravity = 1.0;
    case.elevation = 0.95;
    case.pool_area = 1.0;
    case.loss = 0.0;
    let end = ChannelEnd::Right;
    let (origin, _) = build(&case, end, (10, 0.1))?;
    let (mut positive, mut negative) = (false, false);
    let mut current = origin;
    let mut taken = 0;
    // Stop at the first reversal. Later intervals repeat the same exchange.
    while current.time() < 4.0 && !(positive && negative) {
        let held = forcing(&current, 4.0, end, true, 1.0);
        let start = current.time();
        current = current.trial_to(4.0, &held)?.into_next();
        assert!(current.time() > start && current.time() <= 4.0);
        let transfer = check_pairing(&current, end, 10)?;
        positive |= transfer > 1e-6;
        negative |= transfer < -1e-6;
        taken += 1;
        assert!(taken < 100_000, "the pool end stalls");
    }
    assert!(positive && negative, "the pool end never reverses");
    let ledger = &current.pool_end_ledgers()?[0];
    assert!(ledger.reach_to_pool_m3 > 0.0, "{ledger:?}");
    assert!(ledger.pool_to_reach_m3 > 0.0, "{ledger:?}");
    // The pool-end ledger closes the pool: its storage changed by exactly the net volume.
    let pools = current.pools().ok_or("the pools")?.states()?;
    let net = ledger.reach_to_pool_m3 - ledger.pool_to_reach_m3;
    assert!((pools[0].volume_m3 - (0.95 + net)).abs() <= ledger.roundoff_bound_m3 + 1e-12);
    Ok(())
}
