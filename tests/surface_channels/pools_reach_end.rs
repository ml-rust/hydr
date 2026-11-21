//! A reach end into a pool: filling, then reversing, against the frozen `channel_pools` cases.
//!
//! Frozen criterion, fixed before the first run, "Pool end": `pools_reach_end.rs`:
//! reach end into a pool, filling then reversing. Every `channel_pools` frozen criterion for
//! the same case holds. Pool-end pairing is exact.
//!
//! The cases are the `channel_pools` physics cases, run through the coordinator with a dry
//! surface cell beside them. The surface gravity equals the reach gravity of each case.
//!
//! - Full span: every one of the frozen 150-digit references, at both reach ends, after one
//!   interval of `0.125 s`: reservoir trace, transfer, pool head, pool volume, reach volume,
//!   and reach momentum, each within `1e-10` absolute plus `1e-10` relative. The same reach
//!   under a fixed head takes a different discharge by more than `1e-7`.
//! - Lake at rest: depth `1 m` within `1e-12`, discharge and momentum at most `1e-11`, storage
//!   change at most `1e-11`, transfer at most `1e-11`, for 16 intervals of `0.125 s`, at both
//!   ends and with entrance loss `0` and `1`.
//! - Reversal: a ten-cell reach on a pool of `1 m²` flows into the pool and then back, both
//!   transfers beyond `1e-6 m³`, and both pool-end ledger counters positive.
//! - Pairing: the pool-end volume, the reach face volume, and the pool's incoming and outgoing
//!   volume agree in every bit, in every interval. Every interval checks the mass bound and the
//!   composite balance. A snapshot taken before a trial is unchanged after it.

use crate::channel_pool_reference::{CASES, Case as Reference};
use crate::pools_evidence::{analytic_controls, check};
use crate::pools_fixture::{Batch, Case, DATUM, REACH, Result, pond, pool_end};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelEnd, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelReach, ChannelReachDefinition, RectangularChannelCell,
    TriangularChannelCell,
};
use hydr::coupling::{
    SurfaceChannelForcingInterval, SurfaceChannelPoolEnd, SurfaceChannelReachForcing,
};

/// Interval of the full-span references, in s.
pub(crate) const SPAN_S: f64 = 0.125;

/// Frozen allowance of the references: absolute plus relative.
const REFERENCE: f64 = 1e-10;

/// Frozen allowance of the lake at rest.
const LAKE: f64 = 1e-11;

/// Side index of a reach end in the reach child's traces and faces.
pub(crate) fn side(end: ChannelEnd) -> usize {
    match end {
        ChannelEnd::Left => 0,
        ChannelEnd::Right => 1,
    }
}

/// Outward normal of a reach end.
pub(crate) fn normal(end: ChannelEnd) -> f64 {
    match end {
        ChannelEnd::Left => -1.0,
        ChannelEnd::Right => 1.0,
    }
}

pub(crate) fn near(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= REFERENCE + REFERENCE * expected.abs(),
        "actual={actual:e}, independent reference={expected:e}"
    );
}

/// The `channel_pools` reach: `count` cells of `length` m on bed `0`, rectangular of width
/// `coefficient` or triangular with banks of slope `coefficient`.
fn reach_definition(
    triangle: bool,
    gravity: f64,
    coefficient: f64,
    count: usize,
    length: f64,
) -> ChannelReachDefinition {
    let geometry = if triangle {
        ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: coefficient,
            right_bank_slope_h_per_v: coefficient,
            bed_elevation_m: 0.0,
            cells: (0..count)
                .map(|i| TriangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    } else {
        ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    width_m: coefficient,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit finite-pool analytical reach".into(),
        geometry,
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity,
            max_step: 1.0,
            ..ChannelConfig::default()
        },
    }
}

/// The pool end of `end` into the pool `pool`, with `loss`.
pub(crate) fn end_of(end: ChannelEnd, loss: f64) -> SurfaceChannelPoolEnd {
    SurfaceChannelPoolEnd {
        entrance_loss_coefficient: loss,
        ..pool_end("end", end, "pool")
    }
}

/// The coordinator of `case` with its reach end `end` on a pool, and the reach definition.
fn build(
    case: &Reference,
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
    let definition = reach_definition(case.triangle, case.gravity, case.coefficient, count, length);
    setup.reaches[0].definition = definition.clone();
    initial.reaches[0].1 = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![length * case.coefficient; count],
        longitudinal_momentum_m4_per_s: vec![0.0; count],
    };
    Ok((
        Batch::new(crate::runtime::client(), setup, initial)?,
        definition,
    ))
}

/// Forcing of the reach toward `end_s`: the pool owns `owned`, the other end takes the frozen
/// exterior depth and velocity, or a wall when `wall`.
fn forcing(
    batch: &Batch,
    end_s: f64,
    owned: ChannelEnd,
    wall: bool,
) -> SurfaceChannelForcingInterval {
    let outer = if wall {
        ChannelBoundary::Wall
    } else {
        ChannelBoundary::Exterior {
            depth_m: 1.0,
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

/// Fails unless the last interval pairs the pool end exactly: the pool-end volume against the
/// reach face volume `face` of `end`, and the pool's incoming and outgoing volume against its
/// sign. Checks the interval's mass, pairing, and composite balance too.
pub(crate) fn check_pairing(batch: &Batch, end: ChannelEnd, face: usize) -> Result<f64> {
    check(batch, &[])?;
    let step = batch.last_step()?.ok_or("an accepted interval")?;
    let transfer = &step.pool_ends[0];
    let child = batch.reach(REACH).ok_or("the reach")?.last_step()?;
    let face_volume = child.ok_or("the reach receipt")?.face_volume_m3[face];
    assert_eq!(
        transfer.signed_volume_m3.to_bits(),
        (normal(end) * face_volume).to_bits(),
        "{transfer:?}"
    );
    let pool = &step.pools[0];
    assert_eq!(
        pool.incoming_m3.to_bits(),
        transfer.signed_volume_m3.max(0.0).to_bits()
    );
    assert_eq!(
        pool.outgoing_m3.to_bits(),
        (-transfer.signed_volume_m3).max(0.0).to_bits()
    );
    Ok(transfer.signed_volume_m3)
}

#[test]
fn every_frozen_full_span_reference_holds_at_both_ends() -> Result {
    for case in CASES {
        for end in [ChannelEnd::Left, ChannelEnd::Right] {
            let (origin, definition) = build(&case, end, (1, 128.0))?;
            let before = format!("{:?}", origin.snapshot_data()?);
            let held = forcing(&origin, SPAN_S, end, false);
            let next = origin.trial_to(SPAN_S, &held)?.into_next();
            assert_eq!(next.time().to_bits(), SPAN_S.to_bits());
            let step = next.last_step()?.ok_or("an accepted interval")?;
            assert_eq!(step.retries, 0, "{} area {}", case.name, case.pool_area);
            let child = next.reach(REACH).ok_or("the reach")?;
            let receipt = child.last_step()?.ok_or("the reach receipt")?;
            let trace = receipt.boundary_traces[side(end)]
                .as_ref()
                .ok_or("the reservoir trace")?;
            assert!(trace.reservoir.is_some());
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
                case.elevation.to_bits()
            );
            check_pairing(&next, end, side(end))?;
            // The same reach under the initial head instead of the pool's midpoint head moves
            // a different discharge: the pool feeds back.
            let fixed = ChannelBoundary::QuiescentReservoir {
                elevation_m: case.elevation,
                entrance_loss_coefficient: case.loss,
            };
            let outer = ChannelBoundary::Exterior {
                depth_m: 1.0,
                velocity_m_per_s: 0.0,
            };
            let (left, right) = match end {
                ChannelEnd::Left => (fixed, outer),
                ChannelEnd::Right => (outer, fixed),
            };
            let fixed_reach = ChannelReach::<crate::runtime::Runtime>::new(
                crate::runtime::client(),
                definition,
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: vec![128.0 * case.coefficient],
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
                (fixed_trace.outward_discharge_m3_per_s - trace.outward_discharge_m3_per_s).abs()
                    > 1e-7
            );
            assert_eq!(format!("{:?}", origin.snapshot_data()?), before);
        }
    }
    Ok(())
}

#[test]
fn a_lake_at_rest_keeps_its_depth_at_both_ends_and_both_losses() -> Result {
    for mut case in [CASES[0], CASES[3]] {
        case.elevation = 1.0;
        for end in [ChannelEnd::Left, ChannelEnd::Right] {
            for loss in [0.0, 1.0] {
                case.loss = loss;
                let (origin, _) = build(&case, end, (1, 128.0))?;
                let storage = origin.ledger()?.storage_m3;
                let mut current = origin;
                for index in 0..16 {
                    let target = (index + 1) as f64 / 8.0;
                    let held = forcing(&current, target, end, false);
                    current = current.trial_to(target, &held)?.into_next();
                    assert_eq!(current.time().to_bits(), target.to_bits());
                    let state = current.reach(REACH).ok_or("the reach")?.state()?;
                    assert!((state[0].depth_m - 1.0).abs() <= 1e-12);
                    assert!(state[0].discharge_m3_per_s.abs() <= LAKE);
                    assert!(state[0].longitudinal_momentum_m4_per_s.abs() <= LAKE);
                    assert!((current.ledger()?.storage_m3 - storage).abs() <= LAKE);
                    let moved = check_pairing(&current, end, side(end))?;
                    assert!(moved.abs() <= LAKE, "end={end:?}, loss={loss}: {moved:e}");
                }
            }
        }
    }
    Ok(())
}

#[test]
fn a_filling_pool_reverses_without_a_policy_switch_or_transfer_scaling() -> Result {
    let mut case = CASES[0];
    case.coefficient = 1.0;
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
        let held = forcing(&current, 4.0, end, true);
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
