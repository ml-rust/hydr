//! Restart of standalone surveyed reaches under every endpoint policy.
//!
//! Frozen criteria, `resource/surveyed-endpoint-attachments-plan.md` §6:
//!
//! - Restart, standalone: `endpoints_restart.rs`, every policy. Snapshot, restore, import, and
//!   resume match the uninterrupted run bit for bit. A forged trace or reservoir receipt
//!   rejects.
//! - Mass, standalone: every standalone row. Each interval: ledger residual within its bound.
//!   `Discharge` face volume equals `RN(σ Q dt)` bit for bit. End face low parts equal `+0.0`.
//!
//! Every run starts from the three-band rest state and holds each policy at the left end, at the
//! right end, and at both ends. The right end runs the mirror image: negated discharge. A
//! rejected import leaves the origin reach unchanged.

use super::endpoints_support::{Section, step_checked};
use super::support::{Reach, TestResult};
use super::surveyed_profiles::THREE_BANDS;
use hydr::channel::{
    ChannelBoundary, ChannelBoundaryTrace, ChannelFluxPort, ChannelReservoirTrace,
    ChannelSnapshotData,
};

/// Intervals before the snapshot.
const BEFORE: usize = 3;

/// Intervals after the snapshot.
const AFTER: usize = 3;

/// Longest requested interval in seconds.
const SPAN: f64 = 0.25;

/// Cells per reach.
const CELLS: usize = 8;

/// Cell length in metres.
const LENGTH: f64 = 2.5;

/// Initial depth in metres.
const DEPTH: f64 = 1.5;

fn section() -> Section {
    Section::new(THREE_BANDS, 1.0)
}

/// Every endpoint policy. The port enters water into the rest state.
fn policies() -> TestResult<Vec<(&'static str, ChannelBoundary)>> {
    let section = section();
    // Entering water deepens the end cell, so the held bound keeps a wide margin.
    let speed = 4.0 * section.sound(DEPTH);
    let port = ChannelBoundary::FluxPort(ChannelFluxPort::new(
        -0.3,
        section.gravity * section.hydrostatic(DEPTH),
        speed,
        Some(section.invariant(DEPTH)),
        "test provider of an entering surveyed flux",
    )?);
    Ok(vec![
        (
            "stage rarefaction",
            ChannelBoundary::Stage { elevation_m: 0.81 },
        ),
        ("stage shock", ChannelBoundary::Stage { elevation_m: 2.25 }),
        // Leaving flow well below the end cell's sonic outflow. The end-cell average of a
        // fan delivers less outflow than the rest state, so a target near the rest
        // capacity loses its subcritical root.
        (
            "discharge rarefaction",
            ChannelBoundary::Discharge {
                discharge_m3_per_s: -1.0,
            },
        ),
        (
            "discharge shock",
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 5.768193264813303,
            },
        ),
        (
            "reservoir rarefaction",
            ChannelBoundary::QuiescentReservoir {
                elevation_m: 0.81,
                entrance_loss_coefficient: 0.0,
            },
        ),
        (
            "reservoir shock",
            ChannelBoundary::QuiescentReservoir {
                elevation_m: 2.462704508891379,
                entrance_loss_coefficient: 0.0,
            },
        ),
        (
            "reservoir loss",
            ChannelBoundary::QuiescentReservoir {
                elevation_m: 1.9,
                entrance_loss_coefficient: 1.0,
            },
        ),
        ("flux port", port),
    ])
}

/// Equal values with equal bits: the debug text prints each `f64` distinctly.
fn same(left: &ChannelSnapshotData, right: &ChannelSnapshotData) {
    assert_eq!(left, right);
    assert_eq!(format!("{left:?}"), format!("{right:?}"));
}

fn advance(mut current: Reach, count: usize, ends: &[ChannelBoundary; 2]) -> TestResult<Reach> {
    for _ in 0..count {
        current = step_checked(&current, current.time() + SPAN, ends)?;
    }
    Ok(current)
}

fn bump(value: &mut f64) {
    *value = f64::from_bits(value.to_bits() + 1);
}

fn forged_trace(mutation: usize, trace: &mut ChannelBoundaryTrace) -> bool {
    match mutation {
        0 => bump(&mut trace.depth_m),
        1 => bump(&mut trace.area_m2),
        2 => bump(&mut trace.outward_velocity_m_per_s),
        3 => bump(&mut trace.outward_discharge_m3_per_s),
        4 => bump(&mut trace.momentum_flux_m4_per_s2),
        5 => bump(&mut trace.signal_speed_bound_m_per_s),
        6 => bump(&mut trace.velocity_compatibility_abs_bound_m_per_s),
        7 => bump(&mut trace.water_compatibility_abs_bound_m3_per_s),
        8 => bump(&mut trace.momentum_compatibility_abs_bound_m4_per_s2),
        9 => bump(&mut trace.water_flux_width_m3_per_s),
        10 => bump(&mut trace.momentum_flux_width_m4_per_s2),
        11 => trace.cap_compatibility_approximation = !trace.cap_compatibility_approximation,
        12 => trace.root_iterations += 1,
        13 => {
            trace.reservoir = match trace.reservoir.take() {
                Some(_) => None,
                None => Some(ChannelReservoirTrace {
                    depth_lower_m: trace.depth_m,
                    depth_upper_m: trace.depth_m,
                    head_compatibility_abs_bound_m: 0.0,
                    dissipation_head_lower_m: 0.0,
                    dissipation_head_upper_m: 0.0,
                }),
            }
        }
        14..=18 => {
            let Some(certificate) = trace.reservoir.as_mut() else {
                return false;
            };
            match mutation {
                14 => bump(&mut certificate.depth_lower_m),
                15 => bump(&mut certificate.depth_upper_m),
                16 => bump(&mut certificate.head_compatibility_abs_bound_m),
                17 => bump(&mut certificate.dissipation_head_lower_m),
                _ => bump(&mut certificate.dissipation_head_upper_m),
            }
        }
        _ => return false,
    }
    true
}

/// A snapshot with one forged receipt or forcing field at `side`, or `None` past the last one.
fn forged(
    saved: &ChannelSnapshotData,
    side: usize,
    mutation: usize,
) -> Option<ChannelSnapshotData> {
    let mut altered = saved.clone();
    let last = altered.last_interval.as_mut()?;
    match mutation {
        0 => {
            let slot = if side == 0 {
                &mut last.forcing.left
            } else {
                &mut last.forcing.right
            };
            *slot = match slot.clone() {
                ChannelBoundary::Stage { elevation_m } => ChannelBoundary::Stage {
                    elevation_m: elevation_m + 0.01,
                },
                ChannelBoundary::Discharge { discharge_m3_per_s } => ChannelBoundary::Discharge {
                    discharge_m3_per_s: discharge_m3_per_s + 0.01,
                },
                ChannelBoundary::QuiescentReservoir {
                    elevation_m,
                    entrance_loss_coefficient,
                } => ChannelBoundary::QuiescentReservoir {
                    elevation_m: elevation_m + 0.01,
                    entrance_loss_coefficient,
                },
                ChannelBoundary::FluxPort(port) => ChannelBoundary::FluxPort(
                    ChannelFluxPort::new(
                        port.outward_water_rate_m3_per_s() + 0.01,
                        port.outward_momentum_rate_m4_per_s2(),
                        port.signal_speed_m_per_s(),
                        port.exterior_invariant_m_per_s(),
                        "forged flux port",
                    )
                    .ok()?,
                ),
                other => other,
            };
        }
        1 => {
            let slot = &mut last.diagnostics.boundary_traces[side];
            match slot {
                Some(_) => *slot = None,
                None => return None,
            }
        }
        2 => {
            let slot = &mut last.diagnostics.boundary_traces[side];
            if slot.is_none() {
                *slot = Some(ChannelBoundaryTrace {
                    depth_m: 1.0,
                    area_m2: 1.0,
                    outward_velocity_m_per_s: 0.0,
                    outward_discharge_m3_per_s: 0.0,
                    momentum_flux_m4_per_s2: 0.0,
                    signal_speed_bound_m_per_s: 1.0,
                    velocity_compatibility_abs_bound_m_per_s: 0.0,
                    water_compatibility_abs_bound_m3_per_s: 0.0,
                    momentum_compatibility_abs_bound_m4_per_s2: 0.0,
                    water_flux_width_m3_per_s: 0.0,
                    momentum_flux_width_m4_per_s2: 0.0,
                    cap_compatibility_approximation: false,
                    reservoir: None,
                    root_iterations: 0,
                });
            } else {
                return None;
            }
        }
        _ => {
            let trace = last.diagnostics.boundary_traces[side].as_mut()?;
            if !forged_trace(mutation - 3, trace) {
                return None;
            }
        }
    }
    Some(altered)
}

fn run(name: &str, placement: &str, ends: [ChannelBoundary; 2], sides: &[usize]) -> TestResult {
    let section = section();
    let origin = section.reach(CELLS, LENGTH, section.area(DEPTH), 0.0)?;
    let before = origin.snapshot_data()?;
    let current = advance(origin.clone(), BEFORE, &ends)?;
    let uninterrupted = advance(current.clone(), AFTER, &ends)?.snapshot_data()?;
    let saved = current.snapshot_data()?;
    let trusted = origin.restore(&current.snapshot())?;
    let cold = origin.import_snapshot(&saved)?;
    same(&trusted.snapshot_data()?, &saved);
    same(&cold.snapshot_data()?, &saved);
    for restarted in [trusted, cold] {
        let resumed = advance(restarted, AFTER, &ends)?.snapshot_data()?;
        same(&resumed, &uninterrupted);
    }
    same(&origin.snapshot_data()?, &before);
    let mut rejected = 0;
    for &side in sides {
        for mutation in 0.. {
            let Some(altered) = forged(&saved, side, mutation) else {
                if mutation > 2 + 18 {
                    break;
                }
                continue;
            };
            assert!(
                origin.import_snapshot(&altered).is_err(),
                "{name} {placement}: accepted forged receipt field {mutation} at end {side}"
            );
            same(&origin.snapshot_data()?, &before);
            rejected += 1;
        }
    }
    assert!(rejected > 0, "{name} {placement}: nothing forged");
    Ok(())
}

/// The mirror image of a left-end `policy` at the right end.
///
/// Discharge is global, so the right end negates it. The same physical trace then runs at
/// either end. Every other policy is already frame-free.
fn mirrored(policy: &ChannelBoundary) -> ChannelBoundary {
    match policy {
        ChannelBoundary::Discharge { discharge_m3_per_s } => ChannelBoundary::Discharge {
            discharge_m3_per_s: -discharge_m3_per_s,
        },
        other => other.clone(),
    }
}

fn check(select: impl Fn(&str) -> bool) -> TestResult {
    for (name, policy) in policies()? {
        if !select(name) {
            continue;
        }
        let wall = ChannelBoundary::Wall;
        let right = mirrored(&policy);
        run(name, "left", [policy.clone(), wall.clone()], &[0])?;
        run(name, "right", [wall, right.clone()], &[1])?;
        run(name, "both", [policy, right], &[0, 1])?;
    }
    Ok(())
}

#[test]
fn stage_restarts_bit_for_bit_and_rejects_forged_traces() -> TestResult {
    check(|name| name.starts_with("stage"))
}

#[test]
fn discharge_restarts_bit_for_bit_and_rejects_forged_traces() -> TestResult {
    check(|name| name.starts_with("discharge"))
}

#[test]
fn reservoir_restarts_bit_for_bit_and_rejects_forged_receipts() -> TestResult {
    check(|name| name.starts_with("reservoir"))
}

#[test]
fn flux_port_restarts_bit_for_bit_and_rejects_forged_receipts() -> TestResult {
    check(|name| name.starts_with("flux"))
}
