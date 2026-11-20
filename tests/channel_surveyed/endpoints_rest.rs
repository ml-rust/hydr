//! Lake at rest on standalone surveyed reaches under every endpoint policy.
//!
//! Frozen criteria:
//!
//! - Lake at rest, standalone: `endpoints_rest.rs`: rectangle and three bands, 8 cells at rest,
//!   each of `Stage(η)`, `Discharge(0)`, `QuiescentReservoir(η, k ∈ {0, 1})`, and a `FluxPort`
//!   with zero water and the reference `g I(A)`, at both ends. 50 intervals: every
//!   `|Q| <= 1e-12 m³/s`, stage change `<= 1e-12 m`, trace `|q| <= 1e-12 m³/s`.
//! - Mass, standalone: every standalone row. Each interval: ledger residual within its bound.
//!   `Discharge` face volume equals `RN(σ Q dt)` bit for bit. End face low parts equal `+0.0`.
//!
//! The runs place each policy at the left end, at the right end, and at both ends, at rest
//! depths in each band of the profile.

use super::endpoints_support::{Section, step_checked};
use super::support::TestResult;
use super::surveyed_profiles::{Profile, RECTANGLE_3, THREE_BANDS};
use hydr::channel::ChannelBoundary;

/// Accepted intervals per run.
const INTERVALS: usize = 50;

/// Longest requested interval in seconds.
const SPAN: f64 = 0.25;

/// Cells per reach.
const CELLS: usize = 8;

/// Cell length in metres.
const LENGTH: f64 = 4.0;

/// Gravity in m/s².
const GRAVITY: f64 = 9.81;

/// Frozen allowance of every discharge and stage change.
const LIMIT: f64 = 1e-12;

/// Rest depths: one inside each band of `three-bands`, and the same depths on the rectangle.
const DEPTHS: [f64; 3] = [0.5, 1.5, 2.5];

fn policies(section: &Section, depth: f64) -> TestResult<Vec<(String, ChannelBoundary)>> {
    Ok(vec![
        (
            "stage".into(),
            ChannelBoundary::Stage { elevation_m: depth },
        ),
        (
            "discharge".into(),
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
        ),
        (
            "reservoir k=0".into(),
            ChannelBoundary::QuiescentReservoir {
                elevation_m: depth,
                entrance_loss_coefficient: 0.0,
            },
        ),
        (
            "reservoir k=1".into(),
            ChannelBoundary::QuiescentReservoir {
                elevation_m: depth,
                entrance_loss_coefficient: 1.0,
            },
        ),
        ("flux port".into(), section.rest_port(depth)?),
    ])
}

/// Largest observed change of one run.
#[derive(Default)]
struct Worst {
    discharge: f64,
    stage: f64,
    trace: f64,
}

fn run(profile: Profile) -> TestResult {
    let section = Section::new(profile, GRAVITY);
    let mut worst = Worst::default();
    for depth in DEPTHS {
        let area = section.area(depth);
        for (name, policy) in policies(&section, depth)? {
            let wall = ChannelBoundary::Wall;
            for (placement, ends) in [
                ("left", [policy.clone(), wall.clone()]),
                ("right", [wall.clone(), policy.clone()]),
                ("both", [policy.clone(), policy.clone()]),
            ] {
                let label = format!("{} depth={depth} {name} {placement}", profile.name);
                let mut current = section.reach(CELLS, LENGTH, area, 0.0)?;
                let start = current.state()?;
                for _ in 0..INTERVALS {
                    current = step_checked(&current, current.time() + SPAN, &ends)
                        .map_err(|error| format!("{label}: {error}"))?;
                    for (cell, first) in current.state()?.iter().zip(&start) {
                        let stage = cell.stage_m.ok_or("dry cell at rest")?;
                        let before = first.stage_m.ok_or("dry cell at rest")?;
                        worst.discharge = worst.discharge.max(cell.discharge_m3_per_s.abs());
                        worst.stage = worst.stage.max((stage - before).abs());
                    }
                    let receipt = current.last_step()?.ok_or("missing interval receipt")?;
                    for trace in receipt.boundary_traces.iter().flatten() {
                        worst.trace = worst.trace.max(trace.outward_discharge_m3_per_s.abs());
                    }
                    assert!(
                        worst.discharge <= LIMIT && worst.stage <= LIMIT && worst.trace <= LIMIT,
                        "{label}: discharge={:e}, stage change={:e}, trace={:e}",
                        worst.discharge,
                        worst.stage,
                        worst.trace
                    );
                }
            }
        }
    }
    println!(
        "{}: worst |Q|={:e}, stage change={:e}, trace |q|={:e}",
        profile.name, worst.discharge, worst.stage, worst.trace
    );
    Ok(())
}

#[test]
fn rectangle_lake_at_rest_holds_under_every_endpoint_policy() -> TestResult {
    run(RECTANGLE_3)
}

#[test]
fn three_band_lake_at_rest_holds_under_every_endpoint_policy() -> TestResult {
    run(THREE_BANDS)
}
