//! Uniform flow through a reach-end port, subcritical and supercritical, both directions.
//!
//! The `alpha` reach runs from its left end into the east edge of the surface. Its right end
//! is the port. Both children hold one frictionless state on one flat bed, so every face of
//! both children sees two equal states. The open ends hold that state as an exterior
//! reservoir: the reach's left end and the surface's west faces.

use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, CELLS, GRAVITY, REACH_WIDTH, Reach, Scenario, east, last, reach_cells, x_face,
};
use hydr::channel::{ChannelBoundary, ChannelEnd};
use hydr::coupling::SurfaceChannelPortRegime;
use hydr::model::CellState;
use hydr::solver::{BoundaryForcing, ExteriorBoundary, ExteriorState};

/// Bed of the surface and of `alpha`, in m.
const BED: f64 = 1.0;

/// Depth of the uniform state, in m.
const DEPTH: f64 = 0.5;

/// Accepted intervals of every run.
const INTERVALS: usize = 50;

/// Requested duration of every interval, in s. The CFL limit shortens it.
const DT: f64 = 0.1;

/// Frozen bound on the relative change of every `h` and `q`.
const UNIFORM: f64 = 1e-12;

/// Uniform flow at Froude number `froude`. `toward_surface` runs the reach into the surface.
///
/// Reach-positive discharge runs toward the right end, into the surface along `-x`.
pub(crate) fn uniform(froude: f64, toward_surface: bool, surveyed: bool) -> Scenario {
    let speed = froude * (GRAVITY * DEPTH).sqrt();
    let velocity = if toward_surface { speed } else { -speed };
    let q = DEPTH * velocity;
    let mut case = Scenario::rest(BED, BED + DEPTH, BED, vec![east("p")]).with_surveyed(surveyed);
    case.link = false;
    case.alpha = Reach::uniform(BED, DEPTH, q);
    case.surface = vec![
        CellState {
            h: DEPTH,
            qx: -q,
            qy: 0.0,
        };
        CELLS
    ];
    // The west faces point along -x: outward velocity equals the reach velocity.
    case.open = (0..2)
        .map(|row| {
            BoundaryForcing::Exterior(ExteriorBoundary {
                face: x_face(0, row),
                state: ExteriorState::Wet {
                    elevation: BED + DEPTH,
                    outward_velocity: velocity,
                    tangential_velocity: 0.0,
                },
            })
        })
        .collect();
    case.ends = vec![(
        ALPHA,
        ChannelEnd::Left,
        ChannelBoundary::Exterior {
            depth_m: DEPTH,
            velocity_m_per_s: velocity,
        },
    )];
    case
}

/// `|actual - expected|` over `|scale|`.
fn change(actual: f64, expected: f64, scale: f64) -> f64 {
    (actual - expected).abs() / scale.abs()
}

/// Runs one uniform case. Every interval checks the port regime and its exact flux.
pub(crate) fn holds(
    froude: f64,
    toward_surface: bool,
    regime: SurfaceChannelPortRegime,
    surveyed: bool,
) -> Result {
    let name = format!("Fr {froude}, toward surface {toward_surface}, surveyed {surveyed}");
    let case = uniform(froude, toward_surface, surveyed);
    let mut batch = case.build()?;
    let surface = batch.surface().state()?.cells().to_vec();
    let reach = reach_cells(&batch, ALPHA)?;
    let q = surface[0].qx.abs();
    let discharge = REACH_WIDTH * q;
    // Normal momentum leaving the reach: the uniform state's flux, the same both ways.
    let momentum = REACH_WIDTH * (q * q / DEPTH + 0.5 * GRAVITY * DEPTH * DEPTH);
    for _ in 0..INTERVALS {
        batch = case.step(&batch, DT)?;
        let step = last(&batch)?;
        let transfer = &step.ports[0];
        assert_eq!(transfer.regime, regime, "{name}");
        // The port carries the uniform state's own discharge, into the reach when negative.
        let dt = step.end_s - step.start_s;
        let signed = if toward_surface {
            -discharge
        } else {
            discharge
        };
        assert!(
            change(transfer.signed_volume_m3, signed * dt, signed * dt) <= UNIFORM,
            "{name}: {transfer:?}"
        );
        assert!(
            change(
                transfer.reach_momentum_impulse_m4_per_s,
                momentum * dt,
                momentum * dt
            ) <= UNIFORM,
            "{name}: {transfer:?}"
        );
        assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
    }
    assert_eq!(batch.accepted_steps(), INTERVALS);
    for (cell, start) in batch.surface().state()?.cells().iter().zip(&surface) {
        assert!(
            change(cell.h, start.h, start.h) <= UNIFORM,
            "{name}: {cell:?}"
        );
        assert!(change(cell.qx, start.qx, q) <= UNIFORM, "{name}: {cell:?}");
        assert!(change(cell.qy, start.qy, q) <= UNIFORM, "{name}: {cell:?}");
    }
    for (cell, start) in reach_cells(&batch, ALPHA)?.iter().zip(&reach) {
        assert!(
            change(cell.depth_m, start.depth_m, start.depth_m) <= UNIFORM,
            "{name}: {cell:?}"
        );
        assert!(
            change(
                cell.discharge_m3_per_s,
                start.discharge_m3_per_s,
                start.discharge_m3_per_s
            ) <= UNIFORM,
            "{name}: {cell:?}"
        );
    }
    Ok(())
}

#[test]
fn subcritical_uniform_flow_crosses_the_port_unchanged_both_ways() -> Result {
    holds(0.5, true, SurfaceChannelPortRegime::Subcritical, false)?;
    holds(0.5, false, SurfaceChannelPortRegime::Subcritical, false)
}

#[test]
fn supercritical_uniform_flow_crosses_the_port_unchanged_both_ways() -> Result {
    holds(
        2.0,
        true,
        SurfaceChannelPortRegime::SupercriticalToSurface,
        false,
    )?;
    holds(
        2.0,
        false,
        SurfaceChannelPortRegime::SupercriticalToReach,
        false,
    )
}
