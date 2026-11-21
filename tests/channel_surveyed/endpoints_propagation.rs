//! Refinement of surveyed endpoint propagation against the frozen exact cell averages.
//!
//! Frozen criteria:
//!
//! - Propagation refinement: `endpoints_propagation.rs`: three bands, `20 m`, `g = 1`, initial
//!   depth `1.5 m` at rest, meshes `25/50`, `t = 2 s`. `Stage` to `0.81 m` (rarefaction
//!   across knot 1). `Stage` to `2.25 m` (shock across knot 2). `Discharge` and
//!   `QuiescentReservoir` at the reference trace values. Mean depth and discharge errors
//!   decrease strictly. Finest errors below `35%` of the physical signal. The 50-cell errors
//!   reach about 30% of the signal for a rarefaction and 15% for a shock. Riemann flux
//!   widths `1e-6`. Right-end mirror errors on 25 cells equal left within `1e-8` relative.
//! - Mass, standalone: every standalone row. Each interval: ledger residual within its bound.
//!   `Discharge` face volume equals `RN(σ Q dt)` bit for bit. End face low parts equal `+0.0`.
//!
//! The physical signal is the mean absolute change of the finest exact solution against the
//! unchanged rest state. The right end runs the mirror image: reversed cells, negated discharge.

use super::endpoints_propagation_reference::{Mesh, PROPAGATION, Propagation};
use super::endpoints_reference::Policy;
use super::endpoints_support::{Section, check_interval, step_checked};
use super::support::{TestResult, forcing};
use hydr::channel::{ChannelBoundary, ChannelConfig, ChannelError};

/// Initial depth in metres.
const REST_DEPTH: f64 = 1.5;
/// Frozen meshes in the refinement: 25 and 50 cells.
const REFINED_MESHES: usize = 2;
/// Finest error ceiling as a fraction of the physical signal.
const FINEST_FRACTION: f64 = 0.35;
/// Absolute and relative Riemann flux enclosure widths. They sit far below the
/// discretization errors.
const FLUX_TOLERANCE: f64 = 1e-6;
/// Relative mirror tolerance. The mirror sums faces in reverse order, so the flux controller
/// can stop at another panel level. At the flux width above the mirrored errors differ by
/// under `1e-9` relative.
const MIRROR_RELATIVE: f64 = 1e-8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

/// Mean absolute depth and discharge errors of one run.
#[derive(Clone, Copy, Debug)]
struct Errors {
    depth: f64,
    discharge: f64,
}

fn centre(lower: u64, upper: u64) -> f64 {
    0.5 * (f64::from_bits(lower) + f64::from_bits(upper))
}

/// Policy at the given end. Discharge is global, so the right end negates it.
fn boundary(policy: Policy, side: Side) -> ChannelBoundary {
    match (policy, side) {
        (Policy::Discharge { discharge_m3_per_s }, Side::Right) => ChannelBoundary::Discharge {
            discharge_m3_per_s: -discharge_m3_per_s,
        },
        _ => policy.boundary(),
    }
}

fn measure(case: &Propagation, mesh: &Mesh, side: Side) -> TestResult<Errors> {
    let section = Section::new(case.profile, case.gravity);
    let area = section.area(REST_DEPTH);
    assert_eq!(area.to_bits(), case.interior_area_m2.to_bits());
    let ends = match side {
        Side::Left => [boundary(case.policy, side), ChannelBoundary::Wall],
        Side::Right => [ChannelBoundary::Wall, boundary(case.policy, side)],
    };
    let config = ChannelConfig {
        riemann_water_flux_absolute_m3_per_s: FLUX_TOLERANCE,
        riemann_momentum_flux_absolute_m4_per_s2: FLUX_TOLERANCE,
        riemann_flux_relative: FLUX_TOLERANCE,
        ..section.config()
    };
    let mut current = section.reach_with(mesh.cells, mesh.cell_length_m, area, 0.0, config)?;
    let mut steps = 0;
    while current.time() < case.time_s {
        current = step_checked(&current, case.time_s, &ends)?;
        steps += 1;
        assert!(steps < 1_000_000, "propagation stalls");
    }
    assert_eq!(current.time().to_bits(), case.time_s.to_bits());
    let receipt = current.last_step()?.ok_or("missing propagation receipt")?;
    let traced = if side == Side::Left { 0 } else { 1 };
    assert!(receipt.boundary_traces[traced].is_some());
    assert!(receipt.boundary_traces[1 - traced].is_none());
    let state = current.state()?;
    let mut depth = 0.0;
    let mut discharge = 0.0;
    for (index, cell) in state.iter().enumerate() {
        let (exact, sign) = match side {
            Side::Left => (&mesh.averages[index], 1.0),
            Side::Right => (&mesh.averages[mesh.cells - 1 - index], -1.0),
        };
        depth += (cell.depth_m - centre(exact[4], exact[5])).abs();
        discharge += (cell.discharge_m3_per_s - sign * centre(exact[2], exact[3])).abs();
    }
    let count = mesh.cells as f64;
    Ok(Errors {
        depth: depth / count,
        discharge: discharge / count,
    })
}

/// Mean absolute change of the exact solution against the rest state.
fn signal(mesh: &Mesh) -> Errors {
    let mut depth = 0.0;
    let mut discharge = 0.0;
    for exact in mesh.averages {
        depth += (centre(exact[4], exact[5]) - REST_DEPTH).abs();
        discharge += centre(exact[2], exact[3]).abs();
    }
    let count = mesh.cells as f64;
    Errors {
        depth: depth / count,
        discharge: discharge / count,
    }
}

fn check(name: &str) -> TestResult {
    let case = PROPAGATION
        .iter()
        .find(|case| case.name == name)
        .ok_or("missing propagation reference")?;
    let meshes = &case.meshes[..REFINED_MESHES];
    let physical = signal(&meshes[REFINED_MESHES - 1]);
    assert!(physical.depth > 0.0 && physical.discharge > 0.0);
    let left = meshes
        .iter()
        .map(|mesh| measure(case, mesh, Side::Left))
        .collect::<TestResult<Vec<_>>>()?;
    // The mirror runs the coarse mesh only. Its end code path does not depend on the mesh.
    let mirror = measure(case, &meshes[0], Side::Right)?;
    println!("{name}: signal={physical:?}, left={left:?}, right={mirror:?}");
    for (right, left) in [
        (mirror.depth, left[0].depth),
        (mirror.discharge, left[0].discharge),
    ] {
        assert!(
            (right - left).abs() <= MIRROR_RELATIVE * left.abs(),
            "{name}: mirrored errors differ: right={right:e}, left={left:e}"
        );
    }
    assert!(
        left.windows(2).all(|pair| pair[1].depth < pair[0].depth),
        "{name}: depth refinement: {left:?}"
    );
    assert!(
        left.windows(2)
            .all(|pair| pair[1].discharge < pair[0].discharge),
        "{name}: discharge refinement: {left:?}"
    );
    let finest = left.last().ok_or("missing finest errors")?;
    assert!(
        finest.depth < FINEST_FRACTION * physical.depth,
        "{name}: depth error={:e}, physical signal={:e}",
        finest.depth,
        physical.depth
    );
    assert!(
        finest.discharge < FINEST_FRACTION * physical.discharge,
        "{name}: discharge error={:e}, physical signal={:e}",
        finest.discharge,
        physical.discharge
    );
    Ok(())
}

#[test]
fn stage_rarefaction_refines_toward_the_exact_solution() -> TestResult {
    check("stage-rarefaction")
}

#[test]
fn stage_shock_refines_toward_the_exact_solution() -> TestResult {
    check("stage-shock")
}

#[test]
fn discharge_rarefaction_refines_toward_the_exact_solution() -> TestResult {
    check("discharge-rarefaction")
}

#[test]
fn discharge_shock_refines_toward_the_exact_solution() -> TestResult {
    check("discharge-shock")
}

#[test]
fn reservoir_rarefaction_refines_toward_the_exact_solution() -> TestResult {
    check("reservoir-rarefaction")
}

#[test]
fn reservoir_shock_refines_toward_the_exact_solution() -> TestResult {
    check("reservoir-shock")
}

/// The stage rarefaction trace discharge sits 0.14% under the rest sonic outflow. The end
/// cell average loses that capacity within its first wave crossing, so no subcritical trace
/// exists past a finite time.
///
/// Frozen criteria, 25 cells, left end:
/// - the run rejects within 32 accepted intervals, each with at most 16 retries
/// - the error is `Numerical` and names the sonic outflow trace rejection
/// - the rejecting step leaves the reach state unchanged bit for bit
#[test]
fn discharge_above_the_end_cell_capacity_rejects_in_bounded_retries() -> TestResult {
    let case = PROPAGATION
        .iter()
        .find(|case| case.name == "stage-rarefaction")
        .ok_or("missing propagation reference")?;
    let section = Section::new(case.profile, case.gravity);
    let ends = [
        ChannelBoundary::Discharge {
            discharge_m3_per_s: -1.6968414670944718,
        },
        ChannelBoundary::Wall,
    ];
    let mesh = &case.meshes[0];
    let mut current = section.reach(
        mesh.cells,
        mesh.cell_length_m,
        section.area(REST_DEPTH),
        0.0,
    )?;
    let mut worst = 0;
    for accepted in 0..=32 {
        let before = current.snapshot_data()?;
        let start = current.time();
        match current.step(case.time_s, &forcing(start, case.time_s, ends.clone())) {
            Ok(step) => {
                let receipt = step.next.last_step()?.ok_or("missing receipt")?;
                worst = worst.max(receipt.retries);
                assert!(receipt.retries <= 16, "retries={}", receipt.retries);
                check_interval(&step.next, &ends)?;
                current = step.next;
            }
            Err(error) => {
                println!(
                    "capacity loss: accepted={accepted}, worst retries={worst}, t={start:e}: {error}"
                );
                assert!(
                    matches!(&error, ChannelError::Numerical { context, .. }
                        if context.contains("exceeds the sonic outflow of the end cell")),
                    "{error}"
                );
                assert_eq!(current.snapshot_data()?, before);
                assert_eq!(
                    format!("{:?}", current.snapshot_data()?),
                    format!("{before:?}")
                );
                return Ok(());
            }
        }
    }
    Err("the capacity loss did not reject within 32 accepted intervals".into())
}
