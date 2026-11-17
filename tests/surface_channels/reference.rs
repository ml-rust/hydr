//! Free-weir drain of one prismatic surface cell against its closed form.
//!
//! The cell holds `A = 4 m²` on a bed at the `1 m` crest, starting `0.25 m` deep. Its
//! stage `H = z_b + V/A` drains through one `2 m` free weir with `C = 1.7 m^(1/2)/s`:
//!
//! `dV/dt = -C·L·(H - z_c)^(3/2)`, so `(H - z_c)^(-1/2) = (H0 - z_c)^(-1/2) + C·L·t/(2A)`.
//!
//! The receiving reach stays below the crest, so every interval is free flow. The midpoint-head
//! law makes each interval one implicit-midpoint step. An independent scalar bisection of that
//! recurrence checks the coordinator, and the closed form gives the continuous error.

use crate::fixture::Result;
use crate::scenario::{ReachSpec, Spec, depths, last, step, weir};
use hydr::coupling::SurfaceChannelWeirRegime;

const AREA: f64 = 4.0;
const CREST: f64 = 1.0;
const LENGTH: f64 = 2.0;
const COEFFICIENT: f64 = 1.7;
const DEPTH: f64 = 0.25;
const END: f64 = 4.0;

/// Closed-form head above the crest at `t`.
fn exact(t: f64) -> f64 {
    (DEPTH.powf(-0.5) + COEFFICIENT * LENGTH * t / (2.0 * AREA)).powi(-2)
}

/// Implicit-midpoint recurrence `h1 = h0 - dt·C·L/A·((h0 + h1)/2)^(3/2)`, by bisection.
fn recurrence(dt: f64, steps: usize) -> f64 {
    let rate = COEFFICIENT * LENGTH / AREA;
    let mut h = DEPTH;
    for _ in 0..steps {
        let (mut low, mut high) = (0.0_f64, h);
        for _ in 0..200 {
            let mid = 0.5 * (low + high);
            if mid + rate * dt * (0.5 * (h + mid)).powf(1.5) > h {
                high = mid;
            } else {
                low = mid;
            }
        }
        h = 0.5 * (low + high);
    }
    h
}

/// Final depth and interval count of the coupled drain with common intervals of `dt`.
fn drain(dt: f64) -> Result<(f64, usize)> {
    let mut spec = Spec::new(1, 1, CREST, DEPTH)
        .reach(ReachSpec::new("river", 0.0, &[0.1; 4]))
        .link(weir("bank", 0, "river", 0, CREST, LENGTH));
    spec.config.maximum_step_s = dt;
    let mut batch = spec.build()?;
    let mut steps = 0;
    while batch.time() < END {
        let target = (batch.time() + dt).min(END);
        batch = step(&batch, target)?;
        assert_eq!(batch.time(), target, "dt={dt}: the interval shortened");
        let link = &last(&batch)?.links[0];
        assert_eq!(link.regime, SurfaceChannelWeirRegime::Free, "{link:?}");
        assert!(link.midpoint_channel_stage_m < CREST, "{link:?}");
        steps += 1;
    }
    Ok((depths(&batch)?[0], steps))
}

#[test]
fn analytical_floodplain_drain_refines() -> Result {
    // Frozen continuous errors |h(4 s) - exact|, in metres, for each common interval.
    let frozen = [
        (0.25, 6.406_410_453_939_36e-5),
        (0.125, 1.600_400_129_925_24e-5),
        (0.0625, 4.000_249_645_502_72e-6),
        (0.031_25, 1.000_015_496_285_81e-6),
        (0.015_625, 2.500_009_677_747_22e-7),
    ];
    let reference = exact(END);
    let mut previous: Option<f64> = None;
    for (dt, frozen_error) in frozen {
        let (depth, steps) = drain(dt)?;
        let discrete = recurrence(dt, steps);
        assert!(
            (depth - discrete).abs() <= 1e-12,
            "dt={dt}: depth={depth:.17e}, recurrence={discrete:.17e}"
        );
        let error = (depth - reference).abs();
        assert!(
            (error - frozen_error).abs() <= 1e-12,
            "dt={dt}: error={error:.17e}, frozen={frozen_error:.17e}"
        );
        // The declared first-order bound, with the midpoint law well inside it.
        assert!(error <= 3e-4 * dt, "dt={dt}: error={error:.17e}");
        if let Some(previous) = previous {
            let ratio = previous / error;
            assert!((3.9..=4.1).contains(&ratio), "dt={dt}: ratio={ratio}");
        }
        previous = Some(error);
    }
    Ok(())
}
