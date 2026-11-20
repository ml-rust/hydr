//! Pumps between pools and surface tanks: flat curve, linear curve, suction cap, and a dry
//! surface cell with an outlet.
//!
//! Frozen criteria, fixed before the first run:
//!
//! | Claim | Frozen criterion |
//! | --- | --- |
//! | Pump transfer, flat curve: plateau `Q0` over the head range, pool to surface tank | `\|ΣV - Q0 T\| <= 8 ε N Q0 T` over `N` intervals |
//! | Pump transfer, linear curve: `Q = Q0 - s Δ`. Exact `Δ(t) = Q0/s - (Q0/s - Δ0) exp(-s k t)`, `k = 1/A_1 + 1/A_2` | Finest relative error `<= 1e-6`, order `>= 1.8` |
//! | Suction cap: the pump drains its pool to the suction bound | No rejection. Final stage in `[z_s, z_s + head_absolute_m]`. No negative storage. |
//! | Pump to a dry surface cell with an outlet | Volume equals the curve at lift `outlet - donor stage` within `8 ε` relative per interval |
//!
//! `ε` is the `f64` machine epsilon. `Δ` is the lift: receiving stage minus donor stage. Common
//! controls: `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts zero.
//! Four durations, each half the last, where an order is named. Every interval checks the mass
//! bound, the pairing, and the composite balance.

use crate::pools_evidence::{
    HEAD_ABSOLUTE_M, advance, assert_order, build, orders, pool_stage, pool_volume, surface_stage,
};
use crate::pools_fixture::laws::pump_on;
use crate::pools_fixture::{
    Case, PoolForcing, Result, manual, pond, pool, pump_law, structure, surface,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Frozen finest relative error of the linear curve.
const FINEST: f64 = 1e-6;

/// Frozen order floor on the two finest pairs.
const ORDER: f64 = 1.8;

/// Frozen rounding multiple of the flat-curve volume and of the outlet-lift volume.
const ROUNDING: f64 = 8.0;

fn running() -> StructureOperatingPoint {
    StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
}

/// Plateau discharge of the flat curve, in m³/s.
const PLATEAU: f64 = 0.1;

#[test]
fn a_flat_curve_pump_moves_its_plateau_volume() -> Result {
    const N: usize = 64;
    const DT: f64 = 0.125;
    let mut case = Case::tanks(0.4, 0.5);
    case.ponds = vec![(pond("source", 0.0, 8.0, 4.0)?, 8.0 * 1.0)];
    case.structures = vec![structure(
        "pump",
        pump_on(&[(-1.0, PLATEAU), (3.0, PLATEAU)], 0.1, None)?,
        DirectionPolicy::AtoBOnly,
        manual(running()),
        (pool("source"), 0.0),
        (surface(), 0.0),
    )];
    let mut batch = build(&case)?;
    let mut total = 0.0;
    for _ in 0..N {
        let (next, step) = advance(&batch, DT, &PoolForcing::default(), &[])?;
        batch = next;
        assert_eq!(step.structures[0].regime, StructureRegime::PumpCurve);
        total += step.structures[0].signed_volume_m3;
    }
    let expected = PLATEAU * (N as f64 * DT);
    let gap = (total - expected).abs();
    let allowance = ROUNDING * f64::EPSILON * N as f64 * expected;
    println!("flat pump: gap={gap:e}, allowance={allowance:e}");
    assert!(gap <= allowance, "gap={gap:e}, allowance={allowance:e}");
    Ok(())
}

/// Plan area of both pools of the linear case, in m².
const LINEAR_AREA_M2: f64 = 2.0;

/// Interval counts of the linear case, each double the last.
const LINEAR_COUNTS: [usize; 4] = [64, 128, 256, 512];

/// Run end of the linear case, in s.
const LINEAR_END_S: f64 = 20.0;

/// Reference discharge at zero lift of `pump_law`, in m³/s.
const Q0: f64 = 0.15;

/// Slope of `pump_law`, in m³/s per m of lift.
const SLOPE: f64 = 0.05;

/// Lift after `n` intervals, relative to the exact lift.
fn linear_error(n: usize) -> Result<f64> {
    let (source, receiver) = (2.5, 2.0);
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![
        (
            pond("receiver", 0.0, LINEAR_AREA_M2, 4.0)?,
            LINEAR_AREA_M2 * receiver,
        ),
        (
            pond("source", 0.0, LINEAR_AREA_M2, 4.0)?,
            LINEAR_AREA_M2 * source,
        ),
    ];
    case.structures = vec![structure(
        "pump",
        pump_law(0.1, None)?,
        DirectionPolicy::AtoBOnly,
        manual(running()),
        (pool("source"), 0.0),
        (pool("receiver"), 0.0),
    )];
    let dt = LINEAR_END_S / n as f64;
    let mut batch = build(&case)?;
    for _ in 0..n {
        let (next, step) = advance(&batch, dt, &PoolForcing::default(), &[])?;
        batch = next;
        assert_eq!(step.structures[0].regime, StructureRegime::PumpCurve);
    }
    assert_eq!(batch.time().to_bits(), LINEAR_END_S.to_bits());
    let lift = pool_stage(&batch, "receiver")? - pool_stage(&batch, "source")?;
    let k = 2.0 / LINEAR_AREA_M2;
    let delta0 = receiver - source;
    let shutoff = Q0 / SLOPE;
    let exact = shutoff - (shutoff - delta0) * (-SLOPE * k * LINEAR_END_S).exp();
    Ok((lift - exact).abs() / exact.abs())
}

#[test]
fn a_linear_curve_pump_follows_its_exact_lift_at_second_order() -> Result {
    let errors = LINEAR_COUNTS
        .map(linear_error)
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    let observed = orders(&errors);
    println!("linear pump: errors={errors:?}, orders={observed:?}");
    assert!(
        errors[errors.len() - 1] <= FINEST,
        "errors={errors:?}, orders={observed:?}"
    );
    assert_order(&errors, ORDER);
    Ok(())
}

/// Suction bound of the draining pump, in m.
const SUCTION_M: f64 = 0.1;

/// The suction case: a small pool pumped into a large one.
fn suction_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![
        (pond("receiver", 0.0, 8.0, 4.0)?, 8.0 * 0.5),
        (pond("source", 0.0, 1.0, 4.0)?, 1.0 * 0.5),
    ];
    case.structures = vec![structure(
        "pump",
        pump_law(SUCTION_M, None)?,
        DirectionPolicy::AtoBOnly,
        manual(running()),
        (pool("source"), 0.0),
        (pool("receiver"), 0.0),
    )];
    Ok(case)
}

/// The suction case run until the source reaches its suction bound, with the number of
/// intervals taken and whether a suction cap bound.
fn drain() -> Result<(crate::pools_fixture::Batch, usize, bool)> {
    let mut batch = build(&suction_case()?)?;
    let mut capped = false;
    for taken in 1..=80 {
        let (next, step) = advance(&batch, 0.125, &PoolForcing::default(), &[])?;
        batch = next;
        capped |= step.structures[0].cap_factor < 1.0;
        for id in ["source", "receiver"] {
            assert!(pool_volume(&batch, id)? >= 0.0, "{id} storage is negative");
        }
        if pool_stage(&batch, "source")? <= SUCTION_M + HEAD_ABSOLUTE_M {
            return Ok((batch, taken, capped));
        }
    }
    Err("the pump never reached its suction bound".into())
}

#[test]
fn a_pump_drains_its_pool_to_the_suction_bound() -> Result {
    let (batch, _, capped) = drain()?;
    assert!(capped, "the suction cap must bind");
    let stage = pool_stage(&batch, "source")?;
    assert!(
        (SUCTION_M..=SUCTION_M + HEAD_ABSOLUTE_M).contains(&stage),
        "final stage={stage:.17e}"
    );
    Ok(())
}

/// The running pump keeps its admission at the bound: no rejection, no negative storage, and the
/// stage stays in `[z_s, z_s + head_absolute_m]`.
#[test]
fn a_running_pump_at_its_suction_bound_takes_further_intervals() -> Result {
    let (mut batch, _, _) = drain()?;
    for _ in 0..8 {
        let (next, _) = advance(&batch, 0.125, &PoolForcing::default(), &[])?;
        batch = next;
        assert!(pool_volume(&batch, "source")? >= 0.0);
        let stage = pool_stage(&batch, "source")?;
        assert!(
            (SUCTION_M..=SUCTION_M + HEAD_ABSOLUTE_M).contains(&stage),
            "stage={stage:.17e}"
        );
    }
    Ok(())
}

#[test]
fn a_pump_into_a_dry_surface_cell_lifts_to_its_outlet() -> Result {
    const OUTLET_M: f64 = 1.0;
    const DT: f64 = 0.125;
    let mut case = Case::tanks(0.0, 0.5);
    case.ponds = vec![(pond("source", 0.0, 8.0, 4.0)?, 8.0 * 1.5)];
    case.structures = vec![structure(
        "pump",
        pump_law(0.1, Some(OUTLET_M))?,
        DirectionPolicy::AtoBOnly,
        manual(running()),
        (pool("source"), 0.0),
        (surface(), 0.0),
    )];
    let mut batch = build(&case)?;
    assert_eq!(surface_stage(&batch, &case)?, 0.0, "the cell starts dry");
    for _ in 0..16 {
        let (next, step) = advance(&batch, DT, &PoolForcing::default(), &[])?;
        batch = next;
        let transfer = &step.structures[0];
        let lift = OUTLET_M - transfer.midpoint_stage_a_m;
        let expected = 0.2 * (3.0 - lift) / 4.0 * DT;
        let relative = (transfer.signed_volume_m3 - expected).abs() / expected;
        assert!(
            relative <= ROUNDING * f64::EPSILON,
            "lift={lift}, volume={:e}, expected={expected:e}, relative={relative:e}",
            transfer.signed_volume_m3
        );
    }
    assert!(surface_stage(&batch, &case)? > 0.0, "the cell took water");
    Ok(())
}
