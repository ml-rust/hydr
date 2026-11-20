//! A pond filling through a submerged orifice from a closed surface tank.
//!
//! Frozen criterion, fixed before the first run, "Pond fill through an orifice from
//! the surface"): a one-cell closed surface tank and a constant-area pool. Exact
//! `Δh(t) = (sqrt(Δh0) - k t / 2)^2`, `k = C_d a sqrt(2g) (1/A_1 + 1/A_2)`. At `t*/2`: finest
//! relative error `<= 1e-6`, order `>= 1.8`. After `t*`: heads level within the head allowance,
//! no crossing rejection.
//!
//! `t* = 2 sqrt(Δh0) / k` is the time the heads meet. Common controls: `head_absolute_m =
//! 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts zero. Four durations, each half the
//! last. Every interval checks the regime, the mass bound, the pairing, and the composite
//! balance.

use crate::pools_evidence::{
    HEAD_ABSOLUTE_M, OrificeFill, advance, assert_order, build, orders, pool_stage, surface_stage,
};
use crate::pools_fixture::{
    AREA, Batch, Case, PoolForcing, Result, manual, orifice_law, pond, pool, structure, surface,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Plan area of the pool, in m².
const POOL_AREA_M2: f64 = 1.0;

/// Initial stage of the surface tank, in m.
const SURFACE_M: f64 = 1.2;

/// Initial stage of the pool, in m.
const POOL_M: f64 = 1.0;

/// Interval counts up to `t*/2`, each double the last.
const COUNTS: [usize; 4] = [64, 128, 256, 512];

/// Frozen bound on the finest relative error of the head difference.
const FINEST: f64 = 1e-6;

/// Frozen order floor on the two finest pairs.
const ORDER: f64 = 1.8;

fn fill() -> OrificeFill {
    OrificeFill {
        area_1_m2: AREA,
        area_2_m2: POOL_AREA_M2,
        difference_m: SURFACE_M - POOL_M,
    }
}

fn case() -> Result<Case> {
    let mut case = Case::tanks(SURFACE_M, 0.5);
    case.ponds = vec![(pond("pond", 0.0, POOL_AREA_M2, 4.0)?, POOL_AREA_M2 * POOL_M)];
    case.structures = vec![structure(
        "orifice",
        orifice_law(0.0, 0.2, true)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (surface(), 0.0),
        (pool("pond"), 0.0),
    )];
    Ok(case)
}

/// Head difference after `intervals` intervals of `t*/(2n)`, and the batch.
fn run(case: &Case, n: usize, intervals: usize) -> Result<(f64, Batch)> {
    let dt = fill().meeting_time() / 2.0 / n as f64;
    let mut batch = build(case)?;
    for _ in 0..intervals {
        let (next, step) = advance(&batch, dt, &PoolForcing::default(), &[])?;
        batch = next;
        assert!(
            matches!(
                step.structures[0].regime,
                StructureRegime::SubmergedOrifice | StructureRegime::EqualHeads
            ),
            "{step:?}"
        );
    }
    let difference = surface_stage(&batch, case)? - pool_stage(&batch, "pond")?;
    Ok((difference, batch))
}

#[test]
fn a_pond_fills_through_an_orifice_at_second_order() -> Result {
    let case = case()?;
    let mut errors = Vec::new();
    for n in COUNTS {
        let (difference, _) = run(&case, n, n)?;
        let expected = fill().exact(fill().meeting_time() / 2.0);
        errors.push((difference - expected).abs() / expected);
    }
    let observed = orders(&errors);
    println!("orifice fill: errors={errors:?}, orders={observed:?}");
    assert!(
        errors[errors.len() - 1] <= FINEST,
        "errors={errors:?}, orders={observed:?}"
    );
    assert_order(&errors, ORDER);
    Ok(())
}

#[test]
fn the_heads_level_after_the_meeting_time() -> Result {
    let case = case()?;
    for n in COUNTS {
        // Three halves of `t*` in intervals of `t*/(2n)`: the run crosses `t*` and goes on.
        let (difference, batch) = run(&case, n, 3 * n)?;
        assert!(
            difference.abs() <= HEAD_ABSOLUTE_M,
            "n={n}: heads differ by {difference:e}"
        );
        // Heads that meet carry no more water.
        let step = batch.last_step()?.ok_or("an accepted interval")?;
        assert_eq!(
            step.structures[0].signed_volume_m3.to_bits(),
            0.0_f64.to_bits(),
            "n={n}: {step:?}"
        );
    }
    Ok(())
}
