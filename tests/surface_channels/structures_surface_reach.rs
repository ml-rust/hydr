//! A culvert-approximating orifice between a surface tank and a walled reach cell.
//!
//! Frozen criterion, fixed before the first run, "Culvert-approximating orifice
//! between surface and reach"): a one-cell surface tank and a one-cell walled reach, with the
//! same exact solution and criteria as the orifice fill. Exact
//! `Δh(t) = (sqrt(Δh0) - k t / 2)^2`, `k = C_d a sqrt(2g) (1/A_1 + 1/A_2)`. At `t*/2`: finest
//! relative error `<= 1e-6`, order `>= 1.8`. After `t*`: heads level within the head allowance,
//! no crossing rejection.
//!
//! Common controls: `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts
//! zero. Four durations, each half the last. Every interval checks the regime, the mass bound,
//! the pairing at the surface and in the reach, and the composite balance.

use crate::pools_evidence::{
    HEAD_ABSOLUTE_M, OrificeFill, advance, assert_order, build, orders, reach_stage, surface_stage,
};
use crate::pools_fixture::{
    AREA, Batch, Case, PoolForcing, Result, channel, manual, orifice_law, structure, surface,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Initial stage of the surface tank, in m.
const SURFACE_M: f64 = 1.1;

/// Initial stage of the reach cell, in m.
const REACH_M: f64 = 1.05;

/// Interval counts up to `t*/2`, each double the last.
const COUNTS: [usize; 4] = [64, 128, 256, 512];

/// Frozen bound on the finest relative error of the head difference.
const FINEST: f64 = 1e-6;

/// Frozen order floor on the two finest pairs.
const ORDER: f64 = 1.8;

fn fill() -> OrificeFill {
    OrificeFill {
        area_1_m2: AREA,
        area_2_m2: AREA,
        difference_m: SURFACE_M - REACH_M,
    }
}

fn case() -> Result<Case> {
    let mut case = Case::tanks(SURFACE_M, REACH_M);
    case.structures = vec![structure(
        "culvert",
        orifice_law(0.0, 0.2, true)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (surface(), 0.0),
        (channel(), 0.0),
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
    let difference = surface_stage(&batch, case)? - reach_stage(&batch, case)?;
    Ok((difference, batch))
}

#[test]
fn an_orifice_between_surface_and_reach_levels_at_second_order() -> Result {
    let case = case()?;
    let mut errors = Vec::new();
    for n in COUNTS {
        let (difference, _) = run(&case, n, n)?;
        let expected = fill().exact(fill().meeting_time() / 2.0);
        errors.push((difference - expected).abs() / expected);
    }
    let observed = orders(&errors);
    println!("surface reach: errors={errors:?}, orders={observed:?}");
    assert!(
        errors[errors.len() - 1] <= FINEST,
        "errors={errors:?}, orders={observed:?}"
    );
    assert_order(&errors, ORDER);
    Ok(())
}

#[test]
fn the_surface_and_the_reach_level_after_the_meeting_time() -> Result {
    let case = case()?;
    for n in COUNTS {
        // Three halves of `t*` in intervals of `t*/(2n)`: the run crosses `t*` and goes on.
        let (difference, batch) = run(&case, n, 3 * n)?;
        assert!(
            difference.abs() <= HEAD_ABSOLUTE_M,
            "n={n}: heads differ by {difference:e}"
        );
        let step = batch.last_step()?.ok_or("an accepted interval")?;
        assert_eq!(
            step.structures[0].signed_volume_m3.to_bits(),
            0.0_f64.to_bits(),
            "n={n}: {step:?}"
        );
    }
    Ok(())
}
