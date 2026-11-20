//! A pond draining over a free weir onto a low reach cell.
//!
//! Frozen criterion, fixed before the first run, "Pond drain over a free weir"):
//! a constant-area pool and a free weir onto a low reach cell that stays below the crest. Exact
//! `h(t) = (h0^{-1/2} + C_w L t / (2A))^{-2}`. At `T`: finest relative error of `h <= 1e-6`,
//! order `>= 1.8` on the two finest pairs.
//!
//! Common controls: `head_absolute_m = 1e-13`, `transfer_absolute_m3 = 1e-12`, relative parts
//! zero. Four durations, each half the last. Every interval checks the regime, the mass bound,
//! the pairing, and the composite balance.

use crate::pools_evidence::{advance, assert_order, build, orders, pool_stage, reach_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, channel, manual, pond, pool, structure, weir_law,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Plan area of the pool, in m².
const AREA_M2: f64 = 2.0;

/// Weir crest above both beds, in m.
const CREST_M: f64 = 1.0;

/// Crest length, in m.
const LENGTH_M: f64 = 1.0;

/// Weir coefficient of the fixture law.
const COEFFICIENT: f64 = 1.7;

/// Initial head over the crest, in m.
const H0_M: f64 = 0.5;

/// Initial stage of the reach cell, in m.
const REACH_M: f64 = 0.1;

/// Run end, in s.
const END_S: f64 = 2.0;

/// Interval counts, each double the last. Every duration is exactly `END_S / n`.
const COUNTS: [usize; 4] = [64, 128, 256, 512];

/// Frozen bound on the finest relative error of the head over the crest.
const FINEST: f64 = 1e-6;

/// Frozen order floor on the two finest pairs.
const ORDER: f64 = 1.8;

/// Exact head over the crest at `t`, in m.
fn exact(t: f64) -> f64 {
    (H0_M.powf(-0.5) + COEFFICIENT * LENGTH_M * t / (2.0 * AREA_M2)).powi(-2)
}

fn case() -> Result<Case> {
    let mut case = Case::tanks(0.0, REACH_M);
    case.ponds = vec![(pond("pond", 0.0, AREA_M2, 3.0)?, AREA_M2 * (CREST_M + H0_M))];
    case.structures = vec![structure(
        "weir",
        weir_law(CREST_M, LENGTH_M)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("pond"), 0.0),
        (channel(), 0.0),
    )];
    Ok(case)
}

/// Relative error of the head over the crest at `END_S` after `n` intervals.
fn error(n: usize) -> Result<f64> {
    let case = case()?;
    let dt = END_S / n as f64;
    let mut batch = build(&case)?;
    for _ in 0..n {
        let (next, step) = advance(&batch, dt, &PoolForcing::default(), &[])?;
        batch = next;
        let regime = step.structures[0].regime;
        assert_eq!(regime, StructureRegime::FreeWeirOverflow, "{step:?}");
        assert!(reach_stage(&batch, &case)? < CREST_M);
    }
    assert_eq!(batch.time().to_bits(), END_S.to_bits());
    let head = pool_stage(&batch, "pond")? - CREST_M;
    let expected = exact(END_S);
    Ok((head - expected).abs() / expected)
}

#[test]
fn a_pond_drains_over_a_free_weir_at_second_order() -> Result {
    let errors = COUNTS.map(error).into_iter().collect::<Result<Vec<_>>>()?;
    let observed = orders(&errors);
    println!("weir drain: errors={errors:?}, orders={observed:?}");
    assert!(
        errors[errors.len() - 1] <= FINEST,
        "errors={errors:?}, orders={observed:?}"
    );
    assert_order(&errors, ORDER);
    Ok(())
}
