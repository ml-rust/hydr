//! Two ponds drain into each other through a drowned culvert.
//!
//! Frozen criteria, fixed before the first run, in `resource/evidence/culverts/drain.json`:
//! the relative error of the upper pond's drop at `END_S` stays within `FINEST_RELATIVE` at the
//! finest count, and the order reaches `ORDER` on both pairs. Every interval reports outlet
//! control and passes the mass, pairing, and composite checks.

use crate::drain_rows::{
    AREA_DOWN, AREA_UP, COUNTS, END_DROP_UP, END_S, FINEST_RELATIVE, INVERT_DOWN, INVERT_UP,
    LENGTH, ORDER, RISE, STAGE_DOWN, STAGE_UP,
};
use crate::pools_evidence::{advance, assert_order, build, orders, pool_stage};
use crate::pools_fixture::{Case, PoolForcing, Result, culvert_law, manual, pond, pool, structure};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

fn bits(value: u64) -> f64 {
    f64::from_bits(value)
}

/// The two ponds and the culvert of the frozen reference.
pub fn case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (
            pond("down", 0.0, bits(AREA_DOWN), 3.0)?,
            bits(AREA_DOWN) * bits(STAGE_DOWN),
        ),
        (
            pond("up", 0.0, bits(AREA_UP), 3.0)?,
            bits(AREA_UP) * bits(STAGE_UP),
        ),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(
            bits(RISE),
            bits(LENGTH),
            [bits(INVERT_UP), bits(INVERT_DOWN)],
            true,
        )?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("up"), 0.0),
        (pool("down"), 0.0),
    )];
    Ok(case)
}

/// Relative error of the upper pond's drop at `END_S` after `n` intervals.
fn error(n: usize) -> Result<f64> {
    let dt = bits(END_S) / n as f64;
    let mut batch = build(&case()?)?;
    for _ in 0..n {
        let (next, step) = advance(&batch, dt, &PoolForcing::default(), &[])?;
        batch = next;
        let transfer = step.structures.first().ok_or("the culvert receipt")?;
        assert_eq!(
            transfer.regime,
            StructureRegime::CulvertOutletControl,
            "{transfer:?}"
        );
        assert!(transfer.signed_volume_m3 > 0.0, "{transfer:?}");
    }
    assert_eq!(batch.time().to_bits(), END_S);
    let drop = bits(STAGE_UP) - pool_stage(&batch, "up")?;
    let expected = 0.5 * (bits(END_DROP_UP[0]) + bits(END_DROP_UP[1]));
    Ok((drop - expected).abs() / expected)
}

#[test]
fn two_ponds_drain_through_a_drowned_culvert_at_second_order() -> Result {
    let errors = COUNTS.map(error).into_iter().collect::<Result<Vec<_>>>()?;
    let observed = orders(&errors);
    println!("culvert drain: errors={errors:?}, orders={observed:?}");
    assert!(
        errors[errors.len() - 1] <= FINEST_RELATIVE,
        "errors={errors:?}, orders={observed:?}"
    );
    assert_order(&errors, ORDER);
    Ok(())
}
