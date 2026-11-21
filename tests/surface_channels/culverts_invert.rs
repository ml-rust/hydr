//! Culvert caps at the ends of a drain: the donor invert and level heads.

use crate::culverts_fixture::RISE_M;
use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, culvert_law, manual, pond, pool, step, structure, transfer,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Requested duration of every interval, in s.
const DT: f64 = 0.5;

/// Inlet invert of the draining pond, in m.
const INVERT_M: f64 = 0.4;

/// Absolute and relative head allowance of the pool fixture.
const ALLOWANCE: f64 = 1e-12;

/// Rise of the draining barrel, in m.
const DRAIN_RISE_M: f64 = 0.6;

/// Plan area of the draining pond, in m².
const POND_M2: f64 = 0.01;

/// A small pond 0.5 m deep drains through a wide culvert with its inlet at `INVERT_M` into a
/// dry 40 m² basin. One interval of the culvert's discharge holds more than the pond's water
/// above the invert.
fn invert_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("basin", 0.0, 40.0, 3.0)?, 0.0),
        (pond("pond", 0.0, POND_M2, 3.0)?, 0.5 * POND_M2),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(DRAIN_RISE_M, 4.0, [INVERT_M, 0.0], false)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("pond"), 0.0),
        (pool("basin"), 0.0),
    )];
    Ok(case)
}

/// Intervals of the draining run: the cap binds in the first, and two more rest at the stop.
const DRAIN_INTERVALS: usize = 3;

/// Frozen criteria, fixed before the first run:
/// - Every one of `DRAIN_INTERVALS` intervals accepts and passes the mass, pairing, and
///   composite checks.
/// - The pond never falls below its inlet invert.
/// - The invert cap binds in some interval before the last: its cap factor falls below one.
/// - Every later interval moves `0.0` bit for bit with cap factor `0.0`.
/// - The pond ends within the invert band `[z, z + 2 allowance(z)]`.
#[test]
fn a_draining_culvert_stops_at_its_inlet_invert() -> Result {
    let mut batch = invert_case()?.build()?;
    let mut capped = false;
    let mut rested = false;
    for index in 0..DRAIN_INTERVALS {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
        let pond = pool_stage(&batch, "pond")?;
        assert!(pond >= INVERT_M, "interval {index}: pond stage={pond:e}");
        let link = transfer(&batch, "culvert")?;
        if capped {
            assert_eq!(
                link.signed_volume_m3.to_bits(),
                0.0f64.to_bits(),
                "{link:?}"
            );
            assert_eq!(link.cap_factor.to_bits(), 0.0f64.to_bits(), "{link:?}");
            rested = true;
        }
        capped |= link.cap_factor < 1.0;
    }
    assert!(capped, "the invert cap never bound");
    assert!(rested, "no interval followed the binding cap");
    let band = 2.0 * (ALLOWANCE + ALLOWANCE * INVERT_M);
    assert!(pool_stage(&batch, "pond")? <= INVERT_M + band);
    Ok(())
}

/// Frozen criteria, fixed before the first run: two ponds level at 1 m, both above the
/// culvert inverts, carry no water over 10 intervals. Each interval reports `EqualHeads` and
/// `0.0` bit for bit.
#[test]
fn level_ponds_rest_at_equal_heads_through_a_culvert() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("east", 0.0, 4.0, 3.0)?, 4.0),
        (pond("west", 0.0, 8.0, 3.0)?, 8.0),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(RISE_M, 10.0, [0.1, 0.0], true)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        (pool("west"), 0.0),
        (pool("east"), 0.0),
    )];
    let mut batch = case.build()?;
    for _ in 0..10 {
        batch = step(&batch, DT, &PoolForcing::default())?;
        check(&batch, &[])?;
        let link = transfer(&batch, "culvert")?;
        assert_eq!(link.regime, StructureRegime::EqualHeads, "{link:?}");
        assert_eq!(
            link.signed_volume_m3.to_bits(),
            0.0f64.to_bits(),
            "{link:?}"
        );
    }
    Ok(())
}
