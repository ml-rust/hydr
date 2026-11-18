//! A pond and the surface at rest through a submerged weir, an orifice, and an open gate.
//!
//! Frozen criterion, fixed before the first run (plan §5, "Pond and surface at rest"):
//! 50 intervals: every link volume `0.0` bit for bit, regime `EqualHeads`, stage change
//! `<= 1e-12 m`.
//!
//! The pond and the surface cell stand at one stage of `1 m`. The weir crest, the orifice
//! crown, and the gate aperture all lie below it. Each structure is checked in its own
//! interval receipt, in both lifetime ledger directions, and in the stage of both nodes against
//! the start.

use crate::pools_evidence::{advance, build, pool_stage, surface_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, manual, orifice_law, pond, pool, structure, surface, weir_law,
};
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Frozen interval count.
const INTERVALS: usize = 50;

/// Frozen stage change allowance, in m.
const STAGE_CHANGE_M: f64 = 1e-12;

/// Requested duration of every interval, in s.
const DT: f64 = 0.1;

/// The common stage of the pond and of the surface cell, in m.
const STAGE: f64 = 1.0;

fn case() -> Result<Case> {
    let mut case = Case::tanks(STAGE, 0.5);
    case.ponds = vec![(pond("pond", 0.0, 8.0, 2.0)?, 8.0 * STAGE)];
    let ends = || ((pool("pond"), 0.0), (surface(), 0.0));
    let bidirectional = DirectionPolicy::Bidirectional;
    let ((a, b), (c, d), (e, f)) = (ends(), ends(), ends());
    case.structures = vec![
        structure(
            "weir",
            weir_law(0.5, 1.0)?,
            bidirectional,
            manual(StructureOperatingPoint::Passive),
            a,
            b,
        ),
        structure(
            "orifice",
            orifice_law(0.0, 0.2, true)?,
            bidirectional,
            manual(StructureOperatingPoint::Passive),
            c,
            d,
        ),
        structure(
            "gate",
            crate::pools_fixture::gate_law(0.0)?,
            bidirectional,
            manual(StructureOperatingPoint::GateOpening { opening_m: 0.3 }),
            e,
            f,
        ),
    ];
    Ok(case)
}

#[test]
fn a_pond_at_rest_with_the_surface_exchanges_nothing() -> Result {
    let case = case()?;
    let mut batch = build(&case)?;
    for _ in 0..INTERVALS {
        let (next, step) = advance(&batch, DT, &PoolForcing::default(), &[])?;
        batch = next;
        assert_eq!(step.structures.len(), 3);
        for transfer in &step.structures {
            assert_eq!(
                transfer.signed_volume_m3.to_bits(),
                0.0_f64.to_bits(),
                "{transfer:?}"
            );
            assert_eq!(transfer.regime, StructureRegime::EqualHeads, "{transfer:?}");
        }
        for (label, stage) in [
            ("surface", surface_stage(&batch, &case)?),
            ("pond", pool_stage(&batch, "pond")?),
        ] {
            assert!(
                (stage - STAGE).abs() <= STAGE_CHANGE_M,
                "{label} stage={stage:.17e}"
            );
        }
    }
    assert_eq!(batch.accepted_steps(), INTERVALS);
    for ledger in batch.structure_ledgers()? {
        assert_eq!(ledger.a_to_b_m3.to_bits(), 0.0_f64.to_bits(), "{ledger:?}");
        assert_eq!(ledger.b_to_a_m3.to_bits(), 0.0_f64.to_bits(), "{ledger:?}");
    }
    Ok(())
}
