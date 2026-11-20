//! A tide through a bidirectional orifice: reversal and mirror symmetry.
//!
//! Frozen criteria, fixed before the first run:
//!
//! | Claim | Frozen criterion |
//! | --- | --- |
//! | Reversal: reach with a tidal `Exterior` end, pool through a bidirectional orifice | At least 4 sign changes. Pairing and composite mass pass every interval. |
//! | Mirror symmetry: endpoints A and B swapped | Signed volumes negate bit for bit. Every child state bit matches. |
//!
//! The reach cell takes a sinusoidal depth at its right end. A pond joins the cell through a
//! symmetric orifice, fully submerged throughout. A sign change is two consecutive nonzero
//! orifice volumes of opposite sign. Mirror symmetry runs the orifice twice, once from the pond
//! to the cell and once from the cell to the pond. Common controls: `head_absolute_m = 1e-13`,
//! `transfer_absolute_m3 = 1e-12`, relative parts zero. Every interval checks the mass bound,
//! the pairing, and the composite balance.

use crate::pools_evidence::{build, check};
use crate::pools_fixture::{
    Batch, Case, PoolForcing, REACH, Result, channel, forcing, manual, orifice_law, pond, pool,
    structure,
};
use hydr::channel::ChannelBoundary;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint, StructureRegime};

/// Frozen least count of sign changes.
const SIGN_CHANGES: usize = 4;

/// Tidal period, mean depth, and amplitude at the reach end, in s and m.
const TIDE: (f64, f64, f64) = (20.0, 1.0, 0.3);

/// Requested duration of every interval, in s.
const DT: f64 = 0.125;

/// Run end, in s.
const END_S: f64 = 60.0;

/// The pond, the reach end, and the orifice. `mirrored` swaps endpoints A and B.
fn case(mirrored: bool) -> Result<Case> {
    let mut case = Case::tanks(0.0, TIDE.1);
    case.ponds = vec![(pond("pond", 0.0, 8.0, 4.0)?, 8.0 * TIDE.1)];
    let ends = [(pool("pond"), 0.0), (channel(), 0.0)];
    let [a, b] = if mirrored {
        [ends[1].clone(), ends[0].clone()]
    } else {
        ends
    };
    case.structures = vec![structure(
        "orifice",
        orifice_law(0.0, 0.2, true)?,
        DirectionPolicy::Bidirectional,
        manual(StructureOperatingPoint::Passive),
        a,
        b,
    )];
    Ok(case)
}

/// Tide depth at `t`, in m.
fn tide(t: f64) -> f64 {
    TIDE.1 + TIDE.2 * (std::f64::consts::TAU * t / TIDE.0).sin()
}

/// One accepted interval toward at most `end`, the reach end held at the tide of the span's
/// midpoint. Returns the next batch.
fn step(batch: &Batch, end: f64) -> Result<Batch> {
    let mut held = forcing(batch, end, &PoolForcing::default());
    let mid = 0.5 * (batch.time() + end);
    held.reaches[0].right = Some(ChannelBoundary::Exterior {
        depth_m: tide(mid),
        velocity_m_per_s: 0.0,
    });
    Ok(batch.trial_to(end, &held)?.into_next())
}

/// One run: the signed orifice volume of every accepted interval and the final batch.
fn run(mirrored: bool) -> Result<(Vec<f64>, Vec<String>)> {
    let mut batch = build(&case(mirrored)?)?;
    let (mut volumes, mut states) = (Vec::new(), Vec::new());
    for index in 1..=(END_S / DT).round() as usize {
        let end = index as f64 * DT;
        while batch.time() < end {
            batch = step(&batch, end)?;
            check(&batch, &[])?;
            let transfer = &batch.last_step()?.ok_or("an accepted interval")?.structures[0];
            assert!(
                matches!(
                    transfer.regime,
                    StructureRegime::SubmergedOrifice | StructureRegime::EqualHeads
                ),
                "{transfer:?}"
            );
            volumes.push(transfer.signed_volume_m3);
            states.push(format!(
                "{} {:?} {:?} {:?}",
                batch.time(),
                batch.pools().ok_or("the pools")?.states()?,
                batch.reach(REACH).ok_or("the reach")?.state()?,
                batch.surface().state()?
            ));
        }
    }
    Ok((volumes, states))
}

#[test]
fn a_tide_reverses_the_orifice_with_pairing_and_balance_in_every_interval() -> Result {
    let (volumes, _) = run(false)?;
    let mut signs = volumes.iter().filter(|v| **v != 0.0).map(|v| *v > 0.0);
    let mut previous = signs.next().ok_or("an interval with flow")?;
    let mut changes = 0;
    for sign in signs {
        changes += usize::from(sign != previous);
        previous = sign;
    }
    println!(
        "reversal: {changes} sign changes in {} intervals",
        volumes.len()
    );
    assert!(changes >= SIGN_CHANGES, "sign changes={changes}");
    Ok(())
}

#[test]
fn swapping_the_endpoints_negates_every_volume_and_keeps_every_child_state() -> Result {
    let (straight, straight_states) = run(false)?;
    let (mirror, mirror_states) = run(true)?;
    assert_eq!(straight.len(), mirror.len());
    assert!(straight.iter().any(|v| *v > 0.0) && straight.iter().any(|v| *v < 0.0));
    for (index, (a, b)) in straight.iter().zip(&mirror).enumerate() {
        assert_eq!(
            (-a).to_bits(),
            b.to_bits(),
            "interval {index}: {a:e} against {b:e}"
        );
    }
    for (index, (a, b)) in straight_states.iter().zip(&mirror_states).enumerate() {
        assert_eq!(a, b, "interval {index}");
    }
    Ok(())
}
