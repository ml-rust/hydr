//! Frozen digests of accepted standalone reach state guard bit-identical wave-bound changes.
//!
//! Each digest is FNV-1a over the Debug text of the snapshot package after every interval.
//! Debug text shows every bit of finite values, the step partition, and the CFL receipts.

use super::standalone::{Kind, Result, reach, step};

/// Accepted intervals per fixture.
const INTERVALS: usize = 4;

/// FNV-1a 64-bit digest of `text`.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Digest of every snapshot over `INTERVALS` accepted intervals of `kind`.
fn trajectory(kind: Kind) -> Result<u64> {
    let mut current = reach(kind)?;
    let mut text = String::new();
    for _ in 0..INTERVALS {
        current = step(kind, &current)?;
        text.push_str(&format!("{:?}\n", current.snapshot_data()?));
    }
    Ok(digest(&text))
}

#[test]
fn standalone_trajectories_match_frozen_digests() -> Result {
    let expected = [
        0x0028_56b5_bf80_d4e3_u64,
        0xb350_5b8d_d3bb_ef49,
        0x6de6_f32e_0966_47f3,
    ];
    let found = Kind::ALL
        .iter()
        .map(|&kind| trajectory(kind).map(|digest| format!("{kind:?}={digest:#018x}")))
        .collect::<Result<Vec<_>>>()?;
    let expected: Vec<_> = Kind::ALL
        .iter()
        .zip(expected)
        .map(|(kind, digest)| format!("{kind:?}={digest:#018x}"))
        .collect();
    assert_eq!(found, expected);
    Ok(())
}
