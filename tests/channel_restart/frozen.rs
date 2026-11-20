//! Frozen digests of accepted standalone reach state guard bit-identical wave-bound changes.
//!
//! Each digest is FNV-1a over the Debug text of the snapshot package after every interval.
//! Debug text shows every bit of finite values, the step partition, and the CFL receipts.
//!
//! Correctly rounded `exp`, `log` and `pow` bump the channel contract identity to `3`, so every
//! current digest moves. No accepted bit moves: the text mapped back to the schema `1` format
//! still reproduces that format's digest.
//!
//! Secant probes in certified surveyed roots bump the schema to `3` and the contract to `4`. The
//! surveyed digest moves accepted bits, and the other two move only their identities. Each kind
//! also runs with `Midpoint` probes. Its text mapped back to the format before root probes
//! reproduces the digest of contract `3`.

use super::standalone::{Kind, Result, reach_with, step};
use solvr::optimize::CertifiedRootProbe;

/// Accepted intervals per fixture.
const INTERVALS: usize = 4;

/// FNV-1a 64-bit digest of `text`.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The text in the package format before the FluxPort endpoint.
///
/// That format names schema and contract `1`. A trajectory without a port changes no other
/// field, and correctly rounded arithmetic moves no accepted bit of these trajectories, so this
/// text reproduces the earlier digest bit for bit.
fn before_flux_port(text: &str) -> String {
    text.replace("hydr-channel-checkpoint-2", "hydr-channel-checkpoint-1")
        .replace("hydr-channel-1d-numr-cpu-3", "hydr-channel-1d-numr-cpu-1")
}

/// The text in the package format before secant probes in surveyed roots.
///
/// That format names schema `2` and contract `3`. Its root controls hold no probe field. A run
/// with `Midpoint` probes changes no other field, so this text reproduces the digest of that
/// format bit for bit.
fn before_root_probe(text: &str) -> String {
    text.replace("hydr-channel-checkpoint-3", "hydr-channel-checkpoint-2")
        .replace("hydr-channel-1d-numr-cpu-4", "hydr-channel-1d-numr-cpu-3")
        .replace(", probe: Midpoint", "")
}

/// Text of every snapshot over `INTERVALS` accepted intervals of `kind` with `probe`.
fn trajectory(kind: Kind, probe: CertifiedRootProbe) -> Result<String> {
    let mut current = reach_with(kind, probe)?;
    let mut text = String::new();
    for _ in 0..INTERVALS {
        current = step(kind, &current)?;
        text.push_str(&format!("{:?}\n", current.snapshot_data()?));
    }
    Ok(text)
}

/// Each digest, labelled by kind, for comparison in one assertion.
fn labelled(digests: impl IntoIterator<Item = u64>) -> Vec<String> {
    Kind::ALL
        .iter()
        .zip(digests)
        .map(|(kind, digest)| format!("{kind:?}={digest:#018x}"))
        .collect()
}

#[test]
fn standalone_trajectories_match_frozen_digests() -> Result {
    // Under contract `2` the digests were 0x9c81_f00a_7ec3_3dab, 0x06cd_2018_c5ab_bf01 and
    // 0xdda5_9c6f_523c_4f53. Only the contract identity differs.
    let expected = [
        0xccf0_1fd7_d194_4b1f_u64,
        0x75ee_e6d7_de81_7771,
        0x500a_6986_6336_cf26,
    ];
    // Digests of the same text in the schema `1` format.
    let earlier = [
        0x0028_56b5_bf80_d4e3_u64,
        0xb350_5b8d_d3bb_ef49,
        0x6de6_f32e_0966_47f3,
    ];
    // Digests under contract `3`, before secant probes.
    let probed = [
        0x4022_f020_af6f_957f_u64,
        0x1e76_ea4e_68b2_8421,
        0x689d_4dfa_0f9a_c4ed,
    ];
    let found = Kind::ALL
        .iter()
        .map(|&kind| Ok(digest(&trajectory(kind, CertifiedRootProbe::SlackSecant)?)))
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(labelled(found), labelled(expected));
    let texts = Kind::ALL
        .iter()
        .map(|&kind| {
            Ok(before_root_probe(&trajectory(
                kind,
                CertifiedRootProbe::Midpoint,
            )?))
        })
        .collect::<Result<Vec<_>>>()?;
    let unprobed = texts.iter().map(|text| digest(text));
    assert_eq!(labelled(unprobed), labelled(probed));
    let before = texts.iter().map(|text| digest(&before_flux_port(text)));
    assert_eq!(labelled(before), labelled(earlier));
    Ok(())
}
