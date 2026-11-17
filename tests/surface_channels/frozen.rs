//! Frozen digests of accepted coupled state guard bit-identical coordinator changes.
//!
//! Each digest is FNV-1a over the Debug text of the package, the composite ledger, and the
//! live surface state. Debug text shows every bit of finite values. A digest changes only when
//! an accepted bit, a step partition, or an iteration count changes.

use crate::fixture::Result;
use crate::restoration::{network, state};
use crate::scenario::{Batch, ReachSpec, Spec, forcing, last, step, weir};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};
use hydr::forcing::WaterAmount;

/// FNV-1a 64-bit digest of `text`.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The state text in the package format before the levelling cap.
///
/// That format lacks `level_factor` and names schema and contract `1`. An unbound cap leaves
/// every factor at exactly `1.0`, so this text reproduces the earlier digest bit for bit.
fn before_levelling(text: &str) -> String {
    text.replace("level_factor: 1.0, ", "")
        .replace(
            "hydr-surface-channel-checkpoint-2",
            "hydr-surface-channel-checkpoint-1",
        )
        .replace(
            "hydr-surface-channel-numr-cpu-2",
            "hydr-surface-channel-numr-cpu-1",
        )
}

/// Fails unless `batch` matches the frozen `expected` digest.
///
/// With `before`, the text in the format before levelling must also match that digest.
fn check(batch: &Batch, expected: u64, before: Option<u64>) -> Result {
    let text = state(batch)?;
    let earlier = before.map(|frozen| (digest(&before_levelling(&text)), frozen));
    for (actual, frozen) in earlier.into_iter().chain([(digest(&text), expected)]) {
        if actual != frozen {
            return Err(format!(
                "state digest={actual:#018x}, frozen={frozen:#018x}: an accepted bit changed"
            )
            .into());
        }
    }
    Ok(())
}

#[test]
fn rectangular_network_state_matches_frozen_digest() -> Result {
    // The digest before levelling was 0x0745_885b_0f3e_e0cf. The south links drain the reach
    // into the floodplain toward level heads. The first fixed-point iterates of the third
    // interval use start heads at both ends, and their weir volumes overshoot level, so the cap
    // binds on them. The first attempt of the fourth interval also binds before it exhausts
    // its updates and halves. Every accepted iterate holds a factor of exactly one, but the
    // iteration paths, iteration counts, and the bits they reach change.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        for link in last(&batch)?.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    check(&batch, 0xc161_f544_a433_f567, None)
}

#[test]
fn surveyed_flood_state_matches_frozen_digest() -> Result {
    let mut spec = Spec::new(1, 1, 1.0, 0.0)
        .reach(
            ReachSpec::new("river", 0.0, &[1.0, 1.0])
                .surveyed()
                .binding("tributary", 1, ChannelLateralDirection::Inflow),
        )
        .link(weir("bank", 0, "river", 0, 1.05, 2.0));
    spec.config.head_absolute_m = 1e-10;
    spec.config.transfer_absolute_m3 = 1e-10;
    let mut batch = spec.build()?;
    for _ in 0..3 {
        let end = batch.time() + 1.0;
        let inflow = ChannelLateralInput::Inflow {
            boundary_id: "tributary".into(),
            amount: WaterAmount::Rate { rate_m3_per_s: 4.0 },
            longitudinal_velocity_m_per_s: 0.0,
        };
        let interval = forcing(&batch, end, &[("river", inflow)]);
        batch = batch.trial_to(end, &interval)?.into_next();
    }
    // The cap never binds here, so the text before levelling keeps the earlier digest.
    check(&batch, 0x5bc0_6ab1_e4a8_a937, Some(0xa448_16a4_840e_cdee))
}
