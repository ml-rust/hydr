//! Frozen digests of accepted coupled state guard bit-identical coordinator changes.
//!
//! Each digest is FNV-1a over the Debug text of the package, the composite ledger, and the
//! live surface state. Debug text shows every bit of finite values. A digest changes only when
//! an accepted bit, a step partition, or an iteration count changes.

use crate::fixture::Result;
use crate::restoration::{network, run, state};
use crate::scenario::{Batch, ReachSpec, Spec, forcing, weir};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};
use hydr::forcing::WaterAmount;

/// FNV-1a 64-bit digest of `text`.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Fails unless `batch` matches the frozen `expected` digest.
fn check(batch: &Batch, expected: u64) -> Result {
    let text = state(batch)?;
    let actual = digest(&text);
    if actual != expected {
        return Err(format!(
            "state digest={actual:#018x}, frozen={expected:#018x}: an accepted bit changed"
        )
        .into());
    }
    Ok(())
}

#[test]
fn rectangular_network_state_matches_frozen_digest() -> Result {
    let batch = run(network().build()?, &[0.25, 0.5, 0.75, 1.0])?;
    check(&batch, 0x0745_885b_0f3e_e0cf)
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
    check(&batch, 0xa448_16a4_840e_cdee)
}
