//! Frozen digests of accepted coupled state guard bit-identical coordinator changes.
//!
//! Each digest is FNV-1a over the Debug text of the package, the composite ledger, and the
//! live surface state. Debug text shows every bit of finite values. A digest changes only when
//! an accepted bit, a step partition, or an iteration count changes.
//!
//! The plan-area response step changes every linked iteration path, so both cases carry new
//! digests. The digests of earlier package formats no longer apply.

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

/// Fails unless `batch` matches the frozen `expected` digest.
fn check(batch: &Batch, expected: u64) -> Result {
    let actual = digest(&state(batch)?);
    if actual == expected {
        Ok(())
    } else {
        Err(format!(
            "state digest={actual:#018x}, frozen={expected:#018x}: an accepted bit changed"
        )
        .into())
    }
}

#[test]
fn rectangular_network_state_matches_frozen_digest() -> Result {
    // The digest before the response step was 0x66c4_6418_a052_98af. The south links drain the
    // reach into the floodplain toward level heads. The plain update stalled on the first
    // attempt of the fourth interval, so that interval kept half its duration.
    //
    // The response step converges each interval in one update. The first three intervals keep
    // their partition, and their accepted states agree with the earlier ones within the head
    // and transfer budgets. The fourth interval no longer stalls and keeps its step-limited
    // duration.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        let interval = last(&batch)?;
        assert_eq!(interval.retries, 1, "{interval:?}");
        for link in interval.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    check(&batch, 0xf6d4_a84c_2e66_ef34)
}

#[test]
fn surveyed_flood_state_matches_frozen_digest() -> Result {
    // The digest before the response step was 0xb123_21ad_9264_82d0. The step partition stays
    // unchanged, and the third interval takes fewer updates. Its exchanged volume and heads
    // agree with the earlier ones within the head and transfer budgets.
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
    check(&batch, 0xcfe8_ea92_adc0_edf6)
}
