//! Frozen digests of accepted coupled state guard bit-identical coordinator changes.
//!
//! Each digest is FNV-1a over the Debug text of the package, the composite ledger, and the
//! live surface state. Debug text shows every bit of finite values. A digest changes only when
//! an accepted bit, a step partition, or an iteration count changes.
//!
//! The plan-area response step changes every linked iteration path, so both cases carry new
//! digests. The digests of earlier package formats no longer apply.
//!
//! Reach-end ports change the package format again. Each case also maps its text back to the
//! format before ports and reproduces that format's digest.
//!
//! Correctly rounded `exp`, `log` and `pow` bump the surface/channel, channel and surface solver
//! contract identities, so both current digests move. No accepted bit moves in either case: the
//! text mapped back to the format before ports still reproduces that format's digest.

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

/// The text in the package format before reach-end ports.
///
/// That format names schema `3` and contract `4`, with channel contract `2` and surface solver
/// contract `1` in the nested packages. It holds no port fields, and its reach forcing holds
/// each endpoint policy without `Some`. A case without ports changes no other field, so this
/// text reproduces the earlier digest bit for bit.
fn before_ports(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-4",
        "hydr-surface-channel-checkpoint-3",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-6",
        "hydr-surface-channel-numr-cpu-4",
    )
    .replace("hydr-channel-1d-numr-cpu-3", "hydr-channel-1d-numr-cpu-2")
    .replace("hydr-swe-numr-cpu-2", "hydr-swe-numr-cpu-1")
    .replace(
        ", ports: SurfaceChannelPortCounterData { ledgers: [], last_signed_volume_m3: [], surface_in_m3: 0.0, surface_out_m3: 0.0, paired_bound_m3: 0.0 }",
        "",
    )
    .replace(", ports: [], port_paired_bound_m3: 0.0", "")
    .replace(
        "], ports: [], storage_exclusion_evidence",
        "], storage_exclusion_evidence",
    )
    .replace("left: Some(Wall)", "left: Wall")
    .replace("right: Some(Wall)", "right: Wall")
}

/// Fails unless `batch` matches the frozen `expected` digest, and its text in the format before
/// ports matches the `earlier` digest of that format.
fn check(batch: &Batch, expected: u64, earlier: u64) -> Result {
    let text = state(batch)?;
    let actual = digest(&text);
    if actual != expected {
        return Err(format!(
            "state digest={actual:#018x}, frozen={expected:#018x}: an accepted bit changed"
        )
        .into());
    }
    let before = digest(&before_ports(&text));
    if before != earlier {
        return Err(format!(
            "digest before ports={before:#018x}, frozen={earlier:#018x}: a field other than the identities and port fields changed"
        )
        .into());
    }
    Ok(())
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
    //
    // The digest under contract `5` was 0x6ac9_8aad_15d3_f3a7. Only the identities differ.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        let interval = last(&batch)?;
        assert_eq!(interval.retries, 1, "{interval:?}");
        for link in interval.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    check(&batch, 0xba64_0fc6_cc65_ea08, 0xf6d4_a84c_2e66_ef34)
}

#[test]
fn surveyed_flood_state_matches_frozen_digest() -> Result {
    // The digest before the response step was 0xb123_21ad_9264_82d0. The step partition stays
    // unchanged, and the third interval takes fewer updates. Its exchanged volume and heads
    // agree with the earlier ones within the head and transfer budgets.
    //
    // The digest under contract `5` was 0x0511_a7a3_d27d_263f. Only the identities differ.
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
    check(&batch, 0x84a2_1234_ef25_3316, 0xcfe8_ea92_adc0_edf6)
}
