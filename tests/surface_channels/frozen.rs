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
//!
//! Pools, structure links, pool ends, and operating children change the package format and bump
//! the schema to `5` and the contract to `7`. Each case maps its text back to the format before
//! pools and reproduces the digest of contract `6`, then maps on to the format before ports.
//! No accepted bit moves.
//!
//! Secant probes in certified surveyed roots bump the schema to `6` and the contract to `8`,
//! with channel schema `3` and channel contract `4`. The surveyed case moves accepted bits. It
//! also runs with `Midpoint` probes and maps that text back to the format before root probes,
//! which reproduces the digest of contract `7`. The rectangular case moves only its identities.
//!
//! Culvert structure laws bump the schema to `7`. The contract stays `8`. Each case maps its
//! text back to the format before culverts and reproduces the schema `6` digest, so no accepted
//! bit moves.

use crate::fixture::Result;
use crate::restoration::{network, state};
use crate::scenario::{Batch, ReachSpec, Spec, forcing, last, step, weir};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};
use hydr::forcing::WaterAmount;
use solvr::optimize::CertifiedRootProbe;

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

/// The text in the package format before pools, structure links, and pool ends.
///
/// That format names schema `4` and contract `6`. It holds no pool, structure, pool-end, or
/// operating fields. A case without them changes no other field, so this text reproduces the
/// digest of that format bit for bit.
fn before_pools(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-5",
        "hydr-surface-channel-checkpoint-4",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-7",
        "hydr-surface-channel-numr-cpu-6",
    )
    .replace(
        ", pools: [], pool_support: [], pool_sources: None, pool_inflows: [], storage: LevelPoolConfig { volume_absolute: 1e-9, volume_relative: 1e-12 }, structures: [], pool_ends: [], operating: None",
        "",
    )
    .replace(")], pools: [] }, surface: ", ")] }, surface: ")
    .replace(", pools: None, operating: None, counters: ", ", counters: ")
    .replace(", structures: [], pool_ends: [], pool_inflows: []", "")
    .replace(", pool_sources: None, pool_inflows: []", "")
    .replace(", structures: [], pool_ends: [], pools: []", "")
}

/// The text in the format before secant probes in surveyed roots.
///
/// That format names schema `5` and contract `7`, with channel schema `2` and channel contract
/// `3` in the nested packages. Its root controls hold no probe field. A run with `Midpoint`
/// probes changes no other field, so this text reproduces the digest of that format bit for bit.
fn before_root_probe(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-6",
        "hydr-surface-channel-checkpoint-5",
    )
    .replace(
        "hydr-surface-channel-numr-cpu-8",
        "hydr-surface-channel-numr-cpu-7",
    )
    .replace("hydr-channel-checkpoint-3", "hydr-channel-checkpoint-2")
    .replace("hydr-channel-1d-numr-cpu-4", "hydr-channel-1d-numr-cpu-3")
    .replace(", probe: Midpoint", "")
}

/// The text in the format before culvert structure laws.
///
/// That format names schema `6` and the same contract `8`. Its structure elements hold no
/// culvert law. A case without culverts changes no other field, so this text reproduces the
/// digest of that format bit for bit.
fn before_culverts(text: &str) -> String {
    text.replace(
        "hydr-surface-channel-checkpoint-7",
        "hydr-surface-channel-checkpoint-6",
    )
}

/// Fails unless `batch` matches the frozen `expected` digest, and its text in the format before
/// culverts matches the `uncoupled` digest of schema `6`.
fn check_current(batch: &Batch, expected: u64, uncoupled: u64) -> Result {
    let text = state(batch)?;
    let actual = digest(&text);
    if actual != expected {
        return Err(format!(
            "state digest={actual:#018x}, frozen={expected:#018x}: an accepted bit changed"
        )
        .into());
    }
    let before = digest(&before_culverts(&text));
    if before != uncoupled {
        return Err(format!(
            "digest before culverts={before:#018x}, frozen={uncoupled:#018x}: a field other than the schema identity changed"
        )
        .into());
    }
    Ok(())
}

/// Fails unless the text of `batch` in the format before root probes matches the `probed`
/// digest, its text in the format before pools matches the `current` digest of that format, and
/// its text in the format before ports matches the `earlier` digest of that format.
fn check_earlier(batch: &Batch, probed: u64, current: u64, earlier: u64) -> Result {
    let text = before_root_probe(&before_culverts(&state(batch)?));
    let unprobed = digest(&text);
    if unprobed != probed {
        return Err(format!(
            "digest before root probes={unprobed:#018x}, frozen={probed:#018x}: a field other than the identities and the probe changed"
        )
        .into());
    }
    let text = before_pools(&text);
    let pooled = digest(&text);
    if pooled != current {
        return Err(format!(
            "digest before pools={pooled:#018x}, frozen={current:#018x}: a field other than the identities and pool fields changed"
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
    //
    // The digest under schema `6` was 0x334d_f9db_6dcb_719e. Only the schema identity differs.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        let interval = last(&batch)?;
        assert_eq!(interval.retries, 1, "{interval:?}");
        for link in interval.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    check_current(&batch, 0x2df3_81d7_5702_6f8d, 0x334d_f9db_6dcb_719e)?;
    // The network holds no surveyed reach, so only the identities move.
    check_earlier(
        &batch,
        0x169a_8c05_9590_e7c0,
        0xba64_0fc6_cc65_ea08,
        0xf6d4_a84c_2e66_ef34,
    )
}

#[test]
fn surveyed_flood_state_matches_frozen_digest() -> Result {
    // The digest before the response step was 0xb123_21ad_9264_82d0. The step partition stays
    // unchanged, and the third interval takes fewer updates. Its exchanged volume and heads
    // agree with the earlier ones within the head and transfer budgets.
    //
    // The digest under contract `5` was 0x0511_a7a3_d27d_263f. Only the identities differ.
    //
    // The digest under schema `6` was 0xe58a_2aad_5be8_651f. Only the schema identity differs.
    check_current(
        &flood(CertifiedRootProbe::SlackSecant)?,
        0x7dcb_d4ea_d342_f82a,
        0xe58a_2aad_5be8_651f,
    )?;
    // `Midpoint` probes reproduce the accepted bits before secant probes.
    check_earlier(
        &flood(CertifiedRootProbe::Midpoint)?,
        0x6e58_b9a2_e2d8_5cd6,
        0x84a2_1234_ef25_3316,
        0xcfe8_ea92_adc0_edf6,
    )
}

/// The surveyed flood case after three accepted intervals with `probe` in its root controls.
fn flood(probe: CertifiedRootProbe) -> Result<Batch> {
    let mut spec = Spec::new(1, 1, 1.0, 0.0)
        .reach(
            ReachSpec::new("river", 0.0, &[1.0, 1.0])
                .surveyed()
                .probe(probe)
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
    Ok(batch)
}
