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

/// The format-`2` state text in the package format before the levelling cap.
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

/// The state text in the package format before the contraction step rule.
///
/// That format lacks `contraction_target` and `contraction`, and names schema and contract
/// `2`. Where the rule never changes a step, this text reproduces the earlier digest bit for
/// bit.
fn before_contraction(text: &str) -> String {
    let stripped = ["contraction_target", "contraction"]
        .iter()
        .fold(text.to_string(), |text, name| strip_field(&text, name));
    stripped
        .replace(
            "hydr-surface-channel-checkpoint-3",
            "hydr-surface-channel-checkpoint-2",
        )
        .replace(
            "hydr-surface-channel-numr-cpu-3",
            "hydr-surface-channel-numr-cpu-2",
        )
}

/// `text` without every `, name: value` Debug field. A value ends at `,` or ` }`.
fn strip_field(text: &str, name: &str) -> String {
    let marker = format!(", {name}: ");
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(&marker) {
        out.push_str(&rest[..at]);
        let value = &rest[at + marker.len()..];
        let end = value.find([',', '}']).map_or(value.len(), |end| {
            if value[..end].ends_with(' ') {
                end - 1
            } else {
                end
            }
        });
        rest = &value[end..];
    }
    out.push_str(rest);
    out
}

/// Digests a state must also match in earlier package formats.
struct Earlier {
    /// Digest of the text before the contraction step rule.
    contraction: u64,
    /// Digest of that text before the levelling cap, where the cap never binds.
    levelling: Option<u64>,
}

/// Fails unless `batch` matches the frozen `expected` digest and every `earlier` digest.
fn check(batch: &Batch, expected: u64, earlier: Option<Earlier>) -> Result {
    let text = state(batch)?;
    let mut pairs = vec![(digest(&text), expected)];
    if let Some(earlier) = earlier {
        let text = before_contraction(&text);
        pairs.push((digest(&text), earlier.contraction));
        if let Some(frozen) = earlier.levelling {
            pairs.push((digest(&before_levelling(&text)), frozen));
        }
    }
    for (actual, frozen) in pairs {
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
    //
    // The contraction step rule ends that fourth-interval attempt after 3 updates in place of
    // 64. Its measured contraction stays below one, so the retry keeps half the interval, as
    // before. The step partition and every accepted bit stay unchanged, so the text before the
    // rule keeps the earlier digest.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        for link in last(&batch)?.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    let earlier = Earlier {
        contraction: 0xc161_f544_a433_f567,
        levelling: None,
    };
    check(&batch, 0x966d_d2ce_0df9_c27f, Some(earlier))
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
    // Neither the cap nor the contraction step rule changes a step here, so each earlier
    // format keeps its earlier digest.
    let earlier = Earlier {
        contraction: 0x5bc0_6ab1_e4a8_a937,
        levelling: Some(0xa448_16a4_840e_cdee),
    };
    check(&batch, 0x7417_472e_02eb_3880, Some(earlier))
}
