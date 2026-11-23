//! Frozen digests of accepted coupled state guard bit-identical coordinator changes.
//!
//! Each digest is FNV-1a over the Debug text of the package, the composite ledger, and the
//! live surface state. Every schema or contract bump maps its text back through earlier formats
//! and reproduces each earlier frozen digest bit for bit.

use crate::fixture::Result;
use crate::restoration::{network, state};
use crate::scenario::{Batch, ReachSpec, Spec, forcing, last, step, weir};
use hydr::channel::{ChannelLateralDirection, ChannelLateralInput};
use hydr::forcing::WaterAmount;
use solvr::optimize::CertifiedRootProbe;

#[path = "frozen_formats.rs"]
mod formats;

use formats::{
    before_blockage, before_channel_section_epochs, before_culvert_roots, before_culverts,
    before_geometry_schedules, before_pool_curve_epochs, before_pools, before_ports,
    before_root_probe, before_runoff, before_runoff_reports, before_surface_bed_epochs,
    before_surface_runoff,
};

/// FNV-1a 64-bit digest of `text`.
fn digest(text: &str) -> u64 {
    text.bytes().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// Fails unless `batch` matches the frozen `expected` digest, its text in the format before
/// surface bed epochs matches `fixed_bed` (schema `11`, contract `13`), and each earlier format
/// reproduces its frozen digest.
fn check_current(
    batch: &Batch,
    [
        expected,
        fixed_sections,
        fixed_bed,
        fixed_pools,
        unscheduled,
        unconditioned,
        celled,
        unreported,
        unattached,
        cold,
        uncoupled,
    ]: [u64; 11],
) -> Result {
    let text = state(batch)?;
    let actual = digest(&text);
    if actual != expected {
        return Err(format!(
            "state digest={actual:#018x}, frozen={expected:#018x}: an accepted bit changed"
        )
        .into());
    }
    let text = before_channel_section_epochs(&text);
    let sec_fixed = digest(&text);
    if sec_fixed != fixed_sections {
        return Err(format!(
            "digest before channel section epochs={sec_fixed:#018x}, frozen={fixed_sections:#018x}: a field other than the identities and empty channel section fields changed"
        )
        .into());
    }
    let text = before_surface_bed_epochs(&text);
    let bed_fixed = digest(&text);
    if bed_fixed != fixed_bed {
        return Err(format!(
            "digest before surface bed epochs={bed_fixed:#018x}, frozen={fixed_bed:#018x}: a field other than the identities and empty surface bed fields changed"
        )
        .into());
    }
    let text = before_pool_curve_epochs(&text);
    let fixed = digest(&text);
    if fixed != fixed_pools {
        return Err(format!(
            "digest before pool curve epochs={fixed:#018x}, frozen={fixed_pools:#018x}: a field other than the identities and empty pool curve fields changed"
        )
        .into());
    }
    let text = before_geometry_schedules(&text);
    let plain = digest(&text);
    if plain != unscheduled {
        return Err(format!(
            "digest before geometry schedules={plain:#018x}, frozen={unscheduled:#018x}: a field other than the identities and empty geometry schedule fields changed"
        )
        .into());
    }
    let text = before_blockage(&text);
    let unblocked = digest(&text);
    if unblocked != unconditioned {
        return Err(format!(
            "digest before structure conditions={unblocked:#018x}, frozen={unconditioned:#018x}: a field other than the schema identity and the element conditions changed"
        )
        .into());
    }
    let text = before_surface_runoff(&text);
    let uncelled = digest(&text);
    if uncelled != celled {
        return Err(format!(
            "digest before surface runoff={uncelled:#018x}, frozen={celled:#018x}: a field other than the contract identity changed"
        )
        .into());
    }
    let text = before_runoff_reports(&text);
    let reported = digest(&text);
    if reported != unreported {
        return Err(format!(
            "digest before runoff reports={reported:#018x}, frozen={unreported:#018x}: a field other than the contract identity and runoff report fields changed"
        )
        .into());
    }
    let text = before_runoff(&text);
    let detached = digest(&text);
    if detached != unattached {
        return Err(format!(
            "digest before runoff={detached:#018x}, frozen={unattached:#018x}: a field other than the schema identity and runoff fields changed"
        )
        .into());
    }
    let text = before_culvert_roots(&text);
    let rooted = digest(&text);
    if rooted != cold {
        return Err(format!(
            "digest before warm culvert roots={rooted:#018x}, frozen={cold:#018x}: a field other than the contract identity changed"
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
    let text = before_root_probe(&before_culverts(&before_culvert_roots(&before_runoff(
        &before_runoff_reports(&before_surface_runoff(&before_blockage(
            &before_geometry_schedules(&before_pool_curve_epochs(&before_surface_bed_epochs(
                &before_channel_section_epochs(&state(batch)?),
            ))),
        ))),
    ))));
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
    //
    // The digest under contract `8` was 0x2df3_81d7_5702_6f8d. Only the contract identity
    // differs.
    //
    // The digest under schema `7` was 0xd23e_fe25_5262_7b60. Only the schema identity and the
    // empty runoff fields differ.
    //
    // The digest under contract `9` was 0x0067_7624_eaa1_05ab. Only the contract identity and
    // the empty runoff report fields differ.
    //
    // The digest under contract `10` was 0xb12b_e0a2_7fe0_64b2. Only the contract identity
    // differs.
    //
    // The digest under schema `8` was 0x8fcb_d6a0_2974_076b. Only the schema identity differs.
    let mut batch = network().build()?;
    for target in [0.25, 0.5, 0.75, 1.0] {
        batch = step(&batch, target)?;
        let interval = last(&batch)?;
        assert_eq!(interval.retries, 1, "{interval:?}");
        for link in interval.links {
            assert_eq!(link.level_factor, 1.0, "{link:?}");
        }
    }
    check_current(
        &batch,
        [
            0x8376_6354_ae1a_8b39,
            0x38c7_85ff_5358_c7dd,
            0x3a75_1930_fa6d_cd55,
            0x83fa_9750_3ab2_226d,
            0x47b3_41e0_99b9_461e,
            0x8fcb_d6a0_2974_076b,
            0xb12b_e0a2_7fe0_64b2,
            0x0067_7624_eaa1_05ab,
            0xd23e_fe25_5262_7b60,
            0x2df3_81d7_5702_6f8d,
            0x334d_f9db_6dcb_719e,
        ],
    )?;
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
    //
    // The digest under contract `8` was 0x7dcb_d4ea_d342_f82a. Only the contract identity
    // differs.
    //
    // The digest under schema `7` was 0xd6c2_0242_bcfc_f029. Only the schema identity and the
    // empty runoff fields differ.
    //
    // The digest under contract `9` was 0xbc3d_6695_a90a_4510. Only the contract identity and
    // the empty runoff report fields differ.
    //
    // The digest under contract `10` was 0xf5e4_4c35_e68d_2a65. Only the contract identity
    // differs.
    //
    // The digest under schema `8` was 0x1f01_27b4_850f_2fae. Only the schema identity differs.
    check_current(
        &flood(CertifiedRootProbe::SlackSecant)?,
        [
            0xef04_a404_8666_2b4e,
            0x621a_83c5_98de_558a,
            0xefa5_7c00_c7cc_83a2,
            0x3f11_41bd_1f95_3f66,
            0x574f_42de_1061_7391,
            0x1f01_27b4_850f_2fae,
            0xf5e4_4c35_e68d_2a65,
            0xbc3d_6695_a90a_4510,
            0xd6c2_0242_bcfc_f029,
            0x7dcb_d4ea_d342_f82a,
            0xe58a_2aad_5be8_651f,
        ],
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
