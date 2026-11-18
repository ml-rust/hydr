//! Exchange physics: pairing, reversal, free overflow, dry donors, and still water.

use crate::fixture::Result;
use crate::scenario::{
    Batch, ReachSpec, Spec, advance, depths, forcing, last, step, volumes, weir,
};
use hydr::channel::ChannelBoundary;
use hydr::coupling::SurfaceChannelWeirRegime;

const REACH: &str = "river";

/// Surface at stage 1.5 above two channel cells at stage 0.5 behind 1.2 m crests.
fn spill() -> Spec {
    Spec::new(3, 2, 1.0, 0.5)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5, 0.5, 0.5]))
        .link(weir("bank-0", 0, REACH, 0, 1.2, 2.0))
        .link(weir("bank-1", 1, REACH, 1, 1.2, 2.0))
}

#[test]
fn paired_transfer_conserves_to_bound() -> Result {
    let mut batch = spill().build()?;
    let mut total = 0.0;
    for _ in 0..4 {
        batch = step(&batch, batch.time() + 1.0)?;
        let step = last(&batch)?;
        let signed: f64 = step.links.iter().map(|l| l.signed_volume_m3).sum();
        assert!(
            signed > 0.0,
            "the surface spills into the channel: {step:?}"
        );
        let surface_net = step.surface_exchange_out_m3 - step.surface_exchange_in_m3;
        let channel_net = step.channel_exchange_in_m3 - step.channel_exchange_out_m3;
        assert!(
            (surface_net - signed).abs() <= step.paired_bound_m3,
            "{step:?}"
        );
        assert!(
            (channel_net - signed).abs() <= step.paired_bound_m3,
            "{step:?}"
        );
        assert!(
            step.mass_residual_m3.abs() <= step.mass_bound_m3,
            "{step:?}"
        );
        for link in &step.links {
            assert_eq!(link.donor_factor, 1.0, "no donor limits this spill");
            assert_eq!(link.signed_volume_m3, link.weir_volume_m3);
        }
        total += signed;
    }
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    assert_eq!(balance.surface_exchange_in_m3, 0.0);
    assert_eq!(balance.channel_exchange_out_m3, 0.0);
    let counted: f64 = batch
        .link_ledgers()?
        .iter()
        .map(|l| l.surface_to_channel_m3)
        .sum();
    let bound: f64 = batch
        .link_ledgers()?
        .iter()
        .map(|l| l.roundoff_bound_m3)
        .sum::<f64>()
        + 1e-12 * total;
    assert!((counted - total).abs() <= bound);
    assert!((balance.surface_exchange_out_m3 - total).abs() <= 1e-12 * total);
    assert!((balance.channel_exchange_in_m3 - total).abs() <= 1e-12 * total);
    Ok(())
}

#[test]
fn flood_then_recession_reverses_through_same_link() -> Result {
    let spec = Spec::new(2, 1, 1.0, 0.0)
        .reach(ReachSpec::new(REACH, 0.0, &[1.0, 1.0, 1.0]))
        .link(weir("bank", 0, REACH, 0, 1.2, 2.0));
    let mut batch = spec.build()?;
    // A downstream stage first floods the reach over the bank, then drains it.
    let advance_with = |batch: &Batch, stage: f64| -> Result<Batch> {
        let end = batch.time() + 1.0;
        let mut interval = forcing(batch, end, &[]);
        interval.reaches[0].right = Some(ChannelBoundary::Stage { elevation_m: stage });
        Ok(batch.trial_to(end, &interval)?.into_next())
    };
    let mut flooded = false;
    while batch.time() < 60.0 && !flooded {
        batch = advance_with(&batch, 1.6)?;
        flooded = last(&batch)?.links[0].signed_volume_m3 < 0.0;
    }
    assert!(flooded, "the channel must overtop the bank");
    for _ in 0..4 {
        batch = advance_with(&batch, 1.6)?;
    }
    let flood = batch.link_ledgers()?[0].channel_to_surface_m3;
    assert!(flood > 0.0 && depths(&batch)?[0] > 0.0);
    let mut receded = false;
    let mut stage = 1.6;
    while batch.time() < 240.0 && !receded {
        stage = (stage - 0.02_f64).max(0.6);
        batch = advance_with(&batch, stage)?;
        receded = last(&batch)?.links[0].signed_volume_m3 > 0.0;
    }
    assert!(
        receded,
        "the floodplain must drain back through the same link"
    );
    let ledger = &batch.link_ledgers()?[0];
    assert!(ledger.channel_to_surface_m3 >= flood && ledger.surface_to_channel_m3 > 0.0);
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    Ok(())
}

#[test]
fn dry_floodplain_receives_free_overflow() -> Result {
    let spec = Spec::new(2, 1, 1.0, 0.0)
        .reach(ReachSpec::new(REACH, 0.0, &[1.5, 1.5]))
        .link(weir("bank", 0, REACH, 0, 1.2, 2.0));
    let batch = step(&spec.build()?, 0.05)?;
    let link = &last(&batch)?.links[0];
    assert_eq!(link.regime, SurfaceChannelWeirRegime::Free);
    assert_eq!(link.donor_factor, 1.0);
    assert!(link.signed_volume_m3 < 0.0, "{link:?}");
    let head = link.midpoint_channel_stage_m - 1.2;
    let expected = -1.7 * 2.0 * head.powf(1.5) * 0.05;
    assert!(
        (link.signed_volume_m3 - expected).abs() <= 1e-12 * expected.abs(),
        "{link:?}, expected {expected}"
    );
    let floodplain = depths(&batch)?[0] * 4.0;
    assert!(
        (floodplain + link.signed_volume_m3).abs() <= 1e-12,
        "{floodplain}"
    );
    Ok(())
}

#[test]
fn dry_or_below_crest_donor_transfers_zero() -> Result {
    // Surface stage 1.1 and channel stage 0.5 sit below the 1.2 m crest. Channel cell 1 is dry.
    let below = Spec::new(2, 1, 1.0, 0.1)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5, 0.0]))
        .link(weir("below", 0, REACH, 0, 1.2, 2.0))
        .link(weir("below-dry-channel", 1, REACH, 1, 1.2, 2.0));
    // A dry surface donor at a crest on its bed, and a link with both sides dry.
    let dry = Spec::new(2, 1, 1.0, 0.0)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5]))
        .reach(ReachSpec::new("terrace", 1.0, &[0.0]))
        .link(weir("dry-surface", 0, REACH, 0, 1.0, 2.0))
        .link(weir("dry-both", 1, "terrace", 0, 1.0, 2.0));
    for spec in [below, dry] {
        let batch = advance(spec.build()?, 2.0)?;
        let step = last(&batch)?;
        for link in &step.links {
            assert_eq!(
                link.signed_volume_m3.to_bits(),
                0.0_f64.to_bits(),
                "{link:?}"
            );
            assert_eq!(
                link.regime,
                SurfaceChannelWeirRegime::BelowCrest,
                "{link:?}"
            );
        }
        assert_eq!(step.surface_exchange_in_m3, 0.0);
        assert_eq!(step.surface_exchange_out_m3, 0.0);
        assert_eq!(step.channel_exchange_out_m3, 0.0);
        for ledger in batch.link_ledgers()? {
            assert_eq!(ledger.surface_to_channel_m3, 0.0);
            assert_eq!(ledger.channel_to_surface_m3, 0.0);
        }
    }
    Ok(())
}

#[test]
fn lake_at_rest_level_with_crest_stays_still() -> Result {
    let spec = Spec::new(3, 2, 1.0, 0.2)
        .reach(ReachSpec::new(REACH, 0.0, &[1.2, 1.2, 1.2]))
        .link(weir("bank-0", 0, REACH, 0, 1.2, 2.0))
        .link(weir("bank-1", 1, REACH, 1, 1.2, 2.0));
    let start = spec.build()?;
    let batch = advance(spec.build()?, 3.0)?;
    let bits = |values: Vec<f64>| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(depths(&batch)?), bits(depths(&start)?));
    assert_eq!(bits(volumes(&batch, REACH)?), bits(volumes(&start, REACH)?));
    for link in last(&batch)?.links {
        assert_eq!(link.signed_volume_m3, 0.0);
        assert_eq!(link.regime, SurfaceChannelWeirRegime::BelowCrest);
    }
    Ok(())
}
