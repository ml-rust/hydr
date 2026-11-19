//! Surveyed reaches coupled to the surface through bank weirs.

use crate::fixture::{Result, contains, input_rejection};
use crate::restoration::{run, state};
use crate::scenario::{Batch, ReachSpec, Spec, depths, forcing, last, weir};
use hydr::channel::{
    ChannelBoundary, ChannelGeometry, ChannelLateralDirection, ChannelLateralInput,
    CrossSectionPoint,
};
use hydr::coupling::SurfaceChannelStepDiagnostics;
use hydr::forcing::WaterAmount;

const REACH: &str = "river";
const TRIBUTARY: &str = "tributary";
const ABSTRACTION: &str = "abstraction";

/// Iteration allowances for surveyed cases. Surveyed trials cost more than analytic ones.
fn surveyed_config(spec: &mut Spec) {
    spec.config.head_absolute_m = 1e-10;
    spec.config.transfer_absolute_m3 = 1e-10;
}

/// Dry surface cell on bed 1.0 beside a surveyed reach at stage 1.0 behind a 1.05 m crest.
///
/// A tributary floods the reach over the bank. An abstraction then lowers it.
fn floodplain() -> Spec {
    let mut spec = Spec::new(1, 1, 1.0, 0.0)
        .reach(
            ReachSpec::new(REACH, 0.0, &[1.0, 1.0])
                .surveyed()
                .binding(TRIBUTARY, 1, ChannelLateralDirection::Inflow)
                .binding(ABSTRACTION, 1, ChannelLateralDirection::Withdrawal),
        )
        .link(weir("bank", 0, REACH, 0, 1.05, 2.0));
    surveyed_config(&mut spec);
    spec
}

/// Accepts one interval of at most 1 s with the given caller rates.
fn advance_with(batch: &Batch, inflow: f64, withdrawal: f64) -> Result<Batch> {
    let end = batch.time() + 1.0;
    let rate = |rate_m3_per_s| WaterAmount::Rate { rate_m3_per_s };
    let inputs = [
        (
            REACH,
            ChannelLateralInput::Inflow {
                boundary_id: TRIBUTARY.into(),
                amount: rate(inflow),
                longitudinal_velocity_m_per_s: 0.0,
            },
        ),
        (
            REACH,
            ChannelLateralInput::Withdrawal {
                boundary_id: ABSTRACTION.into(),
                amount: rate(withdrawal),
            },
        ),
    ];
    let interval = forcing(batch, end, &inputs);
    Ok(batch.trial_to(end, &interval)?.into_next())
}

/// Fails unless both sides apply the summed link volume and the composite water balances.
fn check_pairing(step: &SurfaceChannelStepDiagnostics) -> Result {
    let signed: f64 = step.links.iter().map(|l| l.signed_volume_m3).sum();
    let surface_net = step.surface_exchange_out_m3 - step.surface_exchange_in_m3;
    let channel_net = step.channel_exchange_in_m3 - step.channel_exchange_out_m3;
    if (surface_net - signed).abs() > step.paired_bound_m3
        || (channel_net - signed).abs() > step.paired_bound_m3
        || step.mass_residual_m3.abs() > step.mass_bound_m3
    {
        return Err(format!("pairing or composite balance out of bounds: {step:?}").into());
    }
    Ok(())
}

#[test]
fn surveyed_reach_overflow_conserves_and_reverses() -> Result {
    let mut batch = floodplain().build()?;
    let mut flooded = false;
    while batch.time() < 60.0 && !flooded {
        batch = advance_with(&batch, 4.0, 0.0)?;
        check_pairing(&last(&batch)?)?;
        flooded = depths(&batch)?[0] > 0.1;
    }
    assert!(flooded, "the surveyed reach must flood the surface");
    let flood = batch.link_ledgers()?[0].channel_to_surface_m3;
    assert!(flood > 0.0);
    let mut receded = false;
    while batch.time() < 120.0 && !receded {
        batch = advance_with(&batch, 0.0, 4.0)?;
        let step = last(&batch)?;
        check_pairing(&step)?;
        receded = step.links[0].signed_volume_m3 > 0.0;
    }
    assert!(
        receded,
        "the floodplain must drain back through the same link"
    );
    let ledger = &batch.link_ledgers()?[0];
    assert!(ledger.channel_to_surface_m3 >= flood && ledger.surface_to_channel_m3 > 0.0);
    let reach = batch
        .reach(REACH)
        .ok_or("surveyed reach is missing")?
        .ledger();
    assert!(
        reach.lateral_in_m3 > 0.0 && reach.lateral_out_m3 > 0.0,
        "{reach:?}"
    );
    let balance = batch.ledger()?;
    assert!(balance.residual_m3.abs() <= balance.bound_m3, "{balance:?}");
    assert!(balance.channel_exchange_in_m3 > 0.0 && balance.channel_exchange_out_m3 > 0.0);
    Ok(())
}

const FIRST: [f64; 2] = [0.1, 0.2];
const SECOND: [f64; 2] = [0.3, 0.4];

#[test]
fn surveyed_reach_restart_is_bitwise() -> Result {
    // Surface stage 1.5 spills over a 1.2 m crest into a surveyed reach at stage 0.5.
    let mut spec = Spec::new(1, 1, 1.0, 0.5)
        .reach(ReachSpec::new(REACH, 0.0, &[0.5, 0.5]).surveyed())
        .link(weir("bank", 0, REACH, 0, 1.2, 2.0));
    surveyed_config(&mut spec);
    let whole = run(run(spec.build()?, &FIRST)?, &SECOND)?;
    let exchanged: f64 = whole
        .link_ledgers()?
        .iter()
        .map(|l| l.surface_to_channel_m3 + l.channel_to_surface_m3)
        .sum();
    assert!(exchanged > 0.0, "the surveyed network must exchange water");
    let middle = run(spec.build()?, &FIRST)?;
    let data = middle.snapshot_data()?;
    let expected = state(&middle)?;
    let restored = spec.build()?.restore(&middle.snapshot())?;
    let imported = spec.build()?.import_snapshot(&data)?;
    let resumed = Batch::resume(
        crate::runtime::client(),
        &data,
        data.surface.setup.forcing.clone(),
    )?;
    for (name, batch) in [
        ("restored", restored),
        ("imported", imported),
        ("resumed", resumed),
    ] {
        assert_eq!(state(&batch)?, expected, "{name} at the checkpoint");
        assert_eq!(batch.snapshot_data()?, data, "{name} package");
        assert_eq!(
            state(&run(batch, &SECOND)?)?,
            state(&whole)?,
            "{name} after continuing"
        );
    }
    Ok(())
}

#[test]
fn surveyed_scalar_and_exterior_endpoints_admit() -> Result {
    let batch = floodplain().build()?;
    let before = batch.snapshot_data()?;
    let held = [
        ChannelBoundary::Stage { elevation_m: 1.0 },
        ChannelBoundary::QuiescentReservoir {
            elevation_m: 1.0,
            entrance_loss_coefficient: 0.0,
        },
        ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: 0.0,
        },
    ];
    for boundary in held {
        let mut interval = forcing(&batch, 0.1, &[]);
        interval.reaches[0].left = Some(boundary);
        batch.trial_to(0.1, &interval)?;
    }
    assert_eq!(batch.snapshot_data()?, before, "a trial changed the batch");
    Ok(())
}

#[test]
fn surveyed_crest_below_section_bed_rejects() -> Result {
    let (mut setup, _) = floodplain().parts()?;
    // A V section whose lowest point sits at 1.3 m, above the 1.05 m crest.
    let ChannelGeometry::PrismaticSurveyed { profile, .. } =
        &mut setup.reaches[0].definition.geometry
    else {
        return Err("the floodplain reach must be surveyed".into());
    };
    profile.points = [(0.0, 5.0), (0.5, 1.3), (1.0, 5.0)]
        .into_iter()
        .map(|(station_m, elevation_m)| CrossSectionPoint {
            station_m,
            elevation_m,
        })
        .collect();
    profile.maximum_stage_m = 5.0;
    let message = input_rejection(setup.clone())?;
    contains(
        &message,
        &[
            "link=\"bank\" crest_elevation_m=1.05",
            "channel bed=1.3",
            "an unobstructed connection is unsupported",
        ],
    )?;
    // A crest at the section minimum admits.
    setup.links[0].crest_elevation_m = 1.3;
    hydr::coupling::SurfaceChannelDefinition::new(setup)?;
    Ok(())
}
