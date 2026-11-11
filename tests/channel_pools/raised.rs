use super::fixture::{
    DATUM, Result, batch, check_water, connection, definition, forcing, near, single,
};
use super::reference::CASES;
use hydr::channel::{ChannelBoundary, ChannelGeometry, ChannelInitialState};
use hydr::coupling::{
    ChannelPoolEndpoint, ChannelPoolForcingInterval, ChannelPoolInitialState, ChannelPoolReach,
    ChannelPoolSetup,
};
use hydr::storage::{LevelPoolCurve, LevelPoolKnot};

// Frozen binary64 operands from the independently reviewed 150-digit raised-attachment oracle.
const CHANNELS: [(&str, f64, f64, f64, f64, f64, f64); 3] = [
    (
        "high",
        2.25,
        0.140625,
        -0.30618621784789724,
        92.89916654111016,
        1672.184997739983,
        -511.99999999999994,
    ),
    (
        "low",
        -16.0,
        13.9375,
        -3.0,
        0.9566517189835575,
        1706.6666666666665,
        -5120.0,
    ),
    (
        "incoming",
        0.0,
        2.4375,
        1.5,
        10.94017094017094,
        3413.3333333333335,
        5120.0,
    ),
];

fn raised(
    incoming: bool,
) -> Result<(
    ChannelPoolSetup,
    ChannelPoolInitialState,
    ChannelPoolForcingInterval,
)> {
    let (mut setup, mut initial) = single(CASES[0], ChannelPoolEndpoint::Right)?;
    setup.reaches.clear();
    setup.connections.clear();
    initial.reaches.clear();
    setup.pools[0].curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                stage_m: -16.0,
                volume_m3: 0.0,
            },
            LevelPoolKnot {
                stage_m: 2.25,
                volume_m3: 9.0,
            },
            LevelPoolKnot {
                stage_m: 2.5,
                volume_m3: 10.0,
            },
            LevelPoolKnot {
                stage_m: 4.0,
                volume_m3: 16.0,
            },
        ],
        DATUM.into(),
        "Reviewed original-water cuts with a supported lower attachment".into(),
    )?;
    initial.pools[0].volume_m3 = 10.0;
    let count = if incoming { 3 } else { 2 };
    for &(id, bed, _, _, width, volume, momentum) in &CHANNELS[..count] {
        let mut reach = definition(false, 1.0, width, 1, 128.0, bed);
        if let ChannelGeometry::Rectangular { cells } = &mut reach.geometry {
            cells[0].maximum_depth_m = 32.0;
        }
        setup.reaches.push(ChannelPoolReach {
            id: id.into(),
            definition: reach,
        });
        setup
            .connections
            .push(connection(id, id, ChannelPoolEndpoint::Right, "pool", 0.0));
        initial.reaches.push(ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![volume],
            longitudinal_momentum_m4_per_s: vec![momentum],
        });
    }
    let mut held = forcing(&setup, 0.0, 0.125);
    for (record, &(_, _, depth, velocity, _, _, _)) in
        held.reaches.iter_mut().zip(&CHANNELS[..count])
    {
        record.left = Some(ChannelBoundary::Exterior {
            depth_m: depth,
            velocity_m_per_s: velocity,
        });
    }
    Ok((setup, initial, held))
}

#[test]
fn raised_attachment_cuts_admit_original_lower_water_when_final_support_remains_wet() -> Result {
    let (mut setup, initial, held) = raised(true)?;
    setup.config.retry_limit = 0;
    let origin = batch(setup, initial)?;
    let before = origin.snapshot_data()?;
    let next = origin.trial_to(0.125, &held)?.into_next();
    assert_eq!(next.time().to_bits(), 0.125_f64.to_bits());
    let step = next
        .last_step()?
        .ok_or("Missing supported raised-attachment receipt")?;
    assert_eq!(step.retries, 0);
    assert!(step.iterations <= 64);
    near(next.pools().states()?[0].volume_m3, 9.5);
    let end_stage = next.pools().outputs()?[0]
        .stage_m
        .ok_or("Supported raised pool dries")?;
    near(end_stage, 2.375);
    assert!(end_stage > 2.25);
    assert!(step.iteration.attachment_wet.iter().all(|wet| *wet));
    for (id, depth, velocity, discharge, transfer) in [
        ("high", 0.140625, -0.30618621784789724, -4.0, -0.5),
        ("low", 13.9375, -3.0, -40.0, -5.0),
        ("incoming", 2.4375, 1.5, 40.0, 5.0),
    ] {
        let paired = step
            .transfers
            .iter()
            .find(|t| t.connection_id == id)
            .ok_or("Missing raised transfer")?;
        near(paired.signed_volume_m3, transfer);
        near(paired.midpoint_pool_elevation_m, 2.4375);
        let child = next
            .reach(id)
            .ok_or("Missing raised reach")?
            .last_step()?
            .ok_or("Missing raised child receipt")?;
        let trace = child.boundary_traces[1]
            .as_ref()
            .ok_or("Missing raised reservoir trace")?;
        near(trace.depth_m, depth);
        near(trace.outward_velocity_m_per_s, velocity);
        near(trace.outward_discharge_m3_per_s, discharge);
        assert!(trace.reservoir.is_some());
        assert_eq!(
            paired.signed_volume_m3.to_bits(),
            child.face_volume_m3[1].to_bits()
        );
    }
    assert_eq!(step.donor_cuts.len(), 3);
    for (bed, cut_volume, export) in [
        (2.25, 9.0, 0.5),
        (0.0, 576.0 / 73.0, 0.5),
        (-16.0, 0.0, 5.5),
    ] {
        let cut = step
            .donor_cuts
            .iter()
            .find(|c| c.bed_elevation_m == bed)
            .ok_or("Missing raised original-water cut")?;
        let original_available = 10.0 - cut_volume;
        let available_floor = if bed == 0.0 {
            // Greatest binary64 value below the independent exact available volume 154/73.
            f64::from_bits(4611932791009709575)
        } else {
            original_available
        };
        assert!(cut.available_lower_m3 <= available_floor);
        near(cut.available_lower_m3, original_available);
        near(cut.outgoing_upper_m3, export);
        assert!(cut.outgoing_upper_m3 <= cut.available_lower_m3);
        let actual_exports: f64 = step
            .transfers
            .iter()
            .filter(|t| {
                CHANNELS.iter().any(|(id, attachment_bed, _, _, _, _, _)| {
                    *id == t.connection_id && *attachment_bed >= bed
                })
            })
            .map(|t| (-t.signed_volume_m3).max(0.0))
            .sum();
        assert!(actual_exports <= cut.outgoing_upper_m3);
    }
    check_water(&next)?;
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}

#[test]
fn raised_attachment_support_crossing_rejects_or_shortens_the_complete_common_interval() -> Result {
    let (setup, initial, held) = raised(false)?;
    let origin = batch(setup, initial)?;
    let before = origin.snapshot_data()?;
    if let Ok(trial) = origin.trial_to(0.125, &held) {
        let next = trial.into_next();
        assert!(next.time() > 0.0 && next.time() < 0.125);
        assert!(
            next.pools().outputs()?[0]
                .stage_m
                .ok_or("Shortened raised pool dries")?
                > 2.25
        );
        let step = next
            .last_step()?
            .ok_or("Missing shortened raised receipt")?;
        assert!(step.iteration.attachment_wet.iter().all(|wet| *wet));
        check_water(&next)?;
    }
    assert_eq!(origin.snapshot_data()?, before);
    Ok(())
}
