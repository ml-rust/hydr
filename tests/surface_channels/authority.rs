//! Surface/channel coordinator parameter authority table, gate-range conflicts against actuated
//! gates, mixed-owner + geometry schedule bitwise restart, and control signals on culvert nodes.

use crate::pools_evidence::check;
use crate::pools_fixture::{
    Case, PoolForcing, Result, controlled_gate, controlled_pump, culvert_law, gate_law, manual,
    operating, pond, pool, pump_law, signal, step, structure, trajectory, transfer, weir_law,
};
use crate::runtime::client;
use hydr::channel::{
    ChannelGeometry, ChannelSectionEpoch, ChannelSectionSchedule, RectangularChannelCell,
};
use hydr::coupling::{
    SurfaceChannelBatch, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelOperationOwner, surface_channel_capabilities,
};
use hydr::solver::{SurfaceBedEpoch, SurfaceBedSchedule, SurfaceBedUpdate};
use hydr::storage::{LevelPoolCurve, LevelPoolKnot, PoolCurveEpoch, PoolCurveSchedule};
use hydr::structures::{
    BlockageFraction, ControlCommand, DirectionPolicy, DiscreteGeometryKnot,
    ScalarGeometryParameter as Param, StructureBlockage, StructureCondition, StructureGeometryKnot,
    StructureGeometrySchedule, StructureGeometryTrack, StructureOperatingPoint,
};
use hydr_core::capability::UnsupportedReason;

fn expect_input_error(
    result: std::result::Result<SurfaceChannelDefinition, SurfaceChannelError>,
    needle: &str,
) -> Result {
    let Err(SurfaceChannelError::Input(msg)) = result else {
        return Err(format!("expected Input error containing {needle:?}").into());
    };
    assert!(msg.contains(needle), "expected {needle:?} in: {msg}");
    Ok(())
}

fn weir_case() -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("a", 0.0, 10.0, 3.0)?, 10.0),
        (pond("b", 0.0, 10.0, 3.0)?, 0.0),
    ];
    case.structures = vec![structure(
        "w",
        weir_law(0.8, 2.0)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("a"), 0.0),
        (pool("b"), 0.0),
    )];
    Ok(case)
}

fn sched(id: &str, structure_id: &str, track: StructureGeometryTrack) -> StructureGeometrySchedule {
    StructureGeometrySchedule {
        schedule_id: id.into(),
        structure_id: structure_id.into(),
        track,
        evidence: "Scheduled geometry change".into(),
    }
}

fn scalar_step(
    id: &str,
    s_id: &str,
    parameter: Param,
    (t, l, r): (f64, f64, f64),
) -> StructureGeometrySchedule {
    sched(
        id,
        s_id,
        StructureGeometryTrack::Scalar {
            parameter,
            knots: vec![StructureGeometryKnot::step(t, l, r)],
        },
    )
}

/// Every duplicate-writer conflict pair rejects with `duplicate-parameter-authority`, and a
/// manual gate opening exceeding a scheduled gate range rejects with `gate-range-conflict`.
#[test]
fn duplicate_parameter_authority_rejects_every_conflict_pair() -> Result {
    // 1. Two structure geometry schedules on the same (structure_id, track).
    let (mut setup, _) = weir_case()?.parts()?;
    setup.geometry_schedules = vec![
        scalar_step("s1", "w", Param::WeirCrestElevation, (0.5, 0.8, 0.6)),
        scalar_step("s2", "w", Param::WeirCrestElevation, (1.0, 0.8, 0.7)),
    ];
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 2. Static Blocked condition + geometry schedule on the same blockage track.
    let mut case = weir_case()?;
    case.structures[0].element.condition = StructureCondition::Blocked(StructureBlockage::weir(
        BlockageFraction::new(0.2)?,
        "Static debris".into(),
    )?);
    let (mut setup, _) = case.parts()?;
    setup.geometry_schedules = vec![scalar_step("b", "w", Param::WeirBlockage, (0.5, 0.2, 0.5))];
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 3. Static Closed condition + Closed geometry schedule.
    let mut case = weir_case()?;
    case.structures[0].element.condition = StructureCondition::Closed {
        evidence: "Static closure".into(),
    };
    let (mut setup, _) = case.parts()?;
    setup.geometry_schedules = vec![sched(
        "sched-close",
        "w",
        StructureGeometryTrack::Closed {
            knots: vec![DiscreteGeometryKnot::step(0.5, true, false)],
        },
    )];
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 4. Two pool curve schedules on the same pool.
    let (mut setup, _) = weir_case()?.parts()?;
    let curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 0.0,
            },
            LevelPoolKnot {
                volume_m3: 20.0,
                stage_m: 2.0,
            },
        ],
        crate::pools_fixture::DATUM.into(),
        "Updated curve".into(),
    )?;
    setup.pool_curve_schedules = [("p1", 0.5), ("p2", 1.0)]
        .into_iter()
        .map(|(id, t)| PoolCurveSchedule {
            schedule_id: id.into(),
            pool_id: "a".into(),
            epochs: vec![PoolCurveEpoch {
                time_s: t,
                curve: curve.clone(),
                evidence: "Pool curve epoch".into(),
            }],
        })
        .collect();
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 5. Two surface bed schedules on the same surface cell.
    let (mut setup, _) = weir_case()?.parts()?;
    setup.surface_bed_schedules = [("b1", 0.5, 0.1), ("b2", 1.0, 0.2)]
        .into_iter()
        .map(|(id, t, b)| SurfaceBedSchedule {
            schedule_id: id.into(),
            epochs: vec![SurfaceBedEpoch {
                time_s: t,
                updates: vec![SurfaceBedUpdate {
                    cell: 0,
                    bed_elevation_m: b,
                }],
                evidence: "Bed schedule".into(),
            }],
        })
        .collect();
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 6. Two channel section schedules on the same reach.
    let (mut setup, _) = weir_case()?.parts()?;
    let geom = ChannelGeometry::Rectangular {
        cells: vec![RectangularChannelCell {
            id: "cell-0".into(),
            length_m: 2.0,
            width_m: 1.5,
            bed_elevation_m: 0.0,
            maximum_depth_m: 4.0,
            manning_n: 0.0,
        }],
    };
    setup.channel_section_schedules = [("c1", 0.5), ("c2", 1.0)]
        .into_iter()
        .map(|(id, t)| ChannelSectionSchedule {
            schedule_id: id.into(),
            reach_id: crate::pools_fixture::REACH.into(),
            epochs: vec![ChannelSectionEpoch {
                time_s: t,
                geometry: geom.clone(),
                evidence: "Section epoch".into(),
            }],
        })
        .collect();
    expect_input_error(
        SurfaceChannelDefinition::new(setup),
        "duplicate-parameter-authority",
    )?;

    // 7. Manual gate opening exceeding a scheduled gate maximum opening.
    let mut case = weir_case()?;
    case.structures = vec![structure(
        "g",
        gate_law(0.2)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::GateOpening { opening_m: 0.5 }),
        (pool("a"), 0.0),
        (pool("b"), 0.0),
    )];
    let (mut setup, _) = case.parts()?;
    setup.geometry_schedules = vec![scalar_step(
        "g-range",
        "g",
        Param::GateMaximumOpening,
        (0.5, 1.0, 0.3),
    )];
    expect_input_error(SurfaceChannelDefinition::new(setup), "gate-range-conflict")?;

    Ok(())
}

fn actuated_gate_case(decisions: &[f64]) -> Result<Case> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 20.0, 3.0)?, 4.0),
        (pond("upper", 0.0, 20.0, 3.0)?, 30.0),
    ];
    case.structures = vec![structure(
        "gate",
        gate_law(0.2)?,
        DirectionPolicy::AtoBOnly,
        SurfaceChannelOperationOwner::Actuated,
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    )];
    let mut op = operating(vec![signal("gate", pool("upper"))]);
    op.gates = Some(controlled_gate(
        "gate",
        [0.8, 1.2],
        ([0.1, 0.5], 0.1),
        (0.0, decisions, 2.0),
    ));
    case.operating = Some(op);
    Ok(case)
}

/// A gate range geometry event re-checks an actuated gate's position and target against the new
/// effective range, rejecting without retry on conflict.
#[test]
fn actuated_gate_range_event_rejects_position_and_target_conflicts() -> Result {
    let forcing = PoolForcing::default();

    // Case A: actuator at 0.1 moving toward target 0.5; at t = 0.5 s GateMaximumOpening shrinks
    // to 0.3 -> target 0.5 > 0.3 rejects without retry.
    let (mut setup, initial) = actuated_gate_case(&[0.25, 1.0])?.parts()?;
    setup.config.maximum_step_s = 0.25;
    setup.geometry_schedules = vec![scalar_step(
        "shrink",
        "gate",
        Param::GateMaximumOpening,
        (0.5, 1.0, 0.3),
    )];
    let batch1 = step(
        &SurfaceChannelBatch::new(client(), setup, initial)?,
        0.25,
        &forcing,
    )?;
    let batch2 = step(&batch1, 0.25, &forcing)?;
    assert_eq!(batch2.time(), 0.5);
    let err = step(&batch2, 0.25, &forcing)
        .err()
        .ok_or("expected gate-range-conflict rejection")?;
    assert!(
        err.to_string().contains("gate-range-conflict") && err.to_string().contains("target"),
        "unexpected error: {err}"
    );

    // Case B: coincident decision at t = 0.5 s commands target 0.5 when GateOpeningRangeBlockage
    // steps to 0.7 (effective max = 0.3) -> first interval at 0.5 s rejects without retry.
    let (mut setup, initial) = actuated_gate_case(&[0.5])?.parts()?;
    setup.config.maximum_step_s = 0.5;
    setup.geometry_schedules = vec![scalar_step(
        "jam",
        "gate",
        Param::GateOpeningRangeBlockage,
        (0.5, 0.0, 0.7),
    )];
    let batch1 = step(
        &SurfaceChannelBatch::new(client(), setup, initial)?,
        0.5,
        &forcing,
    )?;
    assert_eq!(batch1.time(), 0.5);
    let err = step(&batch1, 0.5, &forcing)
        .err()
        .ok_or("expected gate-range-conflict after coincident decision")?;
    assert!(
        err.to_string().contains("gate-range-conflict") && err.to_string().contains("target"),
        "unexpected error: {err}"
    );

    Ok(())
}

/// A run combining all four operation owner kinds (Manual, Prescribed, Actuated, ControlledPump)
/// and a structure geometry schedule restarts bit-for-bit via restore, import_snapshot, and resume.
#[test]
fn all_owner_kinds_with_geometry_schedule_restart_bitwise() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 30.0, 4.0)?, 15.0),
        (pond("upper", 0.0, 20.0, 4.0)?, 30.0),
    ];
    case.structures = vec![
        structure(
            "weir-manual",
            weir_law(1.0, 1.5)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "gate-prescribed",
            gate_law(0.2)?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::Prescribed,
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "gate-actuated",
            gate_law(0.2)?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::Actuated,
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "pump-ctrl",
            pump_law(0.1, Some(2.0))?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::ControlledPump,
            (pool("lower"), 0.0),
            (pool("upper"), 0.0),
        ),
    ];
    let mut op = operating(vec![
        signal("gate-actuated", pool("upper")),
        signal("pump-ctrl", pool("lower")),
    ]);
    op.trajectories = vec![trajectory("gate-prescribed", &[(0.0, 0.05), (2.0, 0.25)])?];
    op.gates = Some(controlled_gate(
        "gate-actuated",
        [0.8, 1.2],
        ([0.05, 0.25], 0.2),
        (0.0, &[0.5, 1.0], 2.0),
    ));
    let pump_cmds = [
        ControlCommand::PumpStoppedIsolated,
        ControlCommand::PumpRunning { speed_ratio: 1.0 },
    ];
    op.pumps = Some(controlled_pump(
        "pump-ctrl",
        [0.3, 0.45],
        pump_cmds,
        (0.0, &[0.5, 1.0], 2.0),
    ));
    case.operating = Some(op);
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 0.5;
    setup.geometry_schedules = vec![sched(
        "weir-breach",
        "weir-manual",
        StructureGeometryTrack::Scalar {
            parameter: Param::WeirCrestElevation,
            knots: vec![
                StructureGeometryKnot::continuous(0.0, 1.0),
                StructureGeometryKnot::continuous(1.5, 0.7),
            ],
        },
    )];

    let fresh = || SurfaceChannelBatch::new(client(), setup.clone(), initial.clone());
    let advance_to = |mut b: SurfaceChannelBatch<crate::runtime::Runtime>, t: f64| -> Result<_> {
        let forcing = PoolForcing::default();
        while b.time() < t {
            b = step(&b, t - b.time(), &forcing)?;
            check(&b, &[])?;
        }
        Ok(b)
    };

    let whole = advance_to(fresh()?, 1.5)?;
    let whole_data = whole.snapshot_data()?;

    let mid = advance_to(fresh()?, 0.5)?;
    let snap = mid.snapshot();
    let data = mid.snapshot_data()?;

    for (label, batch) in [
        ("restore", fresh()?.restore(&snap)?),
        ("import", fresh()?.import_snapshot(&data)?),
        (
            "resume",
            SurfaceChannelBatch::resume(client(), &data, data.surface.setup.forcing.clone())?,
        ),
    ] {
        let continued = advance_to(batch, 1.5)?;
        assert_eq!(
            continued.snapshot_data()?,
            whole_data,
            "{label} with all owner kinds + geometry schedule must match bit-for-bit"
        );
    }
    Ok(())
}

/// A control decision sampling a node joined by a culvert behaves identically to any pool node.
#[test]
fn control_decision_sampling_culvert_node_switches_cleanly() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 30.0, 4.0)?, 6.0),
        (pond("upper", 0.0, 15.0, 4.0)?, 22.5),
    ];
    case.structures = vec![
        structure(
            "culvert",
            culvert_law(0.6, 4.0, [0.2, 0.0], false)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "pump",
            pump_law(0.1, Some(2.0))?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::ControlledPump,
            (pool("lower"), 0.0),
            (pool("upper"), 0.0),
        ),
    ];
    let mut op = operating(vec![signal("pump", pool("upper"))]);
    op.pumps = Some(controlled_pump(
        "pump",
        [0.8, 1.2],
        [
            ControlCommand::PumpStoppedIsolated,
            ControlCommand::PumpRunning { speed_ratio: 1.0 },
        ],
        (0.0, &[0.5, 1.0], 2.0),
    ));
    case.operating = Some(op);
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 0.5;
    let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
    let forcing = PoolForcing::default();

    // At t = 0.5 s, upper pool stage > 1.2 m so the pump latches High and starts pumping on [0.5, 1.0].
    let batch1 = step(&batch0, 0.5, &forcing)?;
    check(&batch1, &[])?;
    assert_eq!(
        transfer(&batch1, "pump")?.signed_volume_m3.to_bits(),
        0.0_f64.to_bits()
    );
    assert!(transfer(&batch1, "culvert")?.signed_volume_m3 > 0.0);

    let batch2 = step(&batch1, 0.5, &forcing)?;
    check(&batch2, &[])?;
    assert!(transfer(&batch2, "pump")?.signed_volume_m3 > 0.0);
    Ok(())
}

/// Capabilities declare `duplicate-parameter-authority` and `native-control-authority`.
#[test]
fn capabilities_declare_authority_entries() -> Result {
    let caps = surface_channel_capabilities();
    for (id, reason) in [
        (
            "duplicate-parameter-authority",
            UnsupportedReason::RejectedByAdmission,
        ),
        ("native-control-authority", UnsupportedReason::NotAttached),
    ] {
        let entry = caps.unsupported.iter().find(|u| u.id == id).ok_or(id)?;
        assert_eq!(entry.reason, reason);
    }
    Ok(())
}
