//! Surface/channel coordinator structure geometry schedules: linear weir breach ODE convergence,
//! step culvert blockage alignment, dry and equal-head invariance, coincident control decisions,
//! and bitwise restart with forged epoch and receipt rejection.

use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, controlled_gate, culvert_law, gate_law, manual, operating,
    orifice_law, pond, pool, signal, step, structure, transfer, weir_law,
};
use crate::runtime::client;
use hydr::coupling::{
    SurfaceChannelBatch, SurfaceChannelError, SurfaceChannelOperationOwner,
    SurfaceChannelSnapshotData,
};
use hydr::structures::{
    DirectionPolicy, DiscreteGeometryKnot, ScalarGeometryParameter, StructureGeometryKnot,
    StructureGeometrySchedule, StructureGeometryTrack, StructureOperatingPoint, StructureRegime,
};

const ALLOWANCE: f64 = 1e-12;

/// Advances `batch` in intervals up to `target_s`, enforcing step-by-step mass and geometry
/// conservation receipts.
fn advance_to(
    mut batch: SurfaceChannelBatch<crate::runtime::Runtime>,
    target_s: f64,
) -> Result<SurfaceChannelBatch<crate::runtime::Runtime>> {
    let forcing = PoolForcing::default();
    while batch.time() < target_s {
        let remaining = target_s - batch.time();
        batch = step(&batch, remaining, &forcing)?;
        check(&batch, &[])?;
        let diag = batch.last_step()?.ok_or("missing step diagnostics")?;
        for receipt in &diag.geometry {
            assert_eq!(
                receipt.storage_before_m3.to_bits(),
                receipt.storage_after_m3.to_bits(),
                "geometry event must conserve total storage bit-for-bit: {receipt:?}"
            );
        }
    }
    Ok(batch)
}

/// Independent RK4 reference for free weir drainage from a constant-area pond `area_m2` over a
/// linearly lowering crest `z(t) = z0 + slope * t`.
fn rk4_weir_breach_stage(
    h0: f64,
    z0: f64,
    slope: f64,
    cw_l_over_a: f64,
    t_end: f64,
    steps: usize,
) -> f64 {
    let dt = t_end / steps as f64;
    let rhs = |t: f64, h: f64| -> f64 {
        let head = (h - (z0 + slope * t)).max(0.0);
        -cw_l_over_a * head * head.sqrt()
    };
    let mut t = 0.0;
    let mut h = h0;
    for _ in 0..steps {
        let k1 = rhs(t, h);
        let k2 = rhs(t + 0.5 * dt, h + 0.5 * dt * k1);
        let k3 = rhs(t + 0.5 * dt, h + 0.5 * dt * k2);
        let k4 = rhs(t + dt, h + dt * k3);
        h += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += dt;
    }
    h
}

/// Linear weir breach lowers the crest from 1.2 m to 0.8 m over [0, 4] s.
///
/// The coordinator matches an independent RK4 ODE reference and achieves second-order
/// self-refinement (>= 1.8) across dt, dt/2, and dt/4.
#[test]
fn linear_weir_breach_matches_ode_and_refines_second_order() -> Result {
    let area = 10.0;
    let h0 = 1.5;
    let z0 = 1.2;
    let z1 = 0.8;
    let t_end = 4.0;
    let cw = 1.7;
    let length = 2.0;
    let h_ref = rk4_weir_breach_stage(h0, z0, (z1 - z0) / t_end, cw * length / area, t_end, 4000);

    let run_with_dt = |dt: f64| -> Result<f64> {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("lower", 0.0, 100.0, 3.0)?, 0.0),
            (pond("upper", 0.0, area, 3.0)?, h0 * area),
        ];
        case.structures = vec![structure(
            "breach",
            weir_law(z0, length)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        )];
        let (mut setup, initial) = case.parts()?;
        setup.config.maximum_step_s = dt;
        setup.geometry_schedules = vec![StructureGeometrySchedule {
            schedule_id: "breach-crest".into(),
            structure_id: "breach".into(),
            track: StructureGeometryTrack::Scalar {
                parameter: ScalarGeometryParameter::WeirCrestElevation,
                knots: vec![
                    StructureGeometryKnot::continuous(0.0, z0),
                    StructureGeometryKnot::continuous(t_end, z1),
                ],
            },
            evidence: "Linear embankment breach progression".into(),
        }];
        let batch = SurfaceChannelBatch::new(client(), setup, initial)?;
        let final_batch = advance_to(batch, t_end)?;
        let diag = final_batch.last_step()?.ok_or("missing diagnostics")?;
        assert_eq!(diag.geometry.len(), 1);
        assert_eq!(diag.geometry[0].element_id, "breach");
        pool_stage(&final_batch, "upper")
    };

    let h_coarse = run_with_dt(0.5)?;
    let h_mid = run_with_dt(0.25)?;
    let h_fine = run_with_dt(0.125)?;

    let e_coarse = (h_coarse - h_ref).abs();
    let e_mid = (h_mid - h_ref).abs();
    let e_fine = (h_fine - h_ref).abs();
    assert!(
        e_fine < 1e-4,
        "fine stage={h_fine} differs from RK4 reference={h_ref} by {e_fine:e}"
    );
    let order = (e_mid / e_fine).log2();
    assert!(
        order >= 1.8,
        "expected order >= 1.8, got {order:.3} (errors: {e_coarse:e}, {e_mid:e}, {e_fine:e})"
    );
    Ok(())
}

/// Step culvert face blockage at t_e = 0.75 s splits a 1.0 s request at t_e and lowers discharge
/// in the first interval after t_e.
#[test]
fn step_culvert_blockage_splits_at_event_and_reduces_discharge() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 40.0, 3.0)?, 0.0),
        (pond("upper", 0.0, 40.0, 3.0)?, 1.6 * 40.0),
    ];
    case.structures = vec![structure(
        "culvert",
        culvert_law(0.6, 4.0, [0.2, 0.0], false)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    )];
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 1.0;
    setup.geometry_schedules = vec![StructureGeometrySchedule {
        schedule_id: "culvert-face-a".into(),
        structure_id: "culvert".into(),
        track: StructureGeometryTrack::Scalar {
            parameter: ScalarGeometryParameter::CulvertFaceABlockage,
            knots: vec![StructureGeometryKnot::step(0.75, 0.0, 0.5)],
        },
        evidence: "Debris mat arrives at 0.75 s".into(),
    }];
    let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
    let forcing = PoolForcing::default();

    // Requesting 1.0 s must split at the 0.75 s geometry event.
    let batch1 = step(&batch0, 1.0, &forcing)?;
    check(&batch1, &[])?;
    let d1 = batch1.last_step()?.ok_or("missing first step")?;
    assert_eq!(d1.end_s, 0.75, "interval must end at event time 0.75 s");
    let q1 = transfer(&batch1, "culvert")?.signed_volume_m3 / (d1.end_s - d1.start_s);

    // Next interval [0.75, 1.0] sees 50% face blockage.
    let batch2 = step(&batch1, 0.25, &forcing)?;
    check(&batch2, &[])?;
    let d2 = batch2.last_step()?.ok_or("missing second step")?;
    assert_eq!(d2.end_s, 1.0);
    let q2 = transfer(&batch2, "culvert")?.signed_volume_m3 / (d2.end_s - d2.start_s);
    assert!(
        q2 < 0.75 * q1,
        "discharge must drop after 50% face blockage: q1={q1}, q2={q2}"
    );
    assert_eq!(
        d2.geometry[0].storage_before_m3.to_bits(),
        d2.geometry[0].storage_after_m3.to_bits()
    );
    Ok(())
}

/// Geometry events under dry donors and under equal heads move exact zero water.
#[test]
fn dry_and_equal_heads_move_zero_across_geometry_events() -> Result {
    // Dry donors with weir crest step and culvert sediment step.
    {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("a", 0.0, 10.0, 3.0)?, 0.0),
            (pond("b", 0.0, 10.0, 3.0)?, 0.0),
        ];
        case.structures = vec![
            structure(
                "weir",
                weir_law(0.8, 2.0)?,
                DirectionPolicy::Bidirectional,
                manual(StructureOperatingPoint::Passive),
                (pool("a"), 0.0),
                (pool("b"), 0.0),
            ),
            structure(
                "culvert",
                culvert_law(0.6, 4.0, [0.2, 0.2], true)?,
                DirectionPolicy::Bidirectional,
                manual(StructureOperatingPoint::Passive),
                (pool("a"), 0.0),
                (pool("b"), 0.0),
            ),
        ];
        let (mut setup, initial) = case.parts()?;
        setup.geometry_schedules = vec![
            StructureGeometrySchedule {
                schedule_id: "dry-weir-crest".into(),
                structure_id: "weir".into(),
                track: StructureGeometryTrack::Scalar {
                    parameter: ScalarGeometryParameter::WeirCrestElevation,
                    knots: vec![StructureGeometryKnot::step(0.5, 0.8, 0.5)],
                },
                evidence: "Dry crest drop".into(),
            },
            StructureGeometrySchedule {
                schedule_id: "dry-culvert-sediment".into(),
                structure_id: "culvert".into(),
                track: StructureGeometryTrack::CulvertSediment {
                    knots: vec![StructureGeometryKnot::step(0.5, 0.0, 0.2)],
                    bed_manning_n: 0.030,
                },
                evidence: "Dry sediment deposition".into(),
            },
        ];
        let batch = advance_to(SurfaceChannelBatch::new(client(), setup, initial)?, 1.0)?;
        for id in ["weir", "culvert"] {
            assert_eq!(
                transfer(&batch, id)?.signed_volume_m3.to_bits(),
                0.0_f64.to_bits()
            );
        }
    }

    // Equal heads with orifice area step and weir crest step.
    {
        let mut case = Case::tanks(0.0, 0.0);
        case.ponds = vec![
            (pond("a", 0.0, 10.0, 3.0)?, 12.0),
            (pond("b", 0.0, 10.0, 3.0)?, 12.0),
        ];
        case.structures = vec![
            structure(
                "orifice",
                orifice_law(0.1, 0.5, true)?,
                DirectionPolicy::Bidirectional,
                manual(StructureOperatingPoint::Passive),
                (pool("a"), 0.0),
                (pool("b"), 0.0),
            ),
            structure(
                "weir",
                weir_law(0.8, 2.0)?,
                DirectionPolicy::Bidirectional,
                manual(StructureOperatingPoint::Passive),
                (pool("a"), 0.0),
                (pool("b"), 0.0),
            ),
        ];
        let (mut setup, initial) = case.parts()?;
        setup.geometry_schedules = vec![
            StructureGeometrySchedule {
                schedule_id: "eq-orifice-area".into(),
                structure_id: "orifice".into(),
                track: StructureGeometryTrack::Scalar {
                    parameter: ScalarGeometryParameter::OrificeArea,
                    knots: vec![StructureGeometryKnot::step(0.5, 0.4, 0.8)],
                },
                evidence: "Equal-head orifice enlargement".into(),
            },
            StructureGeometrySchedule {
                schedule_id: "eq-weir-crest".into(),
                structure_id: "weir".into(),
                track: StructureGeometryTrack::Scalar {
                    parameter: ScalarGeometryParameter::WeirCrestElevation,
                    knots: vec![StructureGeometryKnot::step(0.5, 0.8, 0.6)],
                },
                evidence: "Equal-head weir crest lowering".into(),
            },
        ];
        let batch = advance_to(SurfaceChannelBatch::new(client(), setup, initial)?, 1.0)?;
        for id in ["orifice", "weir"] {
            assert_eq!(
                transfer(&batch, id)?.signed_volume_m3.to_bits(),
                0.0_f64.to_bits()
            );
        }
        assert!((pool_stage(&batch, "a")? - 1.2).abs() <= ALLOWANCE);
        assert!((pool_stage(&batch, "b")? - 1.2).abs() <= ALLOWANCE);
    }
    Ok(())
}

/// A geometry step coincident with a control decision at t = 1.0 s aligns cleanly: the first
/// interval ends at 1.0 s, and the second interval sees both the new geometry and the control
/// decision.
#[test]
fn coincident_geometry_and_control_decision_align_cleanly() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 20.0, 3.0)?, 4.0),
        (pond("upper", 0.0, 20.0, 3.0)?, 28.0),
    ];
    case.structures = vec![
        structure(
            "gate",
            gate_law(0.4)?,
            DirectionPolicy::AtoBOnly,
            SurfaceChannelOperationOwner::Actuated,
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
        structure(
            "weir",
            weir_law(0.8, 1.0)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), 0.0),
            (pool("lower"), 0.0),
        ),
    ];
    let mut op = operating(vec![signal("gate", pool("upper"))]);
    op.gates = Some(controlled_gate(
        "gate",
        [0.8, 1.2],
        ([0.1, 0.6], 0.5),
        (0.0, &[0.5], 2.0),
    ));
    case.operating = Some(op);
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 1.0;
    setup.geometry_schedules = vec![StructureGeometrySchedule {
        schedule_id: "coincident-weir-close".into(),
        structure_id: "weir".into(),
        track: StructureGeometryTrack::Closed {
            knots: vec![DiscreteGeometryKnot::step(0.5, false, true)],
        },
        evidence: "Stop logs dropped at decision time 0.5 s".into(),
    }];
    let batch0 = SurfaceChannelBatch::new(client(), setup, initial)?;
    let forcing = PoolForcing::default();

    let batch1 = step(&batch0, 1.0, &forcing)?;
    check(&batch1, &[])?;
    assert_eq!(batch1.time(), 0.5);
    assert!(transfer(&batch1, "weir")?.signed_volume_m3 > 0.0);

    let batch2 = step(&batch1, 0.5, &forcing)?;
    check(&batch2, &[])?;
    assert_eq!(batch2.time(), 1.0);
    let weir_after = transfer(&batch2, "weir")?;
    assert_eq!(weir_after.regime, StructureRegime::Closed);
    assert_eq!(weir_after.signed_volume_m3.to_bits(), 0.0_f64.to_bits());
    Ok(())
}

/// Restart on an event time and between events matches the uninterrupted run bit-for-bit;
/// forged `geometry_epochs` and forged `last_interval.diagnostics.geometry` reject at import.
#[test]
fn geometry_schedule_restart_and_forged_epoch_rejection() -> Result {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 20.0, 3.0)?, 0.0),
        (pond("upper", 0.0, 10.0, 3.0)?, 15.0),
    ];
    case.structures = vec![structure(
        "weir",
        weir_law(1.2, 2.0)?,
        DirectionPolicy::AtoBOnly,
        manual(StructureOperatingPoint::Passive),
        (pool("upper"), 0.0),
        (pool("lower"), 0.0),
    )];
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = 0.5;
    setup.geometry_schedules = vec![
        StructureGeometrySchedule {
            schedule_id: "restart-weir-crest".into(),
            structure_id: "weir".into(),
            track: StructureGeometryTrack::Scalar {
                parameter: ScalarGeometryParameter::WeirCrestElevation,
                knots: vec![
                    StructureGeometryKnot::continuous(0.0, 1.2),
                    StructureGeometryKnot::continuous(2.0, 0.8),
                ],
            },
            evidence: "Linear crest lowering".into(),
        },
        StructureGeometrySchedule {
            schedule_id: "restart-weir-close".into(),
            structure_id: "weir".into(),
            track: StructureGeometryTrack::Closed {
                knots: vec![DiscreteGeometryKnot::step(1.0, false, true)],
            },
            evidence: "Emergency closure at 1.0 s".into(),
        },
    ];

    let fresh = || SurfaceChannelBatch::new(client(), setup.clone(), initial.clone());
    let whole = advance_to(fresh()?, 1.5)?;
    let whole_data = whole.snapshot_data()?;

    for checkpoint_t in [0.5, 1.0] {
        let mid = advance_to(fresh()?, checkpoint_t)?;
        let snap = mid.snapshot();
        let data = mid.snapshot_data()?;

        let restored = fresh()?.restore(&snap)?;
        let imported = fresh()?.import_snapshot(&data)?;
        let resumed =
            SurfaceChannelBatch::resume(client(), &data, data.surface.setup.forcing.clone())?;

        for (label, batch) in [
            ("restore", restored),
            ("import", imported),
            ("resume", resumed),
        ] {
            let continued = advance_to(batch, 1.5)?;
            assert_eq!(
                continued.snapshot_data()?,
                whole_data,
                "{label} from t={checkpoint_t} must match uninterrupted run bit-for-bit"
            );
        }

        // Forged geometry_epochs must reject on both import_snapshot and resume.
        let mut forged_epochs = data.clone();
        forged_epochs.geometry_epochs[1] += 1;
        for result in [
            fresh()?.import_snapshot(&forged_epochs),
            SurfaceChannelBatch::resume(
                client(),
                &forged_epochs,
                forged_epochs.surface.setup.forcing.clone(),
            ),
        ] {
            match result {
                Err(SurfaceChannelError::Input(msg)) => {
                    assert!(
                        msg.contains("geometry_epochs"),
                        "expected geometry_epochs in {msg:?}"
                    );
                }
                other => return Err(format!("expected Input error, got {other:?}").into()),
            }
        }

        // Forged geometry receipt must reject on both import_snapshot and resume.
        let mut forged_receipt: SurfaceChannelSnapshotData = data.clone();
        let last = forged_receipt
            .last_interval
            .as_mut()
            .ok_or("missing last_interval")?;
        last.diagnostics.geometry[0].storage_after_m3 += 1e-6;
        for result in [
            fresh()?.import_snapshot(&forged_receipt),
            SurfaceChannelBatch::resume(
                client(),
                &forged_receipt,
                forged_receipt.surface.setup.forcing.clone(),
            ),
        ] {
            match result {
                Err(SurfaceChannelError::Input(msg)) => {
                    assert!(
                        msg.contains("last_interval.diagnostics.geometry"),
                        "expected last_interval.diagnostics.geometry in {msg:?}"
                    );
                }
                other => return Err(format!("expected Input error, got {other:?}").into()),
            }
        }
    }
    Ok(())
}
