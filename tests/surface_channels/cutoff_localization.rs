//! Cutoff crossing localization: two-outlet pool drain closure times against an independent ODE
//! reference, retry reduction vs blind halving, and bitwise restart through a located crossing.

use crate::pools_evidence::{check, pool_stage};
use crate::pools_fixture::{
    Case, PoolForcing, Result, manual, pond, pool, step, structure, transfer, weir_law,
};
use crate::runtime::client;
use hydr::coupling::SurfaceChannelBatch;
use hydr::structures::{DirectionPolicy, StructureOperatingPoint};

const AREA: f64 = 10.0;
const H0: f64 = 1.4;
const Z1: f64 = 1.0;
const Z2: f64 = 0.4;
const L1: f64 = 2.0;
const L2: f64 = 1.5;
const CW: f64 = 1.7;

/// Independent RK4 reference for a constant-area pool draining through two free-overflow weirs
/// at crests `Z1 > Z2`. Returns `(t_close_1, t_close_2)` where `h(t)` crosses `Z1` and `Z2 + 0.05`.
fn rk4_two_outlet_crossing_time(target_stage: f64) -> f64 {
    let dt = 1e-4;
    let rhs = |h: f64| -> f64 {
        let q1 = if h > Z1 {
            CW * L1 * (h - Z1).powf(1.5)
        } else {
            0.0
        };
        let q2 = if h > Z2 {
            CW * L2 * (h - Z2).powf(1.5)
        } else {
            0.0
        };
        -(q1 + q2) / AREA
    };
    let mut t = 0.0;
    let mut h = H0;
    for _ in 0..500_000 {
        let k1 = rhs(h);
        let k2 = rhs(h + 0.5 * dt * k1);
        let k3 = rhs(h + 0.5 * dt * k2);
        let k4 = rhs(h + dt * k3);
        let h_next = h + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        if h_next <= target_stage {
            let frac = (h - target_stage) / (h - h_next);
            return t + frac * dt;
        }
        h = h_next;
        t += dt;
    }
    t
}

/// Two-outlet pool case: `upper` drains into `lower` through two raised-bed weirs at `Z1` and
/// `Z2`.
fn two_outlet_case(
    max_step_s: f64,
) -> Result<(
    hydr::coupling::SurfaceChannelSetup,
    hydr::coupling::SurfaceChannelInitialState,
)> {
    let mut case = Case::tanks(0.0, 0.0);
    case.ponds = vec![
        (pond("lower", 0.0, 200.0, 3.0)?, 0.0),
        (pond("upper", 0.0, AREA, 3.0)?, H0 * AREA),
    ];
    case.structures = vec![
        structure(
            "upper-weir",
            weir_law(Z1, L1)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), Z1),
            (pool("lower"), 0.0),
        ),
        structure(
            "lower-weir",
            weir_law(Z2, L2)?,
            DirectionPolicy::AtoBOnly,
            manual(StructureOperatingPoint::Passive),
            (pool("upper"), Z2),
            (pool("lower"), 0.0),
        ),
    ];
    let (mut setup, initial) = case.parts()?;
    setup.config.maximum_step_s = max_step_s;
    Ok((setup, initial))
}

/// A pool draining through two outlets at different beds localizes the upper outlet's closure
/// time to match the independent ODE reference within 0.01 s, takes far fewer retries than blind
/// halving, and continues draining through the lower outlet cleanly.
#[test]
fn two_outlet_pool_drain_localizes_upper_closure_and_matches_ode() -> Result {
    let t_ode_close = rk4_two_outlet_crossing_time(Z1);
    let (setup, initial) = two_outlet_case(0.3)?;
    let mut batch = SurfaceChannelBatch::new(client(), setup, initial)?;
    let forcing = PoolForcing::default();

    let mut closure_time: Option<f64> = None;
    let mut total_retries = 0usize;
    let mut closed_seen = false;
    while batch.time() < 5.0 {
        let dt = (5.0 - batch.time()).min(0.3);
        batch = step(&batch, dt, &forcing)?;
        check(&batch, &[])?;
        let diag = batch.last_step()?.ok_or("missing diagnostics")?;
        total_retries += diag.retries;
        let upper_vol = transfer(&batch, "upper-weir")?.signed_volume_m3;
        let stage = pool_stage(&batch, "upper")?;
        if closure_time.is_none() && (stage - Z1).abs() < 1e-3 {
            closure_time = Some(batch.time());
        }
        if closure_time.is_some() && upper_vol == 0.0 {
            closed_seen = true;
        }
    }

    let t_loc = closure_time.ok_or("upper outlet closure was not localized")?;
    let gap = (t_loc - t_ode_close).abs();
    assert!(
        gap < 0.01,
        "localized closure t={t_loc:.6} differs from ODE t={t_ode_close:.6} by {gap:.6}"
    );
    assert!(closed_seen, "upper-weir must close after crossing Z1");
    assert!(
        pool_stage(&batch, "upper")? < Z1 - 0.1,
        "lower-weir must continue draining upper pool below Z1"
    );
    // Blind halving toward a raised cutoff takes ~12+ rejected attempts across repeated slivers;
    // secant-guided certified bisection locates it in a small bounded count.
    assert!(
        total_retries <= 10,
        "expected <= 10 total retries with cutoff localization, got {total_retries}"
    );
    Ok(())
}

/// Checkpoint before and on a located crossing interval restarts bit-for-bit via restore,
/// import_snapshot, and resume.
#[test]
fn restart_through_located_cutoff_crossing_is_bitwise() -> Result {
    let (setup, initial) = two_outlet_case(1.0)?;
    let fresh = || SurfaceChannelBatch::new(client(), setup.clone(), initial.clone());
    let forcing = PoolForcing::default();

    let advance_to =
        |mut b: SurfaceChannelBatch<crate::runtime::Runtime>, target: f64| -> Result<_> {
            while b.time() < target {
                b = step(&b, (target - b.time()).min(1.0), &forcing)?;
                check(&b, &[])?;
            }
            Ok(b)
        };

    let whole = advance_to(fresh()?, 4.0)?;
    let whole_data = whole.snapshot_data()?;

    // Step until the first located crossing interval completes (retries >= 1, end < requested).
    let mut at_crossing = fresh()?;
    let mut found_crossing = false;
    while at_crossing.time() < 4.0 {
        at_crossing = step(&at_crossing, (4.0 - at_crossing.time()).min(1.0), &forcing)?;
        let diag = at_crossing.last_step()?.ok_or("missing diagnostics")?;
        if diag.retries >= 1 && diag.end_s < diag.requested_end_s {
            found_crossing = true;
            break;
        }
    }
    assert!(found_crossing, "expected a located crossing interval");

    let snap = at_crossing.snapshot();
    let data = at_crossing.snapshot_data()?;
    for (label, batch) in [
        ("restore", fresh()?.restore(&snap)?),
        ("import", fresh()?.import_snapshot(&data)?),
        (
            "resume",
            SurfaceChannelBatch::resume(client(), &data, data.surface.setup.forcing.clone())?,
        ),
    ] {
        let continued = advance_to(batch, 4.0)?;
        assert_eq!(
            continued.snapshot_data()?,
            whole_data,
            "{label} through located crossing must match uninterrupted run bit-for-bit"
        );
    }
    Ok(())
}
