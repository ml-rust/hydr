//! Pinned §5 river conveyance acceptance tests:
//! 1. Steady M1 backwater profile against exact gradually-varied-flow (GVF) ODE under spatial refinement.
//! 2. Tailwater-driven flow reversal with bit-identical cold restart.
//! 3. Changing-width lake-at-rest equilibrium and signed lateral momentum injection.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReach,
    ChannelReachDefinition, RectangularChannelCell,
};
use hydr::forcing::WaterAmount;
use runtime::Runtime;

type TestResult = Result<(), Box<dyn std::error::Error>>;
const G: f64 = 9.81;

fn build_rect_reach(
    cells: Vec<RectangularChannelCell>,
    lateral_bindings: Vec<ChannelLateralBinding>,
    depths: &[f64],
    discharges: &[f64],
    max_step: f64,
) -> Result<ChannelReach<Runtime>, Box<dyn std::error::Error>> {
    let volume_m3 = cells
        .iter()
        .zip(depths)
        .map(|(c, &d)| c.length_m * c.width_m * d)
        .collect();
    let longitudinal_momentum_m4_per_s = cells
        .iter()
        .zip(discharges)
        .map(|(c, &q)| c.length_m * q)
        .collect();
    let def = ChannelReachDefinition {
        vertical_datum: "NAVD88".into(),
        provenance: "Pinned §5 river conveyance acceptance fixture".into(),
        geometry: ChannelGeometry::Rectangular { cells },
        lateral_bindings,
        config: ChannelConfig {
            max_step,
            ..ChannelConfig::default()
        },
    };
    Ok(ChannelReach::new(
        runtime::client(),
        def,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3,
            longitudinal_momentum_m4_per_s,
        },
    )?)
}

fn advance_reach(
    mut reach: ChannelReach<Runtime>,
    target_s: f64,
    left: ChannelBoundary,
    right: ChannelBoundary,
    lateral_inputs: &[ChannelLateralInput],
) -> Result<ChannelReach<Runtime>, Box<dyn std::error::Error>> {
    while reach.time() < target_s - 1e-12 {
        let forcing = ChannelForcingInterval {
            start_s: reach.time(),
            end_s: target_s,
            left: left.clone(),
            right: right.clone(),
            lateral_inputs: lateral_inputs.to_vec(),
        };
        reach = reach.step(target_s, &forcing)?.next;
    }
    Ok(reach)
}

/// Integrates the exact steady rectangular GVF ODE `dh/dx = (S0 - Sf) / (1 - Fr^2)` backward
/// from `x = length_m` (`h = h_down`) using RK4 and returns exact depth at each cell center.
fn exact_m1_backwater_depths(
    length_m: f64,
    width_m: f64,
    slope: f64,
    manning_n: f64,
    q_m3_per_s: f64,
    h_down: f64,
    n_cells: usize,
) -> Vec<f64> {
    let rhs = |h: f64| -> f64 {
        let area = width_m * h;
        let radius = area / (width_m + 2.0 * h);
        let sf = (manning_n * q_m3_per_s / area).powi(2) / radius.powf(4.0 / 3.0);
        let fr2 = (q_m3_per_s * q_m3_per_s) / (G * width_m * width_m * h.powi(3));
        (slope - sf) / (1.0 - fr2)
    };
    let steps_per_cell = 40usize;
    let dx_cell = length_m / n_cells as f64;
    let ds = dx_cell / steps_per_cell as f64;
    let mut h = h_down;
    let mut cell_depths = vec![0.0; n_cells];
    // Integrate backward in x (from x = length_m to x = 0): dh/d(-x) = -rhs(h).
    for rev_idx in 0..n_cells {
        let cell_idx = n_cells - 1 - rev_idx;
        for step in 0..steps_per_cell {
            if step == steps_per_cell / 2 {
                cell_depths[cell_idx] = h;
            }
            let k1 = -rhs(h);
            let k2 = -rhs(h + 0.5 * ds * k1);
            let k3 = -rhs(h + 0.5 * ds * k2);
            let k4 = -rhs(h + ds * k3);
            h += (ds / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
    }
    cell_depths
}

fn run_m1_backwater(n_cells: usize) -> Result<(Vec<f64>, f64), Box<dyn std::error::Error>> {
    let length_m = 1_000.0;
    let width_m = 10.0;
    let slope = 0.001;
    let manning_n = 0.025;
    let q = 20.0;
    let h_down = 2.50;
    let dx = length_m / n_cells as f64;

    let cells: Vec<RectangularChannelCell> = (0..n_cells)
        .map(|i| {
            let x_center = (i as f64 + 0.5) * dx;
            let bed = 2.0 - slope * x_center;
            RectangularChannelCell {
                id: format!("c{i}"),
                length_m: dx,
                width_m,
                bed_elevation_m: bed,
                maximum_depth_m: 10.0,
                manning_n,
            }
        })
        .collect();
    let z_right_cell = cells[n_cells - 1].bed_elevation_m;
    let stage_right = z_right_cell + h_down;

    let exact = exact_m1_backwater_depths(length_m, width_m, slope, manning_n, q, h_down, n_cells);
    let init_depths = vec![2.0; n_cells];
    let init_discharges = vec![q; n_cells];
    let reach = build_rect_reach(cells, Vec::new(), &init_depths, &init_discharges, 5.0)?;

    let reach = advance_reach(
        reach,
        1_500.0,
        ChannelBoundary::Discharge {
            discharge_m3_per_s: q,
        },
        ChannelBoundary::Stage {
            elevation_m: stage_right,
        },
        &[],
    )?;

    let state = reach.state()?;
    let depths: Vec<f64> = state.iter().map(|c| c.depth_m).collect();
    let l2_err = (depths
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n_cells as f64)
        .sqrt();
    Ok((depths, l2_err))
}

#[test]
fn m1_backwater_profile_converges_to_exact_gvf_ode_under_refinement() -> TestResult {
    let (depths_20, err_20) = run_m1_backwater(20)?;
    let (depths_40, err_40) = run_m1_backwater(40)?;

    // M1 backwater curve must rise monotonically toward the raised downstream stage.
    for w in depths_40.windows(2) {
        assert!(
            w[1] > w[0],
            "M1 backwater depth must increase downstream: {} -> {}",
            w[0],
            w[1]
        );
    }
    assert!(
        depths_40[0] > 1.4 && depths_40[39] > 2.45,
        "M1 backwater range unexpected: upstream={}, downstream={}",
        depths_40[0],
        depths_40[39]
    );
    assert!(
        err_40 < 0.025,
        "M1 backwater L2 error at N=40 too large: {err_40:.6} m"
    );
    assert!(
        err_40 < err_20,
        "M1 backwater L2 error must decrease under spatial refinement: N=20 ({err_20:.6} m) vs N=40 ({err_40:.6} m)"
    );
    assert!(depths_20.len() == 20);
    Ok(())
}

#[test]
fn tailwater_surge_drives_clean_flow_reversal_and_exact_restart() -> TestResult {
    let n = 16;
    let dx = 25.0;
    let cells = || {
        (0..n)
            .map(|i| RectangularChannelCell {
                id: format!("r{i}"),
                length_m: dx,
                width_m: 10.0,
                bed_elevation_m: 0.0,
                maximum_depth_m: 10.0,
                manning_n: 0.02,
            })
            .collect::<Vec<_>>()
    };

    let ext = |depth_m: f64| ChannelBoundary::Exterior {
        depth_m,
        velocity_m_per_s: 0.0,
    };

    // Uninterrupted run: forward flow (0..60 s), then raised tailwater forcing reversal (60..240 s).
    let mut reference = build_rect_reach(cells(), Vec::new(), &vec![2.0; n], &vec![10.0; n], 2.0)?;
    reference = advance_reach(reference, 60.0, ext(2.15), ext(2.00), &[])?;
    assert!(
        reference
            .state()?
            .iter()
            .all(|c| c.discharge_m3_per_s > 0.0),
        "initial phase must have positive forward flow"
    );
    // Reversal phase split at 120 s so cold restart uses identical step partition.
    for target in [120.0, 240.0] {
        reference = advance_reach(reference, target, ext(2.00), ext(2.35), &[])?;
    }
    let ref_state = reference.state()?;
    assert!(
        ref_state.iter().all(|c| c.discharge_m3_per_s < -1.0),
        "raised tailwater must reverse discharge across the entire reach"
    );

    // Cold restart from t = 120 s snapshot into a fresh ChannelReach.
    let origin = build_rect_reach(cells(), Vec::new(), &vec![2.0; n], &vec![10.0; n], 2.0)?;
    let mut split = advance_reach(origin.clone(), 60.0, ext(2.15), ext(2.00), &[])?;
    split = advance_reach(split, 120.0, ext(2.00), ext(2.35), &[])?;
    let snap = split.snapshot_data()?;

    let mut restarted = origin.import_snapshot(&snap)?;
    restarted = advance_reach(restarted, 240.0, ext(2.00), ext(2.35), &[])?;
    let restarted_state = restarted.state()?;
    assert_eq!(
        restarted_state, ref_state,
        "cold restart through flow reversal must match uninterrupted run bit-for-bit"
    );
    Ok(())
}

#[test]
fn changing_width_equilibrium_and_lateral_momentum_audit() -> TestResult {
    // 4-cell changing-width reach with bed steps at constant water surface elevation 3.0 m.
    let widths = [8.0, 16.0, 6.0, 12.0];
    let beds = [0.2, 0.5, 0.1, 0.4];
    let depths: Vec<f64> = beds.iter().map(|&z| 3.0 - z).collect();
    let make_cells = || {
        (0..4)
            .map(|i| RectangularChannelCell {
                id: format!("w{i}"),
                length_m: 20.0,
                width_m: widths[i],
                bed_elevation_m: beds[i],
                maximum_depth_m: 10.0,
                manning_n: 0.0,
            })
            .collect::<Vec<_>>()
    };

    let lake = build_rect_reach(make_cells(), Vec::new(), &depths, &[0.0; 4], 1.0)?;
    let lake = advance_reach(
        lake,
        20.0,
        ChannelBoundary::Wall,
        ChannelBoundary::Wall,
        &[],
    )?;
    for cell in lake.state()? {
        assert!(
            cell.discharge_m3_per_s.abs() < 1e-12,
            "changing-width lake-at-rest equilibrium violated: Q={}",
            cell.discharge_m3_per_s
        );
    }

    // Signed lateral momentum injection: +1.5 m/s vs -1.5 m/s into cell w1.
    let binding = ChannelLateralBinding {
        boundary_id: "trib".into(),
        cell_id: "w1".into(),
        direction: ChannelLateralDirection::Inflow,
        ownership_assertion: "Independent lateral tributary inflow".into(),
    };
    let pos = build_rect_reach(make_cells(), vec![binding.clone()], &depths, &[0.0; 4], 0.5)?;
    let neg = build_rect_reach(make_cells(), vec![binding], &depths, &[0.0; 4], 0.5)?;
    let initial_vol: f64 = pos.state()?.iter().map(|c| c.volume_m3).sum();

    let dt = 2.0;
    let q_lat = 4.0;
    let pos = advance_reach(
        pos,
        dt,
        ChannelBoundary::Wall,
        ChannelBoundary::Wall,
        &[ChannelLateralInput::Inflow {
            boundary_id: "trib".into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: q_lat,
            },
            longitudinal_velocity_m_per_s: 1.5,
        }],
    )?;
    let neg = advance_reach(
        neg,
        dt,
        ChannelBoundary::Wall,
        ChannelBoundary::Wall,
        &[ChannelLateralInput::Inflow {
            boundary_id: "trib".into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: q_lat,
            },
            longitudinal_velocity_m_per_s: -1.5,
        }],
    )?;

    let pos_vol: f64 = pos.state()?.iter().map(|c| c.volume_m3).sum();
    let neg_vol: f64 = neg.state()?.iter().map(|c| c.volume_m3).sum();
    assert!((pos_vol - (initial_vol + q_lat * dt)).abs() < 1e-9);
    assert!((neg_vol - (initial_vol + q_lat * dt)).abs() < 1e-9);

    let pos_mom: f64 = pos
        .state()?
        .iter()
        .map(|c| c.longitudinal_momentum_m4_per_s)
        .sum();
    let neg_mom: f64 = neg
        .state()?
        .iter()
        .map(|c| c.longitudinal_momentum_m4_per_s)
        .sum();
    assert!(
        pos_mom > 0.0 && neg_mom < 0.0,
        "signed lateral momentum must steer longitudinal momentum: pos={pos_mom}, neg={neg_mom}"
    );
    Ok(())
}
