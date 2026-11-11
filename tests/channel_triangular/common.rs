use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelReach, ChannelReachDefinition, TriangularChannelCell,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<crate::runtime::Runtime>;
pub const GRAVITY: f64 = 9.81;

pub fn cell(index: usize, length_m: f64) -> TriangularChannelCell {
    TriangularChannelCell {
        id: format!("cell-{index}"),
        length_m,
        maximum_depth_m: 10.0,
        manning_n: 0.0,
    }
}

pub fn definition(cells: Vec<TriangularChannelCell>, max_step: f64) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: "Synthetic SI datum".into(),
        provenance: "Explicit triangular hydraulic acceptance geometry".into(),
        geometry: ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
            cells,
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            max_step,
            ..ChannelConfig::default()
        },
    }
}

pub fn reach(
    definition: ChannelReachDefinition,
    areas: &[f64],
    discharges: &[f64],
) -> Result<Reach, Box<dyn std::error::Error>> {
    let ChannelGeometry::PrismaticTriangular { cells, .. } = &definition.geometry else {
        return Err("Expected triangular acceptance geometry".into());
    };
    assert_eq!(cells.len(), areas.len());
    assert_eq!(cells.len(), discharges.len());
    let volume_m3 = cells
        .iter()
        .zip(areas)
        .map(|(cell, area)| cell.length_m * area)
        .collect();
    let longitudinal_momentum_m4_per_s = cells
        .iter()
        .zip(discharges)
        .map(|(cell, q)| cell.length_m * q)
        .collect();
    Ok(ChannelReach::new(
        crate::runtime::client(),
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3,
            longitudinal_momentum_m4_per_s,
        },
    )?)
}

pub fn exterior(depth_m: f64, velocity_m_per_s: f64) -> ChannelBoundary {
    ChannelBoundary::Exterior {
        depth_m,
        velocity_m_per_s,
    }
}

pub fn forcing(
    start_s: f64,
    end_s: f64,
    boundaries: [ChannelBoundary; 2],
) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s,
        end_s,
        left: boundaries[0].clone(),
        right: boundaries[1].clone(),
        lateral_inputs: Vec::new(),
    }
}

pub fn walls() -> [ChannelBoundary; 2] {
    [ChannelBoundary::Wall, ChannelBoundary::Wall]
}

pub fn advance(
    mut reach: Reach,
    target: f64,
    boundaries: [ChannelBoundary; 2],
) -> Result<Reach, Box<dyn std::error::Error>> {
    let mut steps = 0;
    while reach.time() < target {
        let start = reach.time();
        reach = reach
            .step(target, &forcing(start, target, boundaries.clone()))?
            .next;
        assert!(reach.time() > start && reach.time() <= target);
        check_water(&reach)?;
        steps += 1;
        assert!(steps < 1_000_000, "Triangular reach did not reach {target}");
    }
    assert_eq!(reach.time().to_bits(), target.to_bits());
    Ok(reach)
}

pub fn near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:e}, expected={expected:e}, tolerance={tolerance:e}"
    );
}

pub fn check_water(reach: &Reach) -> TestResult {
    let ledger = reach.ledger();
    let scale = ledger
        .initial_storage_m3
        .max(ledger.storage_m3)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3)
        .max(ledger.lateral_in_m3)
        .max(ledger.lateral_out_m3)
        .max(1.0);
    let tolerance = 1e-9 + 1e-12 * scale;
    assert!(ledger.numerical_transfer_underflow_bound_m3.is_finite());
    assert!(ledger.numerical_transfer_underflow_bound_m3 >= 0.0);
    assert!(ledger.lateral_counter_abs_bound_m3.is_finite());
    assert!(ledger.lateral_counter_abs_bound_m3 >= 0.0);
    assert!(ledger.lateral_assembly_abs_bound_m3.is_finite());
    assert!(ledger.lateral_assembly_abs_bound_m3 >= 0.0);
    assert!(
        ledger.residual_m3.abs()
            + ledger.numerical_transfer_underflow_bound_m3
            + ledger.lateral_counter_abs_bound_m3
            + ledger.lateral_assembly_abs_bound_m3
            <= tolerance
    );
    let state = reach.state()?;
    for cell in &state {
        assert!(cell.volume_m3 >= 0.0);
        assert!(cell.depth_m >= 0.0);
        if cell.dry {
            assert_eq!(cell.volume_m3, 0.0);
            assert_eq!(cell.longitudinal_momentum_m4_per_s, 0.0);
            assert_eq!(cell.stage_m, None);
            assert_eq!(cell.velocity_m_per_s, None);
        }
    }
    if let Some(step) = reach.last_step()? {
        let scale = step
            .storage_before_m3
            .max(step.storage_after_m3)
            .max(step.boundary_in_m3)
            .max(step.boundary_out_m3)
            .max(step.lateral_in_m3)
            .max(step.lateral_out_m3)
            .max(1.0);
        let tolerance = 1e-9 + 1e-12 * scale;
        assert!(step.transfer_underflow_bound_m3.is_finite());
        assert!(step.transfer_underflow_bound_m3 >= 0.0);
        assert!(step.lateral_assembly_abs_bound_m3.is_finite());
        assert!(step.lateral_assembly_abs_bound_m3 >= 0.0);
        assert!(step.lateral_counter_addition_abs_bound_m3.is_finite());
        assert!(step.lateral_counter_addition_abs_bound_m3 >= 0.0);
        assert!(
            step.residual_m3.abs()
                + step.transfer_underflow_bound_m3
                + step.lateral_assembly_abs_bound_m3
                <= tolerance
        );
        assert_eq!(step.face_transfer_underflow_bound_m3.len(), state.len() + 1);
        for bound in &step.face_transfer_underflow_bound_m3 {
            assert!(bound.is_finite());
            assert!(*bound >= 0.0);
        }
        assert!(step.max_local_residual_m3 <= tolerance);
        let momentum_scale = state
            .iter()
            .map(|cell| cell.longitudinal_momentum_m4_per_s.abs())
            .chain(step.face_momentum_m4_per_s.iter().map(|value| value.abs()))
            .chain(
                step.left_pressure_impulse_m4_per_s
                    .iter()
                    .map(|value| value.abs()),
            )
            .chain(
                step.right_pressure_impulse_m4_per_s
                    .iter()
                    .map(|value| value.abs()),
            )
            .chain(
                step.friction_impulse_m4_per_s
                    .iter()
                    .map(|value| value.abs()),
            )
            .fold(1.0, f64::max);
        let momentum_tolerance =
            reach.config().momentum_absolute + reach.config().momentum_relative * momentum_scale;
        assert!(step.max_local_momentum_residual_m4_per_s.is_finite());
        assert!(step.max_local_momentum_residual_m4_per_s <= momentum_tolerance);
        assert_eq!(step.momentum_underflow_bound_m4_per_s.len(), state.len());
        for bound in &step.momentum_underflow_bound_m4_per_s {
            assert!(bound.is_finite());
            assert!(*bound >= 0.0);
            assert!(*bound <= momentum_tolerance);
        }
        let maximum_underflow_bound = step
            .momentum_underflow_bound_m4_per_s
            .iter()
            .copied()
            .fold(0.0, f64::max);
        assert_eq!(
            step.max_momentum_underflow_bound_m4_per_s,
            maximum_underflow_bound
        );
        assert_eq!(step.lateral_momentum_abs_bound_m4_per_s.len(), state.len());
        for bound in &step.lateral_momentum_abs_bound_m4_per_s {
            assert!(bound.is_finite());
            assert!(*bound >= 0.0);
            assert!(*bound <= momentum_tolerance);
        }
        let riemann = step
            .riemann
            .as_ref()
            .ok_or("Missing triangular Riemann certificate")?;
        let faces = state.len() + 1;
        assert_eq!(riemann.water_flux_width_m3_per_s.len(), faces);
        assert_eq!(riemann.momentum_flux_width_m4_per_s2.len(), faces);
        let dt = step.end_s - step.start_s;
        for (widths, transfers, absolute) in [
            (
                &riemann.water_flux_width_m3_per_s,
                &step.face_volume_m3,
                reach.config().riemann_water_flux_absolute_m3_per_s,
            ),
            (
                &riemann.momentum_flux_width_m4_per_s2,
                &step.face_momentum_m4_per_s,
                reach.config().riemann_momentum_flux_absolute_m4_per_s2,
            ),
        ] {
            for (&width, &transfer) in widths.iter().zip(transfers) {
                assert!(width.is_finite() && width >= 0.0);
                // The selected flux lies inside its bracket, so |flux| + width bounds either endpoint.
                let scale = (transfer / dt).abs() + width;
                let budget = absolute + reach.config().riemann_flux_relative * scale;
                assert!(width <= budget, "Flux width={width:e}, budget={budget:e}");
            }
        }
        assert_eq!(
            riemann.max_water_flux_width_m3_per_s,
            riemann
                .water_flux_width_m3_per_s
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert_eq!(
            riemann.max_momentum_flux_width_m4_per_s2,
            riemann
                .momentum_flux_width_m4_per_s2
                .iter()
                .copied()
                .fold(0.0, f64::max)
        );
        assert!(riemann.root_iterations <= reach.config().root_iterations);
        assert!(dt <= step.cfl_limit_s);
        assert!(dt <= step.depletion_limit_s);
        assert!(dt <= step.accepted_start_cfl_limit_s);
        if let Some(candidate_limit) = step.candidate_cfl_limit_s {
            assert!(dt <= candidate_limit);
        }
    }
    Ok(())
}
