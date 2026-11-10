use hydr::channel::{
    ChannelBoundary, ChannelBoundaryInterval, ChannelConfig, ChannelInitialState, ChannelReach,
    ChannelReachDefinition, RectangularChannelCell,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<crate::runtime::Runtime>;

pub const GRAVITY: f64 = 9.81;

pub fn cell(index: usize, length: f64, width: f64, bed: f64) -> RectangularChannelCell {
    RectangularChannelCell {
        id: format!("cell-{index}"),
        length_m: length,
        width_m: width,
        bed_elevation_m: bed,
        maximum_depth_m: 10.0,
        manning_n: 0.0,
    }
}

pub fn definition(cells: Vec<RectangularChannelCell>, max_step: f64) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: "synthetic SI datum".into(),
        provenance: "Explicit analytical rectangular channel fixture".into(),
        cells,
        config: ChannelConfig {
            max_step,
            ..ChannelConfig::default()
        },
    }
}

pub fn reach(
    definition: ChannelReachDefinition,
    depths: &[f64],
    discharges: &[f64],
) -> Result<Reach, Box<dyn std::error::Error>> {
    assert_eq!(depths.len(), definition.cells.len());
    assert_eq!(discharges.len(), definition.cells.len());
    let volume_m3 = definition
        .cells
        .iter()
        .zip(depths)
        .map(|(cell, depth)| cell.length_m * cell.width_m * depth)
        .collect();
    let longitudinal_momentum_m4_per_s = definition
        .cells
        .iter()
        .zip(discharges)
        .map(|(cell, discharge)| cell.length_m * discharge)
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

pub fn walls() -> [ChannelBoundary; 2] {
    [ChannelBoundary::Wall, ChannelBoundary::Wall]
}

pub fn exterior(depth: f64, velocity: f64) -> ChannelBoundary {
    ChannelBoundary::Exterior {
        depth_m: depth,
        velocity_m_per_s: velocity,
    }
}

pub fn advance(
    mut reach: Reach,
    target: f64,
    boundaries: [ChannelBoundary; 2],
) -> Result<Reach, Box<dyn std::error::Error>> {
    let mut steps = 0usize;
    while reach.time() < target {
        let start = reach.time();
        let forcing = ChannelBoundaryInterval {
            start_s: start,
            end_s: target,
            left: boundaries[0].clone(),
            right: boundaries[1].clone(),
        };
        let next = reach.step(target, &forcing)?.next;
        assert!(next.time() > start);
        assert!(next.time() <= target);
        check_water(&next)?;
        reach = next;
        steps += 1;
        assert!(steps < 1_000_000, "channel did not reach target={target}");
    }
    assert_eq!(reach.time().to_bits(), target.to_bits());
    Ok(reach)
}

pub fn check_water(reach: &Reach) -> TestResult {
    let ledger = reach.ledger();
    let scale = ledger
        .initial_storage_m3
        .max(ledger.storage_m3)
        .max(ledger.boundary_in_m3)
        .max(ledger.boundary_out_m3)
        .max(1.0);
    let tolerance = 1e-9 + 1e-12 * scale;
    assert!(ledger.numerical_transfer_underflow_bound_m3.is_finite());
    assert!(ledger.numerical_transfer_underflow_bound_m3 >= 0.0);
    assert!(ledger.residual_m3.abs() + ledger.numerical_transfer_underflow_bound_m3 <= tolerance);
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
            .max(1.0);
        let tolerance = 1e-9 + 1e-12 * scale;
        assert!(step.transfer_underflow_bound_m3.is_finite());
        assert!(step.transfer_underflow_bound_m3 >= 0.0);
        assert!(step.residual_m3.abs() + step.transfer_underflow_bound_m3 <= tolerance);
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
        let dt = step.end_s - step.start_s;
        assert!(dt <= step.cfl_limit_s);
        assert!(dt <= step.depletion_limit_s);
    }
    Ok(())
}
