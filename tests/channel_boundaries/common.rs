use hydr::channel::{
    ChannelBoundary, ChannelBoundaryTrace, ChannelConfig, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelReach, ChannelReachDefinition, RectangularChannelCell,
    TriangularChannelCell,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<crate::runtime::Runtime>;

#[derive(Clone, Copy, Debug)]
pub enum Shape {
    Rectangle { gravity: f64, width: f64 },
    Triangle { gravity: f64, coefficient: f64 },
}

impl Shape {
    pub fn gravity(self) -> f64 {
        match self {
            Self::Rectangle { gravity, .. } | Self::Triangle { gravity, .. } => gravity,
        }
    }
    pub fn area(self, h: f64) -> f64 {
        match self {
            Self::Rectangle { width, .. } => width * h,
            Self::Triangle { coefficient, .. } => coefficient * h * h,
        }
    }
    pub fn pressure(self, h: f64) -> f64 {
        match self {
            Self::Rectangle { gravity, width } => gravity * width * h * h / 2.0,
            Self::Triangle {
                gravity,
                coefficient,
            } => gravity * coefficient * h * h * h / 3.0,
        }
    }
    pub fn sound(self, h: f64) -> f64 {
        match self {
            Self::Rectangle { gravity, .. } => (gravity * h).sqrt(),
            Self::Triangle { gravity, .. } => (gravity * h / 2.0).sqrt(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Side {
    Left,
    Right,
}
impl Side {
    pub fn index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
    pub fn sign(self) -> f64 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
    pub fn face(self, count: usize) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => count,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Policy {
    Stage,
    Discharge,
}
impl Policy {
    pub fn boundary(self, side: Side, depth: f64, outward_q: f64) -> ChannelBoundary {
        match self {
            Self::Stage => ChannelBoundary::Stage { elevation_m: depth },
            Self::Discharge => ChannelBoundary::Discharge {
                discharge_m3_per_s: side.sign() * outward_q,
            },
        }
    }
}

pub fn definition(
    shape: Shape,
    count: usize,
    length: f64,
    cap: f64,
    max_step: f64,
) -> ChannelReachDefinition {
    let geometry = match shape {
        Shape::Rectangle { width, .. } => ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    width_m: width,
                    bed_elevation_m: 0.0,
                    maximum_depth_m: cap,
                    manning_n: 0.0,
                })
                .collect(),
        },
        Shape::Triangle { coefficient, .. } => ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: coefficient,
            right_bank_slope_h_per_v: coefficient,
            bed_elevation_m: 0.0,
            cells: (0..count)
                .map(|i| TriangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    maximum_depth_m: cap,
                    manning_n: 0.0,
                })
                .collect(),
        },
    };
    ChannelReachDefinition {
        vertical_datum: "Synthetic SI datum".into(),
        provenance: "Explicit scalar endpoint acceptance geometry".into(),
        geometry,
        lateral_bindings: Vec::new(),
        config: ChannelConfig {
            gravity: shape.gravity(),
            max_step,
            ..ChannelConfig::default()
        },
    }
}

pub fn initial(
    shape: Shape,
    lengths: &[f64],
    depths: &[f64],
    velocities: &[f64],
) -> ChannelInitialState {
    let volume_m3: Vec<_> = lengths
        .iter()
        .zip(depths)
        .map(|(dx, h)| dx * shape.area(*h))
        .collect();
    let longitudinal_momentum_m4_per_s = volume_m3
        .iter()
        .zip(velocities)
        .map(|(v, u)| v * u)
        .collect();
    ChannelInitialState {
        time_s: 0.0,
        volume_m3,
        longitudinal_momentum_m4_per_s,
    }
}

pub fn reach(
    layout: ChannelReachDefinition,
    initial: ChannelInitialState,
) -> Result<Reach, Box<dyn std::error::Error>> {
    Ok(ChannelReach::new(
        crate::runtime::client(),
        layout,
        initial,
    )?)
}

pub fn interval(
    side: Side,
    boundary: ChannelBoundary,
    start: f64,
    end: f64,
) -> ChannelForcingInterval {
    let mut ends = [
        ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: 0.0,
        },
        ChannelBoundary::Exterior {
            depth_m: 1.0,
            velocity_m_per_s: 0.0,
        },
    ];
    ends[side.index()] = boundary;
    ChannelForcingInterval {
        start_s: start,
        end_s: end,
        left: ends[0].clone(),
        right: ends[1].clone(),
        lateral_inputs: Vec::new(),
    }
}

pub fn close(actual: f64, expected: f64) {
    let scale = if expected == 0.0 { 1.0 } else { expected.abs() };
    assert!(
        (actual - expected).abs() <= 1e-12 * scale,
        "actual={actual:e}, expected={expected:e}"
    );
}

pub fn check_trace(
    trace: &ChannelBoundaryTrace,
    config: &ChannelConfig,
    shape: Shape,
    interior_depth: f64,
) {
    for value in [
        trace.depth_m,
        trace.area_m2,
        trace.outward_velocity_m_per_s,
        trace.outward_discharge_m3_per_s,
        trace.momentum_flux_m4_per_s2,
        trace.signal_speed_bound_m_per_s,
        trace.velocity_compatibility_abs_bound_m_per_s,
        trace.water_compatibility_abs_bound_m3_per_s,
        trace.momentum_compatibility_abs_bound_m4_per_s2,
        trace.water_flux_width_m3_per_s,
        trace.momentum_flux_width_m4_per_s2,
    ] {
        assert!(value.is_finite());
    }
    assert!(trace.depth_m > 0.0 && trace.area_m2 > 0.0);
    assert!(trace.outward_velocity_m_per_s.abs() < shape.sound(trace.depth_m));
    assert!(trace.signal_speed_bound_m_per_s > 0.0);
    for value in [
        trace.velocity_compatibility_abs_bound_m_per_s,
        trace.water_compatibility_abs_bound_m3_per_s,
        trace.momentum_compatibility_abs_bound_m4_per_s2,
        trace.water_flux_width_m3_per_s,
        trace.momentum_flux_width_m4_per_s2,
    ] {
        assert!(value >= 0.0);
    }
    let controls = &config.boundary;
    let velocity_scale = shape
        .sound(interior_depth)
        .max(shape.sound(trace.depth_m))
        .max(trace.outward_velocity_m_per_s.abs())
        + trace.velocity_compatibility_abs_bound_m_per_s;
    assert!(
        trace.velocity_compatibility_abs_bound_m_per_s
            <= controls.velocity_relative * velocity_scale
    );
    assert!(
        trace.water_compatibility_abs_bound_m3_per_s
            <= controls.discharge_absolute_m3_per_s
                + controls.discharge_relative
                    * (trace.outward_discharge_m3_per_s.abs()
                        + trace.water_compatibility_abs_bound_m3_per_s)
    );
    assert!(
        trace.momentum_compatibility_abs_bound_m4_per_s2
            <= controls.momentum_absolute_m4_per_s2
                + controls.momentum_relative
                    * (trace.momentum_flux_m4_per_s2.abs()
                        + trace.momentum_compatibility_abs_bound_m4_per_s2)
    );
    assert!(trace.root_iterations <= config.root_iterations);
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
    assert!(
        ledger.residual_m3.abs()
            + ledger.numerical_transfer_underflow_bound_m3
            + ledger.lateral_counter_abs_bound_m3
            + ledger.lateral_assembly_abs_bound_m3
            <= reach.config().volume_absolute + reach.config().volume_relative * scale
    );
    for cell in reach.state()? {
        assert!(cell.volume_m3.is_finite() && cell.volume_m3 >= 0.0);
        assert!(cell.depth_m.is_finite() && cell.depth_m >= 0.0);
        assert!(cell.longitudinal_momentum_m4_per_s.is_finite());
    }
    if let Some(step) = reach.last_step()? {
        let dt = step.end_s - step.start_s;
        assert!(
            dt > 0.0
                && dt <= step.cfl_limit_s
                && dt <= step.accepted_start_cfl_limit_s
                && dt <= step.depletion_limit_s
        );
        if let Some(limit) = step.candidate_cfl_limit_s {
            assert!(dt <= limit);
        }
        let momentum_scale = step
            .face_momentum_m4_per_s
            .iter()
            .map(|v| v.abs())
            .chain(step.friction_impulse_m4_per_s.iter().map(|v| v.abs()))
            .fold(1.0, f64::max);
        assert!(
            step.max_local_momentum_residual_m4_per_s
                <= reach.config().momentum_absolute
                    + reach.config().momentum_relative * momentum_scale
        );
    }
    Ok(())
}
