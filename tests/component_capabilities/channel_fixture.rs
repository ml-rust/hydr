//! Minimal reaches of each channel kind for capability checks.

use hydr::capability::ComponentCapabilities;
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelError, ChannelForcingInterval, ChannelGeometry,
    ChannelInitialState, ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput,
    ChannelReach, ChannelReachDefinition, ChannelSurveyedConfig, CrossSectionDefinition,
    CrossSectionPoint, RectangularChannelCell, SurveyedChannelCell, TriangularChannelCell,
    rectangular_reach_capabilities, surveyed_reach_capabilities, triangular_reach_capabilities,
};
use hydr::coupling::{
    ChannelPoolBatch, ChannelPoolConfig, ChannelPoolConnection, ChannelPoolEndpoint,
    ChannelPoolError, ChannelPoolInitialState, ChannelPoolReach, ChannelPoolSetup,
};
use hydr::forcing::WaterAmount;
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};

pub type TestResult = Result<(), Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<crate::runtime::Runtime>;
pub type Case = Result<Reach, Box<dyn std::error::Error>>;

const DATUM: &str = "Synthetic capability datum";

/// Cell length in metres. Waves at unit depth cross it in more than one second.
const LENGTH: f64 = 16.0;

/// Requested interval in seconds, far below every fixture's CFL limit.
pub const END: f64 = 1.0 / 1024.0;

/// Highest supported depth in metres.
const CAP: f64 = 4.0;

/// Channel reach kind, one per declared component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Rectangular,
    Triangular,
    Surveyed,
}

pub const KINDS: [Kind; 3] = [Kind::Rectangular, Kind::Triangular, Kind::Surveyed];

impl Kind {
    pub fn capabilities(self) -> &'static ComponentCapabilities {
        match self {
            Self::Rectangular => rectangular_reach_capabilities(),
            Self::Triangular => triangular_reach_capabilities(),
            Self::Surveyed => surveyed_reach_capabilities(),
        }
    }

    /// Kind of a public geometry. A new geometry variant fails to compile here.
    pub fn of(geometry: &ChannelGeometry) -> Self {
        match geometry {
            ChannelGeometry::Rectangular { .. } => Self::Rectangular,
            ChannelGeometry::PrismaticTriangular { .. } => Self::Triangular,
            ChannelGeometry::PrismaticSurveyed { .. } => Self::Surveyed,
        }
    }

    /// Submerged area at depth `h`. The rectangle is 2 m wide. The triangle has unit bank
    /// slopes. The surveyed trapezoid has a 2 m base and unit side slopes.
    pub fn area(self, h: f64) -> f64 {
        match self {
            Self::Rectangular => 2.0 * h,
            Self::Triangular => h * h,
            Self::Surveyed => 2.0 * h + h * h,
        }
    }

    fn geometry(self, count: usize) -> ChannelGeometry {
        let id = |index| format!("cell-{index}");
        match self {
            Self::Rectangular => ChannelGeometry::Rectangular {
                cells: (0..count)
                    .map(|index| RectangularChannelCell {
                        id: id(index),
                        length_m: LENGTH,
                        width_m: 2.0,
                        bed_elevation_m: 0.0,
                        maximum_depth_m: CAP,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            Self::Triangular => ChannelGeometry::PrismaticTriangular {
                left_bank_slope_h_per_v: 1.0,
                right_bank_slope_h_per_v: 1.0,
                bed_elevation_m: 0.0,
                cells: (0..count)
                    .map(|index| TriangularChannelCell {
                        id: id(index),
                        length_m: LENGTH,
                        maximum_depth_m: CAP,
                        manning_n: 0.0,
                    })
                    .collect(),
            },
            Self::Surveyed => ChannelGeometry::PrismaticSurveyed {
                profile: CrossSectionDefinition {
                    id: "unit-trapezoid".into(),
                    vertical_datum: DATUM.into(),
                    points: [(-5.0, 4.0), (-1.0, 0.0), (1.0, 0.0), (5.0, 4.0)]
                        .into_iter()
                        .map(|(station_m, elevation_m)| CrossSectionPoint {
                            station_m,
                            elevation_m,
                        })
                        .collect(),
                    maximum_stage_m: CAP,
                    profile_provenance: "Explicit analytical trapezoid survey".into(),
                },
                cells: (0..count)
                    .map(|index| SurveyedChannelCell {
                        id: id(index),
                        length_m: LENGTH,
                        manning_n: 0.0,
                    })
                    .collect(),
                numerical: ChannelSurveyedConfig::default(),
            },
        }
    }
}

/// Reach definition with `count` cells. `lateral` binds one inflow and one withdrawal to `cell-0`.
pub fn definition(kind: Kind, count: usize, lateral: bool) -> ChannelReachDefinition {
    let lateral_bindings = if lateral {
        [
            ("in", ChannelLateralDirection::Inflow),
            ("out", ChannelLateralDirection::Withdrawal),
        ]
        .into_iter()
        .map(|(id, direction)| ChannelLateralBinding {
            boundary_id: id.into(),
            cell_id: "cell-0".into(),
            direction,
            ownership_assertion: format!("Independent synthetic lateral {id}"),
        })
        .collect()
    } else {
        Vec::new()
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit analytical capability reach".into(),
        geometry: kind.geometry(count),
        lateral_bindings,
        config: ChannelConfig::default(),
    }
}

/// Initial state from depths and velocities per cell.
pub fn initial(kind: Kind, depths: &[f64], velocities: &[f64]) -> ChannelInitialState {
    let areas = depths.iter().map(|&h| kind.area(h));
    ChannelInitialState {
        time_s: 0.0,
        volume_m3: areas.clone().map(|area| area * LENGTH).collect(),
        longitudinal_momentum_m4_per_s: areas
            .zip(velocities)
            .map(|(area, u)| area * u * LENGTH)
            .collect(),
    }
}

/// Reach of `kind` without lateral inputs, one cell per depth.
pub fn reach(kind: Kind, depths: &[f64], velocities: &[f64]) -> Case {
    Ok(ChannelReach::new(
        crate::runtime::client(),
        definition(kind, depths.len(), false),
        initial(kind, depths, velocities),
    )?)
}

pub fn exterior(depth_m: f64, velocity_m_per_s: f64) -> ChannelBoundary {
    ChannelBoundary::Exterior {
        depth_m,
        velocity_m_per_s,
    }
}

/// Held endpoint policies over `[0, END]` without lateral inputs.
pub fn interval(left: ChannelBoundary, right: ChannelBoundary) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s: 0.0,
        end_s: END,
        left,
        right,
        lateral_inputs: Vec::new(),
    }
}

/// Lateral inputs for `definition(_, _, true)`, both of `amount`.
pub fn lateral_inputs(amount: WaterAmount) -> Vec<ChannelLateralInput> {
    vec![
        ChannelLateralInput::Inflow {
            boundary_id: "in".into(),
            amount,
            longitudinal_velocity_m_per_s: 0.0,
        },
        ChannelLateralInput::Withdrawal {
            boundary_id: "out".into(),
            amount,
        },
    ]
}

/// Binds `forcing`, runs one step, and checks that it is accepted.
pub fn admits(case: Case, forcing: &ChannelForcingInterval) -> TestResult {
    let origin = case?;
    let next = origin.step(END, forcing)?.next;
    assert!(next.time() > origin.time(), "step accepted no time");
    assert!(next.last_step()?.is_some(), "step recorded no receipt");
    Ok(())
}

/// Error class a rejection reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Variant {
    Input,
    Numerical,
}

/// Rejection variant and message, or `None` when the case is accepted.
pub type Outcome = Result<Option<(Variant, String)>, Box<dyn std::error::Error>>;

/// Binds `forcing` and runs one step.
pub fn step_outcome(case: Case, forcing: &ChannelForcingInterval) -> Outcome {
    let origin = case?;
    let before = origin.snapshot_data()?;
    let outcome = match origin.step(END, forcing) {
        Ok(_) => None,
        Err(ChannelError::InvalidInput { context, .. }) => Some((Variant::Input, context)),
        Err(ChannelError::Numerical { context, .. }) => Some((Variant::Numerical, context)),
        Err(ChannelError::Unresolved { reason, .. }) => Some((Variant::Numerical, reason)),
        Err(other) => return Err(format!("unexpected channel rejection: {other}").into()),
    };
    assert_eq!(origin.snapshot_data()?, before, "rejection changed state");
    Ok(outcome)
}

/// Connects the right end of one resting reach of `kind` to a unit-depth level pool.
pub fn pool_outcome(kind: Kind) -> Outcome {
    let area = 4.0;
    let curve = LevelPoolCurve::new(
        vec![
            LevelPoolKnot {
                volume_m3: 0.0,
                stage_m: 0.0,
            },
            LevelPoolKnot {
                volume_m3: area * CAP,
                stage_m: CAP,
            },
        ],
        DATUM.into(),
        "Explicit constant-area analytical basin".into(),
    )?;
    let setup = ChannelPoolSetup {
        reaches: vec![ChannelPoolReach {
            id: "reach".into(),
            definition: definition(kind, 1, false),
        }],
        pools: vec![LevelPoolElement {
            id: "pool".into(),
            curve,
            connected_pool_assertion: "Connected approximately level synthetic basin".into(),
            storage_support_identity: "finite-basin-pool".into(),
        }],
        connections: vec![ChannelPoolConnection {
            id: "connection".into(),
            reach_id: "reach".into(),
            endpoint: ChannelPoolEndpoint::Right,
            pool_id: "pool".into(),
            entrance_loss_coefficient: 0.0,
            physical_connection_assertion: "Unobstructed named channel endpoint at its actual bed"
                .into(),
            quiescent_reservoir_assertion: "Negligible basin velocity with complete exit mixing"
                .into(),
        }],
        vertical_datum: DATUM.into(),
        nonoverlap_assertion: "Basin and reach occupy separate physical storage".into(),
        config: ChannelPoolConfig {
            storage: LevelPoolConfig::default(),
            head_absolute_m: 1e-12,
            head_relative: 1e-12,
            transfer_absolute_m3: 1e-12,
            transfer_relative: 1e-12,
            mass_absolute_m3: 1e-9,
            mass_relative: 1e-12,
            minimum_step_s: 1e-10,
            maximum_step_s: 1.0,
            iteration_limit: 64,
            retry_limit: 16,
        },
    };
    let initial = ChannelPoolInitialState {
        time_s: 0.0,
        reaches: vec![initial(kind, &[1.0], &[0.0])],
        pools: vec![LevelPoolState { volume_m3: area }],
    };
    match ChannelPoolBatch::<crate::runtime::Runtime>::new(crate::runtime::client(), setup, initial)
    {
        Ok(_) => Ok(None),
        Err(ChannelPoolError::Input(message)) => Ok(Some((Variant::Input, message))),
        Err(other) => Err(format!("unexpected channel/pool rejection: {other}").into()),
    }
}
