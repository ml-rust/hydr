//! Standalone rectangular, triangular and surveyed reaches fed by one lateral inflow.
//!
//! A nonzero lateral source makes every trial evaluate the candidate-state wave bounds.
//! The frozen digests and the standalone profiler step these same reaches.

use crate::runtime::{Runtime, client};
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelForcingInterval, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReach,
    ChannelReachDefinition, ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint,
    RectangularChannelCell, SurveyedChannelCell, TriangularChannelCell,
};
use hydr::forcing::WaterAmount;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Reach = ChannelReach<Runtime>;

/// Inflow binding on the middle cell of every fixture.
const INLET: &str = "inlet";
/// Requested span of one interval, in seconds.
const SPAN_S: f64 = 0.05;

/// A standalone fixture geometry.
#[derive(Clone, Copy, Debug)]
pub enum Kind {
    Rectangular,
    Triangular,
    Surveyed,
}

impl Kind {
    pub const ALL: [Self; 3] = [Self::Rectangular, Self::Triangular, Self::Surveyed];

    /// The fixture named `name`: `rect`, `triangular` or `surveyed`.
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "rect" => Some(Self::Rectangular),
            "triangular" => Some(Self::Triangular),
            "surveyed" => Some(Self::Surveyed),
            _ => None,
        }
    }

    /// Lateral inflow rate, in m³/s.
    fn inflow(self) -> f64 {
        match self {
            Self::Rectangular | Self::Triangular => 0.5,
            Self::Surveyed => 2.0,
        }
    }
}

fn binding(cell: &str) -> Vec<ChannelLateralBinding> {
    vec![ChannelLateralBinding {
        boundary_id: INLET.into(),
        cell_id: cell.into(),
        direction: ChannelLateralDirection::Inflow,
        ownership_assertion: "standalone wave-bound fixture inflow".into(),
    }]
}

fn rectangular() -> Result<Reach> {
    let cells = (0..3)
        .map(|i| RectangularChannelCell {
            id: format!("cell-{i}"),
            length_m: 2.0,
            width_m: 2.0,
            bed_elevation_m: 0.0,
            maximum_depth_m: 4.0,
            manning_n: 0.03,
        })
        .collect();
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic metres".into(),
        provenance: "rectangular standalone wave-bound fixture".into(),
        geometry: ChannelGeometry::Rectangular { cells },
        lateral_bindings: binding("cell-1"),
        config: ChannelConfig::default(),
    };
    Ok(ChannelReach::new(
        client(),
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![4.0, 2.4, 3.2],
            longitudinal_momentum_m4_per_s: vec![0.8, -0.1, 0.0],
        },
    )?)
}

fn triangular() -> Result<Reach> {
    let cells = (0..3)
        .map(|i| TriangularChannelCell {
            id: format!("cell-{i}"),
            length_m: 2.0,
            maximum_depth_m: 10.0,
            manning_n: 0.03,
        })
        .collect();
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic metres".into(),
        provenance: "triangular standalone wave-bound fixture".into(),
        geometry: ChannelGeometry::PrismaticTriangular {
            left_bank_slope_h_per_v: 1.0,
            right_bank_slope_h_per_v: 1.0,
            bed_elevation_m: 0.0,
            cells,
        },
        lateral_bindings: binding("cell-1"),
        config: ChannelConfig::default(),
    };
    Ok(ChannelReach::new(
        client(),
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![2.0, 1.0, 1.6],
            longitudinal_momentum_m4_per_s: vec![0.4, 0.0, -0.2],
        },
    )?)
}

fn surveyed() -> Result<Reach> {
    let points = [(-5.0, 4.0), (-1.0, 0.0), (1.0, 0.0), (5.0, 4.0)]
        .into_iter()
        .map(|(station_m, elevation_m)| CrossSectionPoint {
            station_m,
            elevation_m,
        })
        .collect();
    let definition = ChannelReachDefinition {
        vertical_datum: "synthetic metres".into(),
        provenance: "surveyed standalone wave-bound fixture".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "standalone-trapezoid".into(),
                vertical_datum: "synthetic metres".into(),
                points,
                maximum_stage_m: 4.0,
                profile_provenance: "Explicit integer trapezoid survey".into(),
            },
            cells: (0..3)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 4.0,
                    manning_n: 0.03,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        lateral_bindings: binding("cell-1"),
        config: ChannelConfig::default(),
    };
    Ok(ChannelReach::new(
        client(),
        definition,
        ChannelInitialState {
            time_s: 0.0,
            volume_m3: vec![12.0, 8.0, 10.0],
            longitudinal_momentum_m4_per_s: vec![4.0, 0.0, -2.0],
        },
    )?)
}

/// The fixture reach of `kind` at its initial state.
pub fn reach(kind: Kind) -> Result<Reach> {
    match kind {
        Kind::Rectangular => rectangular(),
        Kind::Triangular => triangular(),
        Kind::Surveyed => surveyed(),
    }
}

/// Wall forcing with the fixture inflow from `start_s` to `end_s`.
pub fn forcing(kind: Kind, start_s: f64, end_s: f64) -> ChannelForcingInterval {
    ChannelForcingInterval {
        start_s,
        end_s,
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: vec![ChannelLateralInput::Inflow {
            boundary_id: INLET.into(),
            amount: WaterAmount::Rate {
                rate_m3_per_s: kind.inflow(),
            },
            longitudinal_velocity_m_per_s: 0.0,
        }],
    }
}

/// Accepts one interval of at most one span.
pub fn step(kind: Kind, reach: &Reach) -> Result<Reach> {
    let start = reach.time();
    let end = start + SPAN_S;
    Ok(reach.step(end, &forcing(kind, start, end))?.next)
}
