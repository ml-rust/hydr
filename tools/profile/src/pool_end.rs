//! A three-band surveyed reach cell whose right end fills a constant-area pond, then drains it.
//!
//! This is the `a_surveyed_reach_end_fills_a_pool_then_reverses` case of the coupling crate,
//! built through the public coordinator API. One dry 2 m square surface cell sits beside the
//! reach behind impermeable walls. The pond owns the reach's right end through a pool end.

use crate::fixture::Result;
use hydr::channel::{
    ChannelBoundary, ChannelConfig, ChannelEnd, ChannelGeometry, ChannelInitialState,
    ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput, ChannelReachDefinition,
    ChannelSurveyedConfig, CrossSectionDefinition, CrossSectionPoint, SurveyedChannelCell,
};
use hydr::coupling::{
    CoupledPoolSupport, SurfaceChannelConfig, SurfaceChannelForcingInterval,
    SurfaceChannelInitialState, SurfaceChannelPoolEnd, SurfaceChannelReach,
    SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use hydr::forcing::WaterAmount;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::{
    LevelPoolConfig, LevelPoolCurve, LevelPoolElement, LevelPoolKnot, LevelPoolState,
};

const DATUM: &str = "Synthetic structure datum";
const REACH: &str = "r";
const INFLOW: &str = "caller:inflow";
const WITHDRAWAL: &str = "caller:withdrawal";
/// Plan area of the pond, in m².
const POND_AREA: f64 = 8.0;
/// Initial reach cell depth, in m.
const DEPTH: f64 = 0.5;
/// Intervals that fill the pond before the caller withdrawal drains it.
const FILL: usize = 8;

/// Requested duration of every interval, in s.
pub const SPAN_S: f64 = 0.5;

/// Three-band survey: bottom band 2 m wide with unit side slopes, cap 3 m.
const THREE_BANDS: [(f64, f64); 8] = [
    (-4.0, 3.0),
    (-2.5, 2.0),
    (-2.0, 1.0),
    (-1.0, 0.0),
    (1.0, 0.0),
    (2.0, 1.0),
    (2.5, 2.0),
    (4.0, 3.0),
];

fn reach() -> ChannelReachDefinition {
    let binding = |boundary_id: &str, direction| ChannelLateralBinding {
        boundary_id: boundary_id.into(),
        cell_id: "cell-0".into(),
        direction,
        ownership_assertion: "Caller water at the reach cell".into(),
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit surveyed finite-pool reach".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "three-bands".into(),
                vertical_datum: DATUM.into(),
                points: THREE_BANDS
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 3.0,
                profile_provenance: "Frozen three-band acceptance survey".into(),
            },
            cells: vec![SurveyedChannelCell {
                id: "cell-0".into(),
                length_m: 2.0,
                manning_n: 0.0,
            }],
            numerical: ChannelSurveyedConfig::default(),
        },
        lateral_bindings: vec![
            binding(INFLOW, ChannelLateralDirection::Inflow),
            binding(WITHDRAWAL, ChannelLateralDirection::Withdrawal),
        ],
        config: ChannelConfig::default(),
    }
}

fn pond() -> Result<LevelPoolElement> {
    let knot = |volume_m3, stage_m| LevelPoolKnot { volume_m3, stage_m };
    Ok(LevelPoolElement {
        id: "pond".into(),
        curve: LevelPoolCurve::new(
            vec![knot(0.0, 0.0), knot(POND_AREA * 2.0, 2.0)],
            DATUM.into(),
            "Synthetic constant-area pond".into(),
        )?,
        connected_pool_assertion: "One level water body".into(),
        storage_support_identity: "pond-basin".into(),
    })
}

/// Setup and initial state of the case.
pub fn parts() -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let grid = CartesianGrid::new(1, 1, 2.0, 2.0)?;
    let domain = Domain::new(
        grid,
        vec![0.0],
        vec![0.0],
        vec![true],
        SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 1000.0,
            rain: vec![0.0],
            potential_evaporation: vec![0.0],
            withdrawal: vec![0.0],
            boundaries: Vec::new(),
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        vec![CellState {
            h: 0.0,
            qx: 0.0,
            qy: 0.0,
        }],
    )?;
    let pond = pond()?;
    let support = vec![CoupledPoolSupport {
        pool_id: pond.id.clone(),
        zero_volume_stage_m: pond.curve.origin(),
        zero_volume_limit_assertion: "The pond empties at its curve origin".into(),
    }];
    let end = SurfaceChannelPoolEnd {
        id: "outfall".into(),
        reach_id: REACH.into(),
        end: ChannelEnd::Right,
        pool_id: pond.id.clone(),
        entrance_loss_coefficient: 0.5,
        physical_connection_assertion: "The reach outfall opens into the pond at its bed".into(),
        quiescent_reservoir_assertion: "The pond is quiescent and mixes the outfall water".into(),
    };
    let setup = SurfaceChannelSetup {
        surface: SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        },
        reaches: vec![SurfaceChannelReach {
            id: REACH.into(),
            definition: reach(),
        }],
        links: Vec::new(),
        ports: Vec::new(),
        pools: vec![pond],
        pool_support: support,
        pool_sources: None,
        pool_inflows: Vec::new(),
        storage: LevelPoolConfig::default(),
        structures: Vec::new(),
        pool_ends: vec![end],
        operating: None,
        geometry_schedules: Vec::new(),
        pool_curve_schedules: Vec::new(),
        surface_bed_schedules: Vec::new(),
        channel_section_schedules: Vec::new(),
        runoff: None,
        storage_exclusion_evidence: "The reach tank lies outside the surface cell".into(),
        config: SurfaceChannelConfig {
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
            contraction_target: 0.5,
        },
    };
    let initial = SurfaceChannelInitialState {
        time_s: 0.0,
        surface: InitialState {
            time: 0.0,
            flow,
            soil,
        },
        reaches: vec![(
            REACH.into(),
            ChannelInitialState {
                time_s: 0.0,
                volume_m3: vec![2.0 * DEPTH * (2.0 + DEPTH)],
                longitudinal_momentum_m4_per_s: vec![0.0],
            },
        )],
        pools: vec![LevelPoolState {
            volume_m3: 0.45 * POND_AREA,
        }],
        runoff: Vec::new(),
    };
    Ok((setup, initial))
}

/// Interval `index` from `start` to `end`: a wall on the left, the pool end on the right, and
/// a caller inflow for the first intervals, then a caller withdrawal.
pub fn forcing(
    _setup: &SurfaceChannelSetup,
    start: f64,
    end: f64,
    index: usize,
) -> SurfaceChannelForcingInterval {
    let rate = |rate_m3_per_s| WaterAmount::Rate { rate_m3_per_s };
    let (into, out) = if index % 20 < FILL {
        (0.02, 0.0)
    } else {
        (0.0, 0.04)
    };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![SurfaceChannelReachForcing {
            reach_id: REACH.into(),
            left: Some(ChannelBoundary::Wall),
            right: None,
            lateral_inputs: vec![
                ChannelLateralInput::Inflow {
                    boundary_id: INFLOW.into(),
                    amount: rate(into),
                    longitudinal_velocity_m_per_s: 0.0,
                },
                ChannelLateralInput::Withdrawal {
                    boundary_id: WITHDRAWAL.into(),
                    amount: rate(out),
                },
            ],
        }],
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}
