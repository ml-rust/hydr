//! Port test setups: a 4 x 2 surface grid of 2 m cells and two 4 m wide rectangular reaches.

use super::super::{
    SurfaceChannelConfig, SurfaceChannelDefinition, SurfaceChannelError,
    SurfaceChannelInitialState, SurfaceChannelLink, SurfaceChannelReach, SurfaceChannelSetup,
};
use super::SurfaceChannelPort;
use crate::channel::ChannelEnd;
use crate::channel::{
    ChannelConfig, ChannelGeometry, ChannelInitialState, ChannelReachDefinition,
    RectangularChannelCell,
};
use crate::solver::{
    Axis, BoundaryConditions, BoundaryFace, ForcingInterval, ForcingSchedule, InitialState,
    SolverConfig, SolverSetup,
};
use crate::structures::WeirApplicability;
use hydr_core::grid::CartesianGrid;
use hydr_core::model::{CellState, Domain, FlowState, SpatialReference};
use hydr_soil::soil::{LandModel, SoilState};

pub type TestResult<T = ()> = Result<T, Box<dyn std::error::Error>>;

pub const DATUM: &str = "Synthetic port datum";
pub const ALPHA: &str = "alpha";
pub const BETA: &str = "beta";
/// Bed of every `alpha` cell, in m.
pub const ALPHA_BED: f64 = 0.5;
/// Bed of every `beta` cell, in m.
pub const BETA_BED: f64 = 0.25;
/// Width of every reach cell, equal to two 2 m faces, in m.
pub const REACH_WIDTH: f64 = 4.0;

pub fn x_face(column: usize, row: usize) -> BoundaryFace {
    BoundaryFace {
        axis: Axis::X,
        column,
        row,
    }
}

pub fn y_face(column: usize, row: usize) -> BoundaryFace {
    BoundaryFace {
        axis: Axis::Y,
        column,
        row,
    }
}

pub fn port(
    id: &str,
    reach_id: &str,
    end: ChannelEnd,
    faces: Vec<BoundaryFace>,
) -> SurfaceChannelPort {
    SurfaceChannelPort {
        id: id.into(),
        reach_id: reach_id.into(),
        end,
        faces,
        mapping_evidence: "The reach end abuts these surface faces".into(),
    }
}

/// One port on each reach end, covering both ghost sides of both axes.
///
/// Declared out of identity order to exercise ordering.
pub fn four_ports() -> Vec<SurfaceChannelPort> {
    vec![
        port(
            "p-alpha-right",
            ALPHA,
            ChannelEnd::Right,
            vec![x_face(4, 1), x_face(4, 0)],
        ),
        port(
            "p-alpha-left",
            ALPHA,
            ChannelEnd::Left,
            vec![x_face(0, 0), x_face(0, 1)],
        ),
        port(
            "p-beta-left",
            BETA,
            ChannelEnd::Left,
            vec![y_face(0, 2), y_face(1, 2)],
        ),
        port(
            "p-beta-right",
            BETA,
            ChannelEnd::Right,
            vec![y_face(1, 0), y_face(0, 0)],
        ),
    ]
}

pub fn rectangular(count: usize, bed: f64) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Rectangular reach at the surface grid edge".into(),
        geometry: ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 2.0,
                    width_m: REACH_WIDTH,
                    bed_elevation_m: bed,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        },
        lateral_bindings: Vec::new(),
        config: ChannelConfig::default(),
    }
}

/// Bank weir between surface cell 1 and `alpha` cell 1, crest at 1.2 m.
pub fn link() -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: "bank".into(),
        surface_cell: 1,
        reach_id: ALPHA.into(),
        channel_cell: 1,
        crest_elevation_m: 1.2,
        crest_length_m: 2.0,
        weir_coefficient: 1.7,
        applicability: WeirApplicability {
            coefficient_provenance: "Declared bank coefficient".into(),
            free_nappe_and_approach_assertion: "Suppressed crest with quiescent approach".into(),
            submergence_correction_assertion: "Villemonte correction adopted".into(),
            symmetric_reversal: true,
        },
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "Surface cell 1 borders reach cell 1".into(),
    }
}

/// Surface setup and state of a 4 x 2 grid with `inactive` cells and impermeable walls.
fn surface(inactive: &[usize]) -> TestResult<(SolverSetup, InitialState)> {
    let grid = CartesianGrid::new(4, 2, 2.0, 2.0)?;
    let count = grid.cell_count();
    let active: Vec<bool> = (0..count).map(|cell| !inactive.contains(&cell)).collect();
    let domain = Domain::new(
        grid,
        vec![1.0; count],
        vec![0.0; count],
        active.clone(),
        SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 10.0,
            rain: vec![0.0; count],
            potential_evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: Vec::new(),
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(
        &domain,
        active
            .iter()
            .map(|&active| CellState {
                h: if active { 0.1 } else { 0.0 },
                qx: 0.0,
                qy: 0.0,
            })
            .collect(),
    )?;
    Ok((
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing,
            config: SolverConfig::default(),
        },
        InitialState {
            time: 0.0,
            flow,
            soil,
        },
    ))
}

/// Setup with reaches `alpha` and `beta`, one bank link, and `inactive` surface cells.
pub fn case(inactive: &[usize]) -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (surface, surface_state) = surface(inactive)?;
    let reach_state = ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![2.0; 3],
        longitudinal_momentum_m4_per_s: vec![0.0; 3],
    };
    Ok((
        SurfaceChannelSetup {
            surface,
            reaches: vec![
                SurfaceChannelReach {
                    id: ALPHA.into(),
                    definition: rectangular(3, ALPHA_BED),
                },
                SurfaceChannelReach {
                    id: BETA.into(),
                    definition: rectangular(3, BETA_BED),
                },
            ],
            links: vec![link()],
            ports: Vec::new(),
            storage_exclusion_evidence: "Reach storage lies outside every surface cell".into(),
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
        },
        SurfaceChannelInitialState {
            time_s: 0.0,
            surface: surface_state,
            reaches: vec![
                (ALPHA.into(), reach_state.clone()),
                (BETA.into(), reach_state),
            ],
        },
    ))
}

/// `setup` with `ports` as its ports.
pub fn ported(
    mut setup: SurfaceChannelSetup,
    ports: Vec<SurfaceChannelPort>,
) -> SurfaceChannelSetup {
    setup.ports = ports;
    setup
}

/// Input message of a rejected port definition.
pub fn rejection(setup: SurfaceChannelSetup, ports: Vec<SurfaceChannelPort>) -> TestResult<String> {
    match SurfaceChannelDefinition::new(ported(setup, ports)) {
        Ok(_) => Err("the definition admitted an invalid port".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}

/// Fails unless `message` holds every part of `parts`.
pub fn contains(message: &str, parts: &[&str]) -> TestResult {
    match parts.iter().find(|part| !message.contains(**part)) {
        Some(part) => Err(format!("{message:?} lacks {part:?}").into()),
        None => Ok(()),
    }
}
