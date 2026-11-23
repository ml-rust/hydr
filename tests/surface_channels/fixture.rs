use hydr::channel::*;
use hydr::coupling::*;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::LevelPoolConfig;
use hydr::structures::WeirApplicability;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Batch = SurfaceChannelBatch<crate::runtime::Runtime>;

pub const DATUM: &str = "Synthetic surface/channel datum";
/// Surface bed elevation of every active cell, in metres.
pub const SURFACE_BED: f64 = 1.0;
/// Bed elevation of every linked-reach cell, in metres.
pub const CHANNEL_BED: f64 = 0.0;
/// Inactive surface cell of the 4 x 2 grid.
pub const INACTIVE_CELL: usize = 7;
/// Linked reach identity.
pub const REACH: &str = "reach";
/// Unlinked reach identity.
pub const SPARE: &str = "spare";
/// Caller lateral binding on the linked reach.
pub const CALLER_BINDING: &str = "caller-inflow";

fn surface() -> Result<(SolverSetup, InitialState)> {
    let grid = CartesianGrid::new(4, 2, 2.0, 2.0)?;
    let count = grid.cell_count();
    let active: Vec<bool> = (0..count).map(|cell| cell != INACTIVE_CELL).collect();
    let domain = Domain::new(
        grid,
        vec![SURFACE_BED; count],
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

/// Rectangular reach of `count` 2 m cells, 1 m wide, on `bed`.
pub fn reach_definition(count: usize, bed: f64) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit rectangular reach beside the surface grid".into(),
        geometry: ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: 2.0,
                    width_m: 1.0,
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

fn reach_state(count: usize) -> ChannelInitialState {
    ChannelInitialState {
        time_s: 0.0,
        volume_m3: vec![1.0; count],
        longitudinal_momentum_m4_per_s: vec![0.0; count],
    }
}

pub fn applicability() -> WeirApplicability {
    WeirApplicability {
        coefficient_provenance: "Declared broad bank coefficient".into(),
        free_nappe_and_approach_assertion: "Suppressed bank crest with quiescent approach".into(),
        submergence_correction_assertion: "Villemonte correction adopted for bank overflow".into(),
        symmetric_reversal: true,
    }
}

pub fn link(id: &str, surface_cell: usize, channel_cell: usize) -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: id.into(),
        surface_cell,
        reach_id: REACH.into(),
        channel_cell,
        crest_elevation_m: 1.2,
        crest_length_m: 2.0,
        weir_coefficient: 1.7,
        applicability: applicability(),
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "Surface cell row 0 borders the reach cell of the same column".into(),
    }
}

pub fn config() -> SurfaceChannelConfig {
    SurfaceChannelConfig {
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
    }
}

/// A 4 x 2 surface grid, a linked 4-cell reach, and an unlinked 2-cell reach.
///
/// Links are declared as `link-b`, `link-a`, `link-c` to exercise identity ordering.
pub fn case() -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
    let (surface, surface_state) = surface()?;
    let mut linked = reach_definition(4, CHANNEL_BED);
    linked.lateral_bindings.push(ChannelLateralBinding {
        boundary_id: CALLER_BINDING.into(),
        cell_id: "cell-3".into(),
        direction: ChannelLateralDirection::Inflow,
        ownership_assertion: "Tributary inflow outside the surface grid".into(),
    });
    Ok((
        SurfaceChannelSetup {
            surface,
            reaches: vec![
                SurfaceChannelReach {
                    id: REACH.into(),
                    definition: linked,
                },
                SurfaceChannelReach {
                    id: SPARE.into(),
                    definition: reach_definition(2, CHANNEL_BED),
                },
            ],
            links: vec![
                link("link-b", 1, 1),
                link("link-a", 0, 0),
                link("link-c", 2, 2),
            ],
            ports: Vec::new(),
            pools: Vec::new(),
            pool_support: Vec::new(),
            pool_sources: None,
            pool_inflows: Vec::new(),
            storage: LevelPoolConfig::default(),
            structures: Vec::new(),
            pool_ends: Vec::new(),
            operating: None,
            geometry_schedules: Vec::new(),
            runoff: None,
            storage_exclusion_evidence: "Reach storage lies outside every surface cell".into(),
            config: config(),
        },
        SurfaceChannelInitialState {
            time_s: 0.0,
            surface: surface_state,
            reaches: vec![
                (SPARE.into(), reach_state(2)),
                (REACH.into(), reach_state(4)),
            ],
            pools: Vec::new(),
            runoff: Vec::new(),
        },
    ))
}

/// Returns the input message of a rejected definition.
pub fn input_rejection(setup: SurfaceChannelSetup) -> Result<String> {
    match SurfaceChannelDefinition::new(setup) {
        Ok(_) => Err("the definition accepted an invalid setup".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}

/// Returns the input message of a rejected batch construction.
pub fn batch_input_rejection(
    setup: SurfaceChannelSetup,
    initial: SurfaceChannelInitialState,
) -> Result<String> {
    match Batch::new(crate::runtime::client(), setup, initial) {
        Ok(_) => Err("the batch accepted an invalid setup or initial state".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}

/// Fails unless `message` contains every fragment.
pub fn contains(message: &str, fragments: &[&str]) -> Result {
    for fragment in fragments {
        if !message.contains(fragment) {
            return Err(format!("message {message:?} lacks {fragment:?}").into());
        }
    }
    Ok(())
}
