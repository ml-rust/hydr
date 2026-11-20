//! Configurable surface grids, reaches, and links for common-interval tests.

use super::fixture::{DATUM, Result, applicability, config};
use hydr::channel::*;
use hydr::coupling::*;
use hydr::forcing::WaterAmount;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::LevelPoolConfig;

pub type Batch = SurfaceChannelBatch<crate::runtime::Runtime>;

/// Default surface cell edge length, in metres. A default cell holds 4 m² of area.
pub const DX: f64 = 2.0;
/// Channel cell length, in metres. Every channel cell is 1 m wide.
pub const CELL_LENGTH: f64 = 2.0;

/// One reach of 2 m cells, 1 m wide, on a flat bed.
#[derive(Clone)]
pub struct ReachSpec {
    pub id: String,
    pub bed: f64,
    /// Initial water depth of every cell, in metres.
    pub depths: Vec<f64>,
    /// Caller lateral bindings: identity, cell index, direction.
    pub bindings: Vec<(String, usize, ChannelLateralDirection)>,
    /// A surveyed 1 m rectangular section replaces the rectangular cells.
    pub surveyed: bool,
    /// Root probe of a surveyed section. Default: the surveyed config default.
    pub probe: solvr::optimize::CertifiedRootProbe,
}

impl ReachSpec {
    pub fn new(id: &str, bed: f64, depths: &[f64]) -> Self {
        Self {
            id: id.into(),
            bed,
            depths: depths.to_vec(),
            bindings: Vec::new(),
            surveyed: false,
            probe: ChannelSurveyedConfig::default().normalized_root.probe,
        }
    }

    /// The same reach as one surveyed 1 m rectangular section. Stored volumes stay equal.
    pub fn surveyed(mut self) -> Self {
        self.surveyed = true;
        self
    }

    /// The same reach with `probe` in its surveyed root controls.
    pub fn probe(mut self, probe: solvr::optimize::CertifiedRootProbe) -> Self {
        self.probe = probe;
        self
    }

    pub fn binding(mut self, id: &str, cell: usize, direction: ChannelLateralDirection) -> Self {
        self.bindings.push((id.into(), cell, direction));
        self
    }
}

/// A flat surface grid with walls, zero surface forcing, reaches, and links.
#[derive(Clone)]
pub struct Spec {
    pub nx: usize,
    pub ny: usize,
    /// Edge length of every square surface cell, in metres.
    pub dx: f64,
    pub bed: f64,
    /// Initial depth of every surface cell in row-major order, in metres.
    pub depth: Vec<f64>,
    /// Prescribed land withdrawal rate of every surface cell, in metres per second.
    pub withdrawal: Vec<f64>,
    /// Surface forcing interval boundaries, ascending from zero.
    pub breaks: Vec<f64>,
    pub reaches: Vec<ReachSpec>,
    pub links: Vec<SurfaceChannelLink>,
    pub config: SurfaceChannelConfig,
}

impl Spec {
    /// A `nx` x `ny` grid of `DX` cells on `bed` with uniform `depth`.
    pub fn new(nx: usize, ny: usize, bed: f64, depth: f64) -> Self {
        Self {
            nx,
            ny,
            dx: DX,
            bed,
            depth: vec![depth; nx * ny],
            withdrawal: vec![0.0; nx * ny],
            breaks: vec![0.0, 100.0],
            reaches: Vec::new(),
            links: Vec::new(),
            config: config(),
        }
    }

    pub fn reach(mut self, reach: ReachSpec) -> Self {
        self.reaches.push(reach);
        self
    }

    pub fn link(mut self, link: SurfaceChannelLink) -> Self {
        self.links.push(link);
        self
    }

    pub fn parts(&self) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(self.nx, self.ny, self.dx, self.dx)?;
        let count = grid.cell_count();
        let domain = Domain::new(
            grid,
            vec![self.bed; count],
            vec![0.0; count],
            vec![true; count],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let intervals = self
            .breaks
            .windows(2)
            .map(|w| ForcingInterval {
                start: w[0],
                end: w[1],
                rain: vec![0.0; count],
                potential_evaporation: vec![0.0; count],
                withdrawal: self.withdrawal.clone(),
                boundaries: Vec::new(),
            })
            .collect();
        let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let cells = self
            .depth
            .iter()
            .map(|&h| CellState {
                h,
                qx: 0.0,
                qy: 0.0,
            })
            .collect();
        let flow = FlowState::new(&domain, cells)?;
        let reaches = self
            .reaches
            .iter()
            .map(|reach| SurfaceChannelReach {
                id: reach.id.clone(),
                definition: reach_definition(reach),
            })
            .collect();
        let states = self
            .reaches
            .iter()
            .map(|reach| {
                (
                    reach.id.clone(),
                    ChannelInitialState {
                        time_s: 0.0,
                        volume_m3: reach.depths.iter().map(|d| d * CELL_LENGTH).collect(),
                        longitudinal_momentum_m4_per_s: vec![0.0; reach.depths.len()],
                    },
                )
            })
            .collect();
        Ok((
            SurfaceChannelSetup {
                surface: SolverSetup {
                    domain,
                    land,
                    boundaries,
                    forcing,
                    config: SolverConfig::default(),
                },
                reaches,
                links: self.links.clone(),
                ports: Vec::new(),
                pools: Vec::new(),
                pool_support: Vec::new(),
                pool_sources: None,
                pool_inflows: Vec::new(),
                storage: LevelPoolConfig::default(),
                structures: Vec::new(),
                pool_ends: Vec::new(),
                operating: None,
                storage_exclusion_evidence: "Reach storage lies outside every surface cell".into(),
                config: self.config.clone(),
            },
            SurfaceChannelInitialState {
                time_s: 0.0,
                surface: InitialState {
                    time: 0.0,
                    flow,
                    soil,
                },
                reaches: states,
                pools: Vec::new(),
            },
        ))
    }

    pub fn build(&self) -> Result<Batch> {
        let (setup, initial) = self.parts()?;
        Ok(Batch::new(crate::runtime::client(), setup, initial)?)
    }
}

/// One surveyed 1 m rectangular section from `bed` to 4 m above it, over `count` cells.
pub fn surveyed_geometry(bed: f64, count: usize) -> ChannelGeometry {
    ChannelGeometry::PrismaticSurveyed {
        profile: CrossSectionDefinition {
            id: "surveyed-bank-section".into(),
            vertical_datum: DATUM.into(),
            points: [(0.0, bed + 4.0), (0.0, bed), (1.0, bed), (1.0, bed + 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: bed + 4.0,
            profile_provenance: "Explicit analytical rectangular survey".into(),
        },
        cells: (0..count)
            .map(|i| SurveyedChannelCell {
                id: format!("cell-{i}"),
                length_m: CELL_LENGTH,
                manning_n: 0.0,
            })
            .collect(),
        numerical: ChannelSurveyedConfig::default(),
    }
}

fn reach_definition(reach: &ReachSpec) -> ChannelReachDefinition {
    let count = reach.depths.len();
    let geometry = if reach.surveyed {
        let mut geometry = surveyed_geometry(reach.bed, count);
        if let ChannelGeometry::PrismaticSurveyed { numerical, .. } = &mut geometry {
            numerical.normalized_root.probe = reach.probe;
        }
        geometry
    } else {
        ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: CELL_LENGTH,
                    width_m: 1.0,
                    bed_elevation_m: reach.bed,
                    maximum_depth_m: 4.0,
                    manning_n: 0.0,
                })
                .collect(),
        }
    };
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Explicit reach beside the surface grid".into(),
        geometry,
        lateral_bindings: reach
            .bindings
            .iter()
            .map(|(id, cell, direction)| ChannelLateralBinding {
                boundary_id: id.clone(),
                cell_id: format!("cell-{cell}"),
                direction: *direction,
                ownership_assertion: "Caller water outside the surface grid".into(),
            })
            .collect(),
        config: ChannelConfig::default(),
    }
}

/// Bank weir link with explicit crest, length, and coefficient.
pub fn weir(
    id: &str,
    surface_cell: usize,
    reach: &str,
    channel_cell: usize,
    crest: f64,
    length: f64,
) -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: id.into(),
        surface_cell,
        reach_id: reach.into(),
        channel_cell,
        crest_elevation_m: crest,
        crest_length_m: length,
        weir_coefficient: 1.7,
        applicability: applicability(),
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "Surface cell borders the reach cell".into(),
    }
}

/// Walls and zero caller rates, overridden by `inputs` as `(reach, input)` pairs.
pub fn forcing(
    batch: &Batch,
    end: f64,
    inputs: &[(&str, ChannelLateralInput)],
) -> SurfaceChannelForcingInterval {
    let reaches = batch
        .setup()
        .reaches
        .iter()
        .map(|reach| SurfaceChannelReachForcing {
            reach_id: reach.id.clone(),
            left: Some(ChannelBoundary::Wall),
            right: Some(ChannelBoundary::Wall),
            lateral_inputs: reach
                .definition
                .lateral_bindings
                .iter()
                .map(|binding| {
                    let given = inputs.iter().find(|(id, input)| {
                        *id == reach.id && input_id(input) == binding.boundary_id
                    });
                    match given {
                        Some((_, input)) => input.clone(),
                        None => zero_input(&binding.boundary_id, binding.direction),
                    }
                })
                .collect(),
        })
        .collect();
    SurfaceChannelForcingInterval {
        start_s: batch.time(),
        end_s: end,
        reaches,
        pool_sources: None,
        pool_inflows: Vec::new(),
    }
}

fn input_id(input: &ChannelLateralInput) -> &str {
    match input {
        ChannelLateralInput::Inflow { boundary_id, .. }
        | ChannelLateralInput::Withdrawal { boundary_id, .. } => boundary_id,
    }
}

fn zero_input(id: &str, direction: ChannelLateralDirection) -> ChannelLateralInput {
    let amount = WaterAmount::Rate { rate_m3_per_s: 0.0 };
    match direction {
        ChannelLateralDirection::Inflow => ChannelLateralInput::Inflow {
            boundary_id: id.into(),
            amount,
            longitudinal_velocity_m_per_s: 0.0,
        },
        ChannelLateralDirection::Withdrawal => ChannelLateralInput::Withdrawal {
            boundary_id: id.into(),
            amount,
        },
    }
}

/// Accepts one common interval toward `end` with zero caller forcing.
pub fn step(batch: &Batch, end: f64) -> Result<Batch> {
    let forcing = forcing(batch, end, &[]);
    Ok(batch.trial_to(end, &forcing)?.into_next())
}

/// Accepts intervals until `end` with zero caller forcing.
pub fn advance(mut batch: Batch, end: f64) -> Result<Batch> {
    while batch.time() < end {
        batch = step(&batch, end)?;
    }
    Ok(batch)
}

/// Diagnostics of the last accepted interval.
pub fn last(batch: &Batch) -> Result<SurfaceChannelStepDiagnostics> {
    batch
        .last_step()?
        .ok_or_else(|| "the batch has no accepted interval".into())
}

/// Surface depth of every cell.
pub fn depths(batch: &Batch) -> Result<Vec<f64>> {
    Ok(batch
        .surface()
        .state()?
        .cells()
        .iter()
        .map(|c| c.h)
        .collect())
}

/// Stored volume of every cell of reach `id`.
pub fn volumes(batch: &Batch, id: &str) -> Result<Vec<f64>> {
    let reach = batch.reach(id).ok_or("unknown reach")?;
    Ok(reach.state()?.iter().map(|c| c.volume_m3).collect())
}
