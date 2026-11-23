//! Reach-end port cases on a 4 x 2 grid of 2 m cells beside two 4 m wide rectangular reaches.
//!
//! Surface cell `row * 4 + column` is 2 m square. Each reach has three cells, by default 2 m
//! long and of the 4 m port width. West edge faces `x_face(0, *)` and east edge faces
//! `x_face(4, *)` each match one reach end. Every case uses the public coordinator API only.

use super::fixture::{Result, applicability, config};
use hydr::channel::*;
use hydr::coupling::*;
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, ForcingInterval, ForcingSchedule,
    InitialState, SolverConfig, SolverSetup,
};
use hydr::storage::LevelPoolConfig;

pub type Batch = SurfaceChannelBatch<crate::runtime::Runtime>;

pub const DATUM: &str = "Synthetic port datum";
pub const ALPHA: &str = "alpha";
pub const BETA: &str = "beta";
/// Width of every reach cell, equal to two 2 m faces, in m.
pub const REACH_WIDTH: f64 = 4.0;
/// Reach cell length, in m.
pub const LENGTH: f64 = 2.0;
/// Surface cells of the grid.
pub const CELLS: usize = 8;
/// Gravity of every child, in m/s².
pub const GRAVITY: f64 = 9.81;

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

/// The `alpha` left end on the west edge.
pub fn west(id: &str) -> SurfaceChannelPort {
    port(
        id,
        ALPHA,
        ChannelEnd::Left,
        vec![x_face(0, 0), x_face(0, 1)],
    )
}

/// The `alpha` right end on the east edge.
pub fn east(id: &str) -> SurfaceChannelPort {
    port(
        id,
        ALPHA,
        ChannelEnd::Right,
        vec![x_face(4, 0), x_face(4, 1)],
    )
}

/// One port on each reach end, covering both ghost sides of both axes.
///
/// Declared out of identity order and with faces out of line order.
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

/// Rectangular reach of `count` cells of the port width on `bed`.
pub fn rectangular(count: usize, bed: f64) -> ChannelReachDefinition {
    sized(count, bed, [REACH_WIDTH, LENGTH])
}

/// The reach of [`sized`] as `count` surveyed cells on one rectangular section of `width`.
///
/// The section runs from the bed to 4 m above it, as the rectangular cells do.
pub fn surveyed_sized(count: usize, bed: f64, [width, length]: [f64; 2]) -> ChannelReachDefinition {
    ChannelReachDefinition {
        provenance: "Surveyed rectangular reach at the surface grid edge".into(),
        geometry: ChannelGeometry::PrismaticSurveyed {
            profile: CrossSectionDefinition {
                id: "port-section".into(),
                vertical_datum: DATUM.into(),
                points: [
                    (0.0, bed + 4.0),
                    (0.0, bed),
                    (width, bed),
                    (width, bed + 4.0),
                ]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
                maximum_stage_m: bed + 4.0,
                profile_provenance: "Analytical rectangular survey of the port width".into(),
            },
            cells: (0..count)
                .map(|i| SurveyedChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    manning_n: 0.0,
                })
                .collect(),
            numerical: ChannelSurveyedConfig::default(),
        },
        ..sized(count, bed, [width, length])
    }
}

/// Rectangular reach of `count` cells of `[width, length]` on `bed`, in m.
pub fn sized(count: usize, bed: f64, [width, length]: [f64; 2]) -> ChannelReachDefinition {
    ChannelReachDefinition {
        vertical_datum: DATUM.into(),
        provenance: "Rectangular reach at the surface grid edge".into(),
        geometry: ChannelGeometry::Rectangular {
            cells: (0..count)
                .map(|i| RectangularChannelCell {
                    id: format!("cell-{i}"),
                    length_m: length,
                    width_m: width,
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
pub fn bank() -> SurfaceChannelLink {
    SurfaceChannelLink {
        id: "bank".into(),
        surface_cell: 1,
        reach_id: ALPHA.into(),
        channel_cell: 1,
        crest_elevation_m: 1.2,
        crest_length_m: 2.0,
        weir_coefficient: 1.7,
        applicability: applicability(),
        reach_tangent: [1.0, 0.0],
        mapping_evidence: "Surface cell 1 borders reach cell 1".into(),
    }
}

/// Initial state of one reach: its bed, and depth and discharge per width in each cell.
#[derive(Clone, Debug)]
pub struct Reach {
    pub bed: f64,
    pub depth: [f64; 3],
    pub discharge_per_width: [f64; 3],
}

impl Reach {
    /// A reach at rest at `stage`, dry where the stage lies below its bed.
    pub fn rest(bed: f64, stage: f64) -> Self {
        Self::uniform(bed, (stage - bed).max(0.0), 0.0)
    }

    /// A reach with depth `depth` and discharge per width `q` in every cell.
    pub fn uniform(bed: f64, depth: f64, q: f64) -> Self {
        Self {
            bed,
            depth: [depth; 3],
            discharge_per_width: [q; 3],
        }
    }
}

/// A still surface cell of depth `h`.
pub fn still(h: f64) -> CellState {
    CellState {
        h,
        qx: 0.0,
        qy: 0.0,
    }
}

/// One ported coordinator case.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub surface_bed: f64,
    /// Surface cell states in row-major order.
    pub surface: Vec<CellState>,
    /// Inactive surface cells.
    pub inactive: Vec<usize>,
    pub alpha: Reach,
    pub beta: Reach,
    /// `alpha` cell width and length, in m.
    pub alpha_size: [f64; 2],
    /// Declare `alpha` as a surveyed rectangle of its cell width in place of rectangular cells.
    pub surveyed: bool,
    /// Keep the bank link on `alpha` cell 1.
    pub link: bool,
    pub ports: Vec<SurfaceChannelPort>,
    /// Held forcing of every open surface face. The other edge faces are walls.
    pub open: Vec<BoundaryForcing>,
    /// Caller policies at unported reach ends. Every other unported end is a wall.
    pub ends: Vec<(&'static str, ChannelEnd, ChannelBoundary)>,
}

impl Scenario {
    /// Every surface cell and both reaches at rest at `stage`, with the bank link and `ports`.
    pub fn rest(
        surface_bed: f64,
        stage: f64,
        alpha_bed: f64,
        ports: Vec<SurfaceChannelPort>,
    ) -> Self {
        Self {
            surface_bed,
            surface: vec![still((stage - surface_bed).max(0.0)); CELLS],
            inactive: Vec::new(),
            alpha: Reach::rest(alpha_bed, stage),
            beta: Reach::rest(0.25, stage),
            alpha_size: [REACH_WIDTH, LENGTH],
            surveyed: false,
            link: true,
            ports,
            open: Vec::new(),
            ends: Vec::new(),
        }
    }

    /// This case with `alpha` surveyed when `surveyed` is true.
    pub fn with_surveyed(mut self, surveyed: bool) -> Self {
        self.surveyed = surveyed;
        self
    }

    /// Forcing over `[start, end]`: no policy at each ported end, the declared policy or a
    /// wall at every other end.
    pub fn forcing(
        &self,
        setup: &SurfaceChannelSetup,
        start: f64,
        end: f64,
    ) -> SurfaceChannelForcingInterval {
        let mut interval = forcing(setup, start, end);
        for reach in &mut interval.reaches {
            for (id, side, policy) in &self.ends {
                if reach.reach_id != *id {
                    continue;
                }
                let slot = match side {
                    ChannelEnd::Left => &mut reach.left,
                    ChannelEnd::Right => &mut reach.right,
                };
                *slot = Some(policy.clone());
            }
        }
        interval
    }

    /// One accepted interval of this case toward `start + dt`.
    pub fn step(&self, batch: &Batch, dt: f64) -> Result<Batch> {
        let start = batch.time();
        let end = start + dt;
        Ok(batch
            .trial_to(end, &self.forcing(batch.setup(), start, end))?
            .into_next())
    }

    /// Setup and initial state of this case.
    pub fn parts(&self) -> Result<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(4, 2, 2.0, 2.0)?;
        let active: Vec<bool> = (0..CELLS)
            .map(|cell| !self.inactive.contains(&cell))
            .collect();
        let domain = Domain::new(
            grid,
            vec![self.surface_bed; CELLS],
            vec![0.0; CELLS],
            active.clone(),
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::with_open_faces(
            &domain,
            self.open.iter().map(BoundaryForcing::face).collect(),
        )?;
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![0.0; CELLS],
                potential_evaporation: vec![0.0; CELLS],
                withdrawal: vec![0.0; CELLS],
                boundaries: self.open.clone(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let cells = self
            .surface
            .iter()
            .zip(&active)
            .map(|(cell, &active)| if active { *cell } else { still(0.0) })
            .collect();
        let flow = FlowState::new(&domain, cells)?;
        let reach_state = |reach: &Reach, [width, length]: [f64; 2]| ChannelInitialState {
            time_s: 0.0,
            volume_m3: reach.depth.iter().map(|h| h * width * length).collect(),
            longitudinal_momentum_m4_per_s: reach
                .discharge_per_width
                .iter()
                .map(|q| q * width * length)
                .collect(),
        };
        let setup = SurfaceChannelSetup {
            surface: SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            reaches: vec![
                SurfaceChannelReach {
                    id: ALPHA.into(),
                    definition: if self.surveyed {
                        surveyed_sized(3, self.alpha.bed, self.alpha_size)
                    } else {
                        sized(3, self.alpha.bed, self.alpha_size)
                    },
                },
                SurfaceChannelReach {
                    id: BETA.into(),
                    definition: rectangular(3, self.beta.bed),
                },
            ],
            links: if self.link { vec![bank()] } else { Vec::new() },
            ports: self.ports.clone(),
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
        };
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![
                (ALPHA.into(), reach_state(&self.alpha, self.alpha_size)),
                (BETA.into(), reach_state(&self.beta, [REACH_WIDTH, LENGTH])),
            ],
            pools: Vec::new(),
            runoff: Vec::new(),
        };
        Ok((setup, initial))
    }

    /// The batch of this case.
    pub fn build(&self) -> Result<Batch> {
        let (setup, initial) = self.parts()?;
        Ok(Batch::new(crate::runtime::client(), setup, initial)?)
    }
}

/// Forcing over `[start, end]`: no policy at each ported end, a wall at every other end.
pub fn forcing(setup: &SurfaceChannelSetup, start: f64, end: f64) -> SurfaceChannelForcingInterval {
    let policy = |id: &str, end: ChannelEnd| {
        let ported = setup.ports.iter().any(|p| p.reach_id == id && p.end == end);
        (!ported).then_some(ChannelBoundary::Wall)
    };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: setup
            .reaches
            .iter()
            .map(|reach| SurfaceChannelReachForcing {
                reach_id: reach.id.clone(),
                left: policy(&reach.id, ChannelEnd::Left),
                right: policy(&reach.id, ChannelEnd::Right),
                lateral_inputs: Vec::new(),
            })
            .collect(),
        pool_sources: None,
        pool_inflows: Vec::new(),
        runoff: Vec::new(),
    }
}

/// One accepted interval toward `start + dt`.
pub fn step(batch: &Batch, dt: f64) -> Result<Batch> {
    let start = batch.time();
    let end = start + dt;
    Ok(batch
        .trial_to(end, &forcing(batch.setup(), start, end))?
        .into_next())
}

/// Diagnostics of the last accepted interval.
pub fn last(batch: &Batch) -> Result<SurfaceChannelStepDiagnostics> {
    batch
        .last_step()?
        .ok_or_else(|| "the batch holds no accepted interval".into())
}

/// The first port receipt of the last accepted interval.
pub fn transfer(batch: &Batch) -> Result<SurfaceChannelPortTransfer> {
    Ok(last(batch)?
        .ports
        .first()
        .ok_or("the port receipt")?
        .clone())
}

/// Cell results of reach `id`.
pub fn reach_cells(batch: &Batch, id: &str) -> Result<Vec<ChannelCellResult>> {
    Ok(batch.reach(id).ok_or("the reach exists")?.state()?)
}

/// Fails unless the lifetime composite balance holds.
pub fn balanced(batch: &Batch) -> Result {
    let balance = batch.ledger()?;
    if balance.residual_m3.abs() <= balance.bound_m3 {
        Ok(())
    } else {
        Err(format!(
            "composite residual={}, bound={}",
            balance.residual_m3, balance.bound_m3
        )
        .into())
    }
}

/// Input message of a rejected definition of `setup` with `ports`.
pub fn rejection(mut setup: SurfaceChannelSetup, ports: Vec<SurfaceChannelPort>) -> Result<String> {
    setup.ports = ports;
    match SurfaceChannelDefinition::new(setup) {
        Ok(_) => Err("the definition admitted an invalid port".into()),
        Err(SurfaceChannelError::Input(message)) => Ok(message),
        Err(other) => Err(format!("expected an input rejection, got {other}").into()),
    }
}
