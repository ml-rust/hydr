//! Ported batches on the 4 x 2 fixture grid, and steps through the coordinator-owned ports.
//!
//! Surface cell `row * 4 + column` is 2 m square. Both reaches have three 2 m long cells of
//! the 4 m port width. West edge faces `x_face(0, *)` and east edge faces `x_face(4, *)` each
//! match one reach end.

use super::super::{
    SurfaceChannelBatch, SurfaceChannelConfig, SurfaceChannelDefinition,
    SurfaceChannelForcingInterval, SurfaceChannelInitialState, SurfaceChannelReach,
    SurfaceChannelReachForcing, SurfaceChannelSetup,
};
use super::fixture::{
    ALPHA, BETA, DATUM, REACH_WIDTH, TestResult, link, port, rectangular, x_face,
};
use super::{PortInterval, SurfaceChannelPort};
use crate::channel::{ChannelBoundary, ChannelInitialState};
use crate::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
};
use hydr_channel::internal::ChannelCouplingEnd;
use hydr_core::grid::CartesianGrid;
use hydr_core::model::{CellState, Domain, FlowState, SpatialReference};
use hydr_soil::soil::{LandModel, SoilState};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

pub type Batch = SurfaceChannelBatch<CpuRuntime>;

/// Surface cells of the fixture grid.
pub const CELLS: usize = 8;
/// Reach cell length, in m.
pub const LENGTH: f64 = 2.0;
/// Fixture gravity, in m/s².
pub const GRAVITY: f64 = 9.81;

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

/// One ported coordinator case.
#[derive(Clone, Debug)]
pub struct Scenario {
    pub surface_bed: f64,
    pub surface: Vec<CellState>,
    pub alpha: Reach,
    pub beta: Reach,
    /// Keep the fixture bank link on `alpha` cell 1.
    pub link: bool,
    pub ports: Vec<SurfaceChannelPort>,
}

/// The `alpha` left end on the west edge.
pub fn west(id: &str) -> SurfaceChannelPort {
    port(
        id,
        ALPHA,
        ChannelCouplingEnd::Left,
        vec![x_face(0, 0), x_face(0, 1)],
    )
}

/// The `alpha` right end on the east edge.
pub fn east(id: &str) -> SurfaceChannelPort {
    port(
        id,
        ALPHA,
        ChannelCouplingEnd::Right,
        vec![x_face(4, 0), x_face(4, 1)],
    )
}

/// A still surface cell of depth `h`.
pub fn still(h: f64) -> CellState {
    CellState {
        h,
        qx: 0.0,
        qy: 0.0,
    }
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
            alpha: Reach::rest(alpha_bed, stage),
            beta: Reach::rest(0.25, stage),
            link: true,
            ports,
        }
    }

    /// Setup and initial state of this case.
    pub fn parts(&self) -> TestResult<(SurfaceChannelSetup, SurfaceChannelInitialState)> {
        let grid = CartesianGrid::new(4, 2, 2.0, 2.0)?;
        let domain = Domain::new(
            grid,
            vec![self.surface_bed; CELLS],
            vec![0.0; CELLS],
            vec![true; CELLS],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![0.0; CELLS],
                potential_evaporation: vec![0.0; CELLS],
                withdrawal: vec![0.0; CELLS],
                boundaries: Vec::new(),
            }],
        )?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(&domain, self.surface.clone())?;
        let reach_state = |reach: &Reach| ChannelInitialState {
            time_s: 0.0,
            volume_m3: reach
                .depth
                .iter()
                .map(|h| h * REACH_WIDTH * LENGTH)
                .collect(),
            longitudinal_momentum_m4_per_s: reach
                .discharge_per_width
                .iter()
                .map(|q| q * REACH_WIDTH * LENGTH)
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
                    definition: rectangular(3, self.alpha.bed),
                },
                SurfaceChannelReach {
                    id: BETA.into(),
                    definition: rectangular(3, self.beta.bed),
                },
            ],
            links: if self.link { vec![link()] } else { Vec::new() },
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
        };
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![
                (ALPHA.into(), reach_state(&self.alpha)),
                (BETA.into(), reach_state(&self.beta)),
            ],
        };
        Ok((setup, initial))
    }

    /// The batch of this case on a single-threaded CPU client.
    pub fn build(&self) -> TestResult<Batch> {
        let (setup, initial) = self.parts()?;
        let definition = SurfaceChannelDefinition::with_ports(setup, self.ports.clone())?;
        Ok(SurfaceChannelBatch::from_definition(
            client(),
            definition,
            initial,
        )?)
    }
}

pub fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

/// Wall forcing at every end of both reaches over `[start, end]`.
///
/// A ported trial does not read the policy at a ported end.
pub fn forcing(start: f64, end: f64) -> SurfaceChannelForcingInterval {
    let reach = |id: &str| SurfaceChannelReachForcing {
        reach_id: id.into(),
        left: ChannelBoundary::Wall,
        right: ChannelBoundary::Wall,
        lateral_inputs: Vec::new(),
    };
    SurfaceChannelForcingInterval {
        start_s: start,
        end_s: end,
        reaches: vec![reach(ALPHA), reach(BETA)],
    }
}

/// One accepted interval toward `start + dt`, with the coordinator's ports.
pub fn step(batch: &Batch, dt: f64) -> TestResult<Batch> {
    let start = batch.time();
    let end = start + dt;
    Ok(batch.trial_ported(end, &forcing(start, end))?.into_next())
}

/// Port receipts of the last accepted interval.
pub fn receipt(batch: &Batch) -> TestResult<&PortInterval> {
    Ok(&batch
        .last
        .as_ref()
        .ok_or("the batch holds no accepted interval")?
        .ports)
}
