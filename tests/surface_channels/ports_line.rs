//! One-row lines of a flat rectangular reach and a flat surface strip joined by one reach-end
//! port, and their pure-2D twins.
//!
//! The line runs along `x` from `0` to `2 n dx`. One side holds `n` reach cells of length
//! `dx`, the other `n` surface cells of `dx` by [`WIDTH`]. The reach width equals [`WIDTH`],
//! so the port joins one section. Every other face and reach end is a wall. Composite cell
//! `i` has its centre at `(i + 1/2) dx`. The twin is one `2n` x 1 surface grid of the same
//! cells.

use crate::fixture::Result;
use crate::impervious::impervious_solver;
use crate::ports_fixture::{
    ALPHA, Batch, DATUM, balanced, forcing, last, port, reach_cells, sized, surveyed_sized,
    transfer, x_face,
};
use hydr::channel::{ChannelEnd, ChannelInitialState};
use hydr::coupling::{
    SurfaceChannelInitialState, SurfaceChannelPortTransfer, SurfaceChannelReach,
    SurfaceChannelSetup,
};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    AdvanceStatus, BoundaryConditions, ForcingInterval, ForcingSchedule, InitialState, Solver,
    SolverConfig, SolverSetup,
};
use hydr::storage::LevelPoolConfig;

pub type Twin = Solver<crate::runtime::Runtime>;

/// Strip width, reach width, and port width, in m.
pub const WIDTH: f64 = 1.0;

/// End of the surface forcing schedule, in s.
const SCHEDULE_END_S: f64 = 1.0e3;

/// Which child holds the western half of the line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// The reach runs east into the strip's west edge through its right end.
    ReachFirst,
    /// The strip's east edge runs into the reach's left end.
    SurfaceFirst,
}

/// One line case: the side order, the spacing, and the initial composite depths.
#[derive(Clone, Debug)]
pub struct Line {
    pub order: Order,
    pub dx: f64,
    /// Declare the reach as a surveyed rectangle of the strip width.
    pub surveyed: bool,
    /// Initial still-water depth of every composite cell, west to east, in m.
    pub depths: Vec<f64>,
}

impl Line {
    /// `n` cells per side of spacing `dx`, with depth `depth(x)` at each cell centre.
    pub fn new(order: Order, n: usize, dx: f64, depth: impl Fn(f64) -> f64) -> Self {
        Self {
            order,
            dx,
            surveyed: false,
            depths: (0..2 * n).map(|i| depth(centre(i, dx))).collect(),
        }
    }

    /// This line with a surveyed reach when `surveyed` is true.
    pub fn with_surveyed(mut self, surveyed: bool) -> Self {
        self.surveyed = surveyed;
        self
    }

    /// Cells per side.
    pub fn n(&self) -> usize {
        self.depths.len() / 2
    }

    /// Composite cell centres, west to east, in m.
    pub fn centres(&self) -> Vec<f64> {
        (0..self.depths.len()).map(|i| centre(i, self.dx)).collect()
    }

    /// Initial reach depths and surface depths.
    fn halves(&self) -> (&[f64], &[f64]) {
        let (west, east) = self.depths.split_at(self.n());
        match self.order {
            Order::ReachFirst => (west, east),
            Order::SurfaceFirst => (east, west),
        }
    }

    /// The coordinated batch.
    pub fn batch(&self) -> Result<Batch> {
        let n = self.n();
        let (reach, surface) = self.halves();
        let domain = Domain::new(
            CartesianGrid::new(n, 1, self.dx, WIDTH)?,
            vec![0.0; n],
            vec![0.0; n],
            vec![true; n],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let schedule = schedule(&domain, &boundaries)?;
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(&domain, surface.iter().map(|&h| still(h)).collect())?;
        let face = match self.order {
            Order::ReachFirst => port("p", ALPHA, ChannelEnd::Right, vec![x_face(0, 0)]),
            Order::SurfaceFirst => port("p", ALPHA, ChannelEnd::Left, vec![x_face(n, 0)]),
        };
        let cell_volume = WIDTH * self.dx;
        let setup = SurfaceChannelSetup {
            surface: SolverSetup {
                domain,
                land,
                boundaries,
                forcing: schedule,
                config: SolverConfig::default(),
            },
            reaches: vec![SurfaceChannelReach {
                id: ALPHA.into(),
                definition: if self.surveyed {
                    surveyed_sized(n, 0.0, [WIDTH, self.dx])
                } else {
                    sized(n, 0.0, [WIDTH, self.dx])
                },
            }],
            links: Vec::new(),
            ports: vec![face],
            pools: Vec::new(),
            pool_support: Vec::new(),
            pool_sources: None,
            pool_inflows: Vec::new(),
            storage: LevelPoolConfig::default(),
            structures: Vec::new(),
            pool_ends: Vec::new(),
            operating: None,
            geometry_schedules: Vec::new(),
            pool_curve_schedules: Vec::new(),
            surface_bed_schedules: Vec::new(),
            runoff: None,
            storage_exclusion_evidence: "Reach storage lies outside every surface cell".into(),
            config: crate::fixture::config(),
        };
        let initial = SurfaceChannelInitialState {
            time_s: 0.0,
            surface: InitialState {
                time: 0.0,
                flow,
                soil,
            },
            reaches: vec![(
                ALPHA.into(),
                ChannelInitialState {
                    time_s: 0.0,
                    volume_m3: reach.iter().map(|h| h * cell_volume).collect(),
                    longitudinal_momentum_m4_per_s: vec![0.0; n],
                },
            )],
            pools: Vec::new(),
            runoff: Vec::new(),
        };
        Ok(Batch::new(crate::runtime::client(), setup, initial)?)
    }

    /// The pure-2D twin.
    pub fn twin(&self) -> Result<Twin> {
        let count = self.depths.len();
        let domain = Domain::new(
            CartesianGrid::new(count, 1, self.dx, WIDTH)?,
            vec![0.0; count],
            vec![0.0; count],
            vec![true; count],
            SpatialReference::new("Synthetic projected CRS".into(), DATUM.into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let schedule = schedule(&domain, &boundaries)?;
        let flow = FlowState::new(&domain, self.depths.iter().map(|&h| still(h)).collect())?;
        impervious_solver(
            domain,
            flow,
            boundaries,
            schedule,
            SolverConfig::default(),
            0.0,
        )
    }

    /// Composite depths of `batch`, west to east.
    pub fn depths_of(&self, batch: &Batch) -> Result<Vec<f64>> {
        let reach: Vec<f64> = reach_cells(batch, ALPHA)?
            .iter()
            .map(|cell| cell.depth_m)
            .collect();
        let surface: Vec<f64> = batch
            .surface()
            .state()?
            .cells()
            .iter()
            .map(|c| c.h)
            .collect();
        Ok(match self.order {
            Order::ReachFirst => [reach, surface].concat(),
            Order::SurfaceFirst => [surface, reach].concat(),
        })
    }
}

/// Centre of composite cell `i`, in m.
fn centre(i: usize, dx: f64) -> f64 {
    (i as f64 + 0.5) * dx
}

fn still(h: f64) -> CellState {
    CellState {
        h,
        qx: 0.0,
        qy: 0.0,
    }
}

fn schedule(domain: &Domain, boundaries: &BoundaryConditions) -> Result<ForcingSchedule> {
    let count = domain.grid().cell_count();
    Ok(ForcingSchedule::new(
        domain,
        boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: SCHEDULE_END_S,
            rain: vec![0.0; count],
            potential_evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: Vec::new(),
        }],
    )?)
}

/// One accepted interval of a line case: its start, its end, and its port receipt.
#[derive(Clone, Debug)]
pub struct Interval {
    pub start_s: f64,
    pub end_s: f64,
    pub port: SurfaceChannelPortTransfer,
}

/// Advances `batch` to exactly `end_s`. Each interval requests the rest of the run, and the
/// step limits shorten it.
///
/// Every interval checks the port pairing, the step mass bound, and the lifetime composite
/// balance. Returns the last batch and every interval.
pub fn advance(mut batch: Batch, end_s: f64) -> Result<(Batch, Vec<Interval>)> {
    let mut run = Vec::new();
    while batch.time() < end_s {
        let start_s = batch.time();
        batch = batch
            .trial_to(end_s, &forcing(batch.setup(), start_s, end_s))?
            .into_next();
        let diagnostics = last(&batch)?;
        let port = transfer(&batch)?;
        assert!(
            port.paired_gap_m3.abs() <= port.paired_bound_m3,
            "t={start_s}: {port:?}"
        );
        assert!(
            diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3,
            "t={start_s}: {diagnostics:?}"
        );
        balanced(&batch)?;
        run.push(Interval {
            start_s,
            end_s: batch.time(),
            port,
        });
    }
    Ok((batch, run))
}

/// Advances `twin` to exactly `end_s` and returns its depths, west to east.
pub fn twin_depths(mut twin: Twin, end_s: f64) -> Result<Vec<f64>> {
    assert_eq!(
        twin.advance_to(end_s, 1_000_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    Ok(twin.state()?.cells().iter().map(|cell| cell.h).collect())
}
