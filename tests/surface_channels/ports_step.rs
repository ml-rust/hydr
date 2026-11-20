//! Steady flow across a bed step at a reach-end port, against a pure-2D twin.
//!
//! The coupled case: `alpha` runs east into the west edge of the 4 x 2 surface through a port
//! on its right end. Its left end holds an exterior reservoir. The surface's east faces hold a
//! stage. The twin: one 7 x 2 grid of the same 2 m cells. Its first three columns stand for
//! the reach on the reach bed. The other four are the surface on the surface bed. The twin
//! holds the same reservoir on its west faces and the same stage on its east faces.

use crate::fixture::Result;
use crate::ports_fixture::{ALPHA, Batch, LENGTH, Scenario, last, port, reach_cells, x_face};
use hydr::channel::{ChannelBoundary, ChannelEnd};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryConditions, BoundaryForcing, ExteriorBoundary, ExteriorState, ForcingInterval,
    ForcingSchedule, InitialState, Solver, SolverConfig, SolverSetup, StageBoundary,
};

type Twin = Solver<crate::runtime::Runtime>;

/// Frozen steady-state test: over one accepted interval, every cell's `|Δh| <= STEADY h` and
/// `|Δq| <= STEADY max|q|`, both components of `q`.
const STEADY: f64 = 1e-12;

/// Frozen relative bound between steady inflow and outflow.
const BALANCE: f64 = 1e-9;

/// Frozen bound on the upstream stage gap to the twin, as a fraction of local depth.
const STAGE: f64 = 0.01;

/// Most accepted intervals before each run must be steady.
const MAXIMUM_INTERVALS: usize = 6000;

/// Requested duration of every interval, in s. The CFL limit shortens it.
const DT: f64 = 1.0;

/// Reservoir depth above the reach bed, in m.
const RESERVOIR_DEPTH: f64 = 1.0;

/// Reservoir velocity toward the surface, in m/s.
const RESERVOIR_VELOCITY: f64 = 0.2;

/// Twin columns that stand for the reach.
const REACH_COLUMNS: usize = 3;

/// Twin columns.
const COLUMNS: usize = 7;

/// One bed step: reach bed, surface bed, and the held downstream stage, in m.
#[derive(Clone, Copy, Debug)]
struct Step {
    reach_bed: f64,
    surface_bed: f64,
    stage: f64,
}

/// Held downstream stage on east face `row` of a grid with `columns` columns.
fn outlet(step: Step, columns: usize, row: usize) -> BoundaryForcing {
    BoundaryForcing::Stage(StageBoundary {
        face: x_face(columns, row),
        elevation: step.stage,
        inflow_tangential_velocity: None,
    })
}

/// The coupled case, at rest at the downstream stage.
fn coupled(step: Step, surveyed: bool) -> Scenario {
    let mut case = Scenario::rest(
        step.surface_bed,
        step.stage,
        step.reach_bed,
        vec![port(
            "p",
            ALPHA,
            ChannelEnd::Right,
            vec![x_face(0, 0), x_face(0, 1)],
        )],
    )
    .with_surveyed(surveyed);
    case.link = false;
    case.open = (0..2).map(|row| outlet(step, 4, row)).collect();
    case.ends = vec![(
        ALPHA,
        ChannelEnd::Left,
        ChannelBoundary::Exterior {
            depth_m: RESERVOIR_DEPTH,
            velocity_m_per_s: RESERVOIR_VELOCITY,
        },
    )];
    case
}

/// The pure-2D twin, at rest at the downstream stage.
fn twin(step: Step) -> Result<Twin> {
    let count = COLUMNS * 2;
    let bed: Vec<f64> = (0..count)
        .map(|cell| {
            if cell % COLUMNS < REACH_COLUMNS {
                step.reach_bed
            } else {
                step.surface_bed
            }
        })
        .collect();
    let cells = bed
        .iter()
        .map(|z| CellState {
            h: step.stage - z,
            qx: 0.0,
            qy: 0.0,
        })
        .collect();
    let domain = Domain::new(
        CartesianGrid::new(COLUMNS, 2, LENGTH, 2.0)?,
        bed,
        vec![0.0; count],
        vec![true; count],
        SpatialReference::new("Synthetic projected CRS".into(), "twin".into(), [0.0; 2])?,
    )?;
    let mut forcing: Vec<BoundaryForcing> = (0..2)
        .map(|row| {
            BoundaryForcing::Exterior(ExteriorBoundary {
                face: x_face(0, row),
                state: ExteriorState::Wet {
                    elevation: step.reach_bed + RESERVOIR_DEPTH,
                    outward_velocity: -RESERVOIR_VELOCITY,
                    tangential_velocity: 0.0,
                },
            })
        })
        .collect();
    forcing.extend((0..2).map(|row| outlet(step, COLUMNS, row)));
    let boundaries = BoundaryConditions::with_open_faces(
        &domain,
        forcing.iter().map(BoundaryForcing::face).collect(),
    )?;
    let schedule = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 1.0e5,
            rain: vec![0.0; count],
            potential_evaporation: vec![0.0; count],
            withdrawal: vec![0.0; count],
            boundaries: forcing,
        }],
    )?;
    let land = LandModel::impervious(&domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(&domain, cells)?;
    Ok(Solver::new(
        crate::runtime::client(),
        SolverSetup {
            domain,
            land,
            boundaries,
            forcing: schedule,
            config: SolverConfig::default(),
        },
        InitialState {
            time: 0.0,
            flow,
            soil,
        },
    )?)
}

/// Depth and both discharge components of every cell.
type Fields = Vec<[f64; 3]>;

/// True when no field of `after` moved past [`STEADY`] from `before`.
fn steady(before: &Fields, after: &Fields) -> bool {
    let scale = after
        .iter()
        .flat_map(|cell| [cell[1].abs(), cell[2].abs()])
        .fold(0.0, f64::max);
    before.iter().zip(after).all(|(a, b)| {
        (a[0] - b[0]).abs() <= STEADY * b[0]
            && (a[1] - b[1]).abs() <= STEADY * scale
            && (a[2] - b[2]).abs() <= STEADY * scale
    })
}

/// Surface cells, then `alpha` cells as depth and discharge per width.
fn coupled_fields(batch: &Batch) -> Result<Fields> {
    let mut fields: Fields = batch
        .surface()
        .state()?
        .cells()
        .iter()
        .map(|cell| [cell.h, cell.qx, cell.qy])
        .collect();
    for cell in reach_cells(batch, ALPHA)? {
        fields.push([cell.depth_m, cell.discharge_m3_per_s, 0.0]);
    }
    Ok(fields)
}

fn twin_fields(twin: &Twin) -> Result<Fields> {
    Ok(twin
        .state()?
        .cells()
        .iter()
        .map(|cell| [cell.h, cell.qx, cell.qy])
        .collect())
}

/// Volume crossing the east faces of a grid with `columns` columns in the last interval.
fn east_volume(faces: &[hydr::solver::FaceTransfer], columns: usize) -> f64 {
    faces
        .iter()
        .filter(|transfer| transfer.face == x_face(columns, transfer.face.row))
        .map(|transfer| transfer.volume)
        .sum()
}

/// Steady inflow, outflow, and upstream stages of the coupled case.
fn coupled_steady(step: Step, surveyed: bool) -> Result<(f64, f64, Vec<f64>)> {
    let case = coupled(step, surveyed);
    let mut batch = case.build()?;
    let mut fields = coupled_fields(&batch)?;
    for _ in 0..MAXIMUM_INTERVALS {
        batch = case.step(&batch, DT)?;
        let next = coupled_fields(&batch)?;
        let done = steady(&fields, &next);
        fields = next;
        if !done {
            continue;
        }
        let reach = batch
            .reach(ALPHA)
            .ok_or("alpha")?
            .last_step()?
            .ok_or("a reach receipt")?;
        let inflow = reach.face_volume_m3[0];
        let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
        let outflow = east_volume(&surface.faces, 4);
        assert!(last(&batch)?.mass_residual_m3.abs() <= last(&batch)?.mass_bound_m3);
        let stages = reach_cells(&batch, ALPHA)?
            .iter()
            .map(|cell| cell.stage_m.ok_or("the reach is wet"))
            .collect::<std::result::Result<_, _>>()?;
        return Ok((inflow, outflow, stages));
    }
    Err(format!("{step:?}: the coupled case is not steady").into())
}

/// Steady inflow, outflow, reach-column stages, and reach-column depths of the twin.
fn twin_steady(step: Step) -> Result<(f64, f64, Vec<f64>, Vec<f64>)> {
    let mut twin = twin(step)?;
    let mut fields = twin_fields(&twin)?;
    for _ in 0..MAXIMUM_INTERVALS {
        let target = twin.time() + DT;
        twin.advance_to(target, 1)?;
        let next = twin_fields(&twin)?;
        let done = steady(&fields, &next);
        fields = next;
        if !done {
            continue;
        }
        let receipt = twin.last_step()?.ok_or("a twin receipt")?;
        let inflow: f64 = receipt
            .faces
            .iter()
            .filter(|transfer| transfer.face == x_face(0, transfer.face.row))
            .map(|transfer| transfer.volume)
            .sum();
        let outflow = east_volume(&receipt.faces, COLUMNS);
        // Both rows hold one state. Row 0 stands for the reach.
        let depths: Vec<f64> = fields[..REACH_COLUMNS].iter().map(|cell| cell[0]).collect();
        let stages = depths.iter().map(|h| h + step.reach_bed).collect();
        return Ok((inflow, outflow, stages, depths));
    }
    Err(format!("{step:?}: the twin is not steady").into())
}

fn relative(a: f64, b: f64) -> f64 {
    (a - b).abs() / a.abs().max(b.abs())
}

/// Runs both systems to steady flow and checks the frozen criteria.
fn agrees(step: Step, surveyed: bool) -> Result {
    let (inflow, outflow, stages) = coupled_steady(step, surveyed)?;
    assert!(inflow > 0.0, "{step:?}: water must run through the port");
    assert!(
        relative(inflow, outflow) <= BALANCE,
        "{step:?}: {inflow} against {outflow}"
    );
    let (twin_in, twin_out, twin_stages, depths) = twin_steady(step)?;
    assert!(
        relative(twin_in, twin_out) <= BALANCE,
        "{step:?}: {twin_in} against {twin_out}"
    );
    for ((stage, expected), depth) in stages.iter().zip(&twin_stages).zip(&depths) {
        assert!(
            (stage - expected).abs() <= STAGE * depth,
            "{step:?}: reach stage {stage} against twin {expected}, depth {depth}"
        );
    }
    Ok(())
}

/// Steady flow up a bed step, on the rectangular reach or on its surveyed rectangle.
pub(crate) fn steady_up(surveyed: bool) -> Result {
    agrees(
        Step {
            reach_bed: 0.0,
            surface_bed: 0.2,
            stage: 0.95,
        },
        surveyed,
    )
}

/// Steady flow down a bed step, on the rectangular reach or on its surveyed rectangle.
pub(crate) fn steady_down(surveyed: bool) -> Result {
    agrees(
        Step {
            reach_bed: 0.2,
            surface_bed: 0.0,
            stage: 1.15,
        },
        surveyed,
    )
}

#[test]
fn steady_flow_up_a_bed_step_matches_the_pure_2d_twin() -> Result {
    steady_up(false)
}

#[test]
fn steady_flow_down_a_bed_step_matches_the_pure_2d_twin() -> Result {
    steady_down(false)
}
