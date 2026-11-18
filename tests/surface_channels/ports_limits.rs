//! Port CFL coverage, a binding port speed, and depletion of a draining end cell.

use crate::fixture::Result;
use crate::ports_fixture::{
    ALPHA, Batch, CELLS, GRAVITY, LENGTH, Reach, Scenario, balanced, east, last, port, reach_cells,
    step, still, west, y_face,
};
use hydr::channel::{ChannelConfig, ChannelEnd};
use hydr::coupling::{SurfaceChannelPort, SurfaceChannelPortRegime};
use hydr::model::CellState;
use hydr::solver::{Axis, BoundaryFace, FaceTransfer, SolverConfig};
use hydr_core::numerics::{TensorFaceState, hydrostatic_hll_tensor};
use numr::tensor::Tensor;

/// `|u| + sqrt(g h)` of a wet state, zero when dry.
fn characteristic(depth: f64, velocity: f64) -> f64 {
    if depth > 0.0 {
        velocity.abs() + (GRAVITY * depth).sqrt()
    } else {
        0.0
    }
}

/// Active surface cell of a grid edge face of the 4 x 2 grid.
fn face_cell(face: BoundaryFace) -> usize {
    match face.axis {
        Axis::X => face.row * 4 + face.column.min(3),
        Axis::Y => face.row.min(1) * 4 + face.column,
    }
}

/// `|u| + c` of the reach end and of every surface face cell of `port` in `batch`.
fn face_speeds(batch: &Batch, port: &SurfaceChannelPort) -> Result<Vec<f64>> {
    let cells = reach_cells(batch, &port.reach_id)?;
    let end = match port.end {
        ChannelEnd::Left => cells.first(),
        ChannelEnd::Right => cells.last(),
    }
    .ok_or("the reach has cells")?;
    let mut speeds = vec![characteristic(
        end.depth_m,
        end.velocity_m_per_s.unwrap_or(0.0),
    )];
    let surface = batch.surface().state()?;
    for &face in &port.faces {
        let cell = surface.cells()[face_cell(face)];
        let normal = match face.axis {
            Axis::X => cell.qx,
            Axis::Y => cell.qy,
        };
        let velocity = if cell.h > 0.0 { normal / cell.h } else { 0.0 };
        speeds.push(characteristic(cell.h, velocity));
    }
    Ok(speeds)
}

/// Runs `intervals` of `case`. Every interval checks the port speed against every face state
/// of the start, and the accepted duration against both children's limits.
fn covered(name: &str, case: &Scenario, dt: f64, intervals: usize) -> Result {
    let mut batch = case.build()?;
    for _ in 0..intervals {
        let ports = batch.setup().ports.clone();
        let mut speeds = Vec::with_capacity(ports.len());
        for port in &ports {
            speeds.push(face_speeds(&batch, port)?);
        }
        batch = case.step(&batch, dt)?;
        let step = last(&batch)?;
        let duration = step.end_s - step.start_s;
        for (transfer, required) in step.ports.iter().zip(&speeds) {
            for &speed in required {
                assert!(
                    transfer.signal_speed_m_per_s >= speed,
                    "{name}: {transfer:?} against |u|+c={speed}"
                );
            }
        }
        let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
        assert!(duration <= surface.cfl_limit, "{name}: {surface:?}");
        for reach in &batch.setup().reaches {
            let limits = batch
                .reach(&reach.id)
                .ok_or("the reach exists")?
                .last_step()?
                .ok_or("a reach receipt")?;
            assert!(duration <= limits.cfl_limit_s, "{name} {}", reach.id);
        }
        balanced(&batch)?;
    }
    Ok(())
}

/// Every surface cell at depth `h` with x discharge per width `qx`.
fn moving(h: f64, qx: f64) -> Vec<CellState> {
    vec![CellState { h, qx, qy: 0.0 }; CELLS]
}

/// `alpha` in state `alpha` against `surface` through the east port, on a flat 1 m bed.
fn east_case(alpha: Reach, surface: Vec<CellState>) -> Scenario {
    let mut case = Scenario::rest(1.0, 1.5, 1.0, vec![east("p")]);
    case.surface = surface;
    case.alpha = alpha;
    case.link = false;
    case
}

#[test]
fn the_port_speed_covers_every_face_state_and_both_children_limit_the_interval() -> Result {
    let h = 0.5;
    let q = 2.0 * h * (GRAVITY * h).sqrt();
    let mut seiche = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
    seiche.alpha = Reach::rest(0.5, 1.75);
    covered("seiche", &seiche, 0.25, 20)?;
    covered(
        "supercritical",
        &east_case(Reach::uniform(1.0, h, q), moving(h, -q)),
        0.1,
        10,
    )?;
    covered(
        "dry floodplain",
        &east_case(Reach::uniform(1.0, h, 0.0), vec![still(0.0); CELLS]),
        0.1,
        20,
    )?;
    covered(
        "shock",
        &east_case(Reach::uniform(1.0, 1.0, 10.0), vec![still(1.0); CELLS]),
        0.1,
        10,
    )?;
    let mut north = Scenario::rest(
        1.0,
        1.6,
        0.5,
        vec![port(
            "p",
            ALPHA,
            ChannelEnd::Right,
            (0..4).map(|column| y_face(column, 2)).collect(),
        )],
    );
    north.alpha = Reach::rest(0.5, 1.9);
    north.alpha_size = [8.0, LENGTH];
    north.link = false;
    covered("north edge", &north, 0.25, 20)
}

#[test]
fn the_port_speed_binds_the_common_interval() -> Result {
    // A reach end at 10 m/s runs into still water. Its |u| + c exceeds the HLL bound of the
    // port faces. The port speed enters the end cell's CFL frequency in both directions.
    let (h, u) = (1.0, 10.0);
    let case = east_case(Reach::uniform(1.0, h, h * u), vec![still(1.0); CELLS]);
    let batch = step(&case.build()?, 1.0)?;
    let diagnostics = last(&batch)?;
    assert!(
        diagnostics.retries >= 1,
        "the requested second must shorten"
    );
    let transfer = &diagnostics.ports[0];
    let surface = batch.surface().last_step()?.ok_or("a surface receipt")?;
    let hll = surface
        .faces
        .iter()
        .filter(|face| case.ports[0].faces.contains(&face.face))
        .map(|face| face.signal_speed)
        .fold(0.0, f64::max);
    assert!(hll < characteristic(h, u) && transfer.signal_speed_m_per_s >= characteristic(h, u));
    let reach = batch
        .reach(ALPHA)
        .ok_or("alpha")?
        .last_step()?
        .ok_or("a reach receipt")?;
    // The end cell takes the port speed and the interior face's |u| + c as inward waves.
    let cfl = ChannelConfig::default().cfl;
    let bound = cfl * 2.0 * LENGTH / (transfer.signal_speed_m_per_s + characteristic(h, u));
    assert!(reach.cfl_limit_s <= bound, "{reach:?}");
    assert!(reach.cfl_limit_s < surface.cfl_limit, "{surface:?}");
    let dt = diagnostics.end_s - diagnostics.start_s;
    assert!(dt <= reach.cfl_limit_s && dt >= 0.9 * reach.cfl_limit_s);
    Ok(())
}

/// One-lane F64 tensor fields of one side of each port face.
fn side(states: &[[f64; 4]]) -> Result<TensorFaceState<crate::runtime::Runtime>> {
    let device = numr::runtime::cpu::CpuDevice::new();
    let field = |index: usize| -> Result<Tensor<crate::runtime::Runtime>> {
        let values: Vec<f64> = states.iter().map(|state| state[index]).collect();
        Ok(Tensor::from_slice(&values, &[values.len()], &device)?)
    };
    Ok(TensorFaceState {
        h: field(0)?,
        q_normal: field(1)?,
        q_tangent: field(2)?,
        bed: field(3)?,
    })
}

/// Face-frame HLL flux of the west port faces from the accepted start states of `batch`.
///
/// The ghost is the reach's left end cell. A left end maps reach-positive discharge onto `-x`.
/// Returns `[water, normal, tangent]` and the speed of each face, in face line order.
fn west_reference(batch: &Batch, faces: &[BoundaryFace]) -> Result<Vec<[f64; 4]>> {
    let end = reach_cells(batch, ALPHA)?[0].clone();
    let reach_bed = batch.setup().reaches[0].definition.geometry.clone();
    let hydr::channel::ChannelGeometry::Rectangular { cells } = reach_bed else {
        return Err("alpha is rectangular".into());
    };
    let ghost = [
        end.depth_m,
        -(end.discharge_m3_per_s / cells[0].width_m),
        0.0,
        cells[0].bed_elevation_m,
    ];
    let surface = batch.surface().state()?;
    let bed = batch.surface().domain().bed();
    let cells: Vec<[f64; 4]> = faces
        .iter()
        .map(|&face| {
            let cell = face_cell(face);
            let state = surface.cells()[cell];
            [state.h, state.qx, state.qy, bed[cell]]
        })
        .collect();
    let flux = hydrostatic_hll_tensor(
        &crate::runtime::client(),
        side(&vec![ghost; faces.len()])?,
        side(&cells)?,
        GRAVITY,
        &SolverConfig::default(),
    )?;
    let read = |tensor: &Tensor<crate::runtime::Runtime>| tensor.try_to_vec::<f64>();
    let (water, normal, tangent, speed) = (
        read(&flux.shared[0])?,
        read(&flux.shared[1])?,
        read(&flux.shared[2])?,
        read(&flux.speed)?,
    );
    Ok((0..faces.len())
        .map(|lane| [water[lane], normal[lane], tangent[lane], speed[lane]])
        .collect())
}

/// Receipt of `face` in the last surface interval of `batch`.
fn receipt(batch: &Batch, face: BoundaryFace) -> Result<FaceTransfer> {
    let step = batch.surface().last_step()?.ok_or("a surface receipt")?;
    Ok(step
        .faces
        .into_iter()
        .find(|transfer| transfer.face == face)
        .ok_or("the port face receipt")?)
}

#[test]
fn a_draining_end_cell_takes_the_hll_flux_of_its_start_states() -> Result {
    // Only the reach end cell holds water, and the floodplain is dry.
    let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
    case.link = false;
    case.alpha = Reach {
        bed: 1.0,
        depth: [0.05, 0.0, 0.0],
        discharge_per_width: [0.0; 3],
    };
    let faces = case.ports[0].faces.clone();
    let mut batch = case.build()?;
    let held = reach_cells(&batch, ALPHA)?[0].volume_m3;
    let mut drained = 0.0;
    for index in 0..10 {
        let reference = west_reference(&batch, &faces)?;
        batch = step(&batch, 1.0)?;
        let diagnostics = last(&batch)?;
        let transfer = &diagnostics.ports[0];
        if index == 0 {
            assert!(
                diagnostics.retries >= 1,
                "the requested second must shorten"
            );
            assert_eq!(transfer.regime, SurfaceChannelPortRegime::WetDryFront);
        }
        // Each port face takes the HLL flux of the start states, bit for bit.
        let dt = diagnostics.end_s - diagnostics.start_s;
        let mut rate = 0.0;
        for (&face, expected) in faces.iter().zip(&reference) {
            let accepted = receipt(&batch, face)?;
            let bits = |values: &[f64]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&accepted.flux), bits(&expected[..3]), "{accepted:?}");
            assert_eq!(accepted.signal_speed.to_bits(), expected[3].to_bits());
            rate += accepted.flux[0] * 2.0;
        }
        assert_eq!(
            transfer.signed_volume_m3.to_bits(),
            (-(rate * dt)).to_bits()
        );
        drained -= transfer.signed_volume_m3;
        // The CFL limit binds before the depletion limit: depletion never scales the flux.
        let limits = batch
            .reach(ALPHA)
            .ok_or("alpha")?
            .last_step()?
            .ok_or("a reach receipt")?;
        assert!(dt <= limits.cfl_limit_s && limits.cfl_limit_s <= limits.depletion_limit_s);
        assert!(
            reach_cells(&batch, ALPHA)?
                .iter()
                .all(|cell| cell.volume_m3 >= 0.0)
        );
        assert!(
            batch
                .surface()
                .state()?
                .cells()
                .iter()
                .all(|cell| cell.h >= 0.0)
        );
        balanced(&batch)?;
    }
    assert!(drained > 0.0 && drained <= held);
    Ok(())
}
