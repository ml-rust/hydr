//! Source momentum, full-flux port signs, and donor-budget contracts in SI units.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/impervious.rs"]
mod impervious;

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, FluxPort, ForcingInterval,
    ForcingSchedule, Solver, SolverConfig,
};
use impervious::impervious_solver;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn flat_domain(
    nx: usize,
    ny: usize,
    dx: f64,
    dy: f64,
) -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(nx, ny, dx, dy)?,
        vec![0.0; nx * ny],
        vec![0.0; nx * ny],
        vec![true; nx * ny],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic datum".into(),
            [0.0; 2],
        )?,
    )?)
}

fn full_ports(domain: &Domain, cell: CellState) -> Vec<FluxPort> {
    let (nx, ny) = domain.grid().dimensions();
    let (dx, dy) = domain.grid().spacing();
    let pressure = 0.5 * 9.81 * cell.h * cell.h;
    let x_flux = [
        cell.qx,
        cell.qx * cell.qx / cell.h + pressure,
        cell.qx * cell.qy / cell.h,
    ];
    let y_flux = [
        cell.qy,
        cell.qx * cell.qy / cell.h,
        cell.qy * cell.qy / cell.h + pressure,
    ];
    let speed =
        (9.81 * cell.h).sqrt() + (cell.qx / cell.h).abs().max((cell.qy / cell.h).abs()) + 1.0;
    let mut ports = Vec::new();
    for (axis, count, outer, width, flux) in
        [(Axis::X, ny, nx, dy, x_flux), (Axis::Y, nx, ny, dx, y_flux)]
    {
        for tangent in 0..count {
            for (normal, sign) in [(0, -1.0), (outer, 1.0)] {
                let face = match axis {
                    Axis::X => BoundaryFace {
                        axis,
                        column: normal,
                        row: tangent,
                    },
                    Axis::Y => BoundaryFace {
                        axis,
                        column: tangent,
                        row: normal,
                    },
                };
                ports.push(FluxPort {
                    face,
                    water_rate: sign * flux[0] * width,
                    momentum_rate: [sign * flux[1] * width, sign * flux[2] * width],
                    signal_speed: speed,
                    provenance: "Uniform flat-bed physical full flux, speed exceeds gravity-wave speed plus both component velocities".into(),
                });
            }
        }
    }
    ports
}

fn solver(
    domain: Domain,
    cells: Vec<CellState>,
    ports: Vec<FluxPort>,
    withdrawal: f64,
    config: SolverConfig,
) -> Result<Solver<crate::runtime::Runtime>, Box<dyn std::error::Error>> {
    let count = domain.grid().cell_count();
    let state = FlowState::new(&domain, cells)?;
    let boundaries =
        BoundaryConditions::with_open_faces(&domain, ports.iter().map(|port| port.face).collect())?;
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![ForcingInterval {
            start: 0.0,
            end: 2.0,
            potential_evaporation: vec![0.0; count],
            rain: vec![0.0; count],
            withdrawal: vec![withdrawal; count],
            boundaries: ports.into_iter().map(Into::into).collect(),
        }],
    )?;
    impervious_solver(domain, state, boundaries, forcing, config, 0.0)
}

#[test]
fn opposing_full_flux_ports_preserve_two_dimensional_uniform_flow() -> TestResult {
    let domain = flat_domain(3, 2, 2.0, 3.0)?;
    let cell = CellState {
        h: 1.0,
        qx: 0.2,
        qy: 0.1,
    };
    let ports = full_ports(&domain, cell);
    assert_eq!(ports.len(), 10);
    let mut solver = solver(
        domain,
        vec![cell; 6],
        ports.clone(),
        0.0,
        SolverConfig::default(),
    )?;
    assert_eq!(
        solver.advance_to(0.1, 100)?.status,
        AdvanceStatus::ReachedTarget
    );
    for actual in solver.state()?.cells() {
        assert!((actual.h - cell.h).abs() <= 1e-12, "{actual:?}");
        assert!((actual.qx - cell.qx).abs() <= 1e-12, "{actual:?}");
        assert!((actual.qy - cell.qy).abs() <= 1e-12, "{actual:?}");
    }
    let step = solver.last_step()?.ok_or("accepted step is absent")?;
    for port in &ports {
        let face = step
            .faces
            .iter()
            .find(|face| face.face == port.face)
            .ok_or("port diagnostic is absent")?;
        let (sign, width) = match port.face.axis {
            Axis::X => (if port.face.column == 0 { -1.0 } else { 1.0 }, 3.0),
            Axis::Y => (if port.face.row == 0 { -1.0 } else { 1.0 }, 2.0),
        };
        assert!((face.flux[0] - sign * port.water_rate / width).abs() <= 1e-12);
        assert!((face.flux[1] - sign * port.momentum_rate[0] / width).abs() <= 1e-12);
        assert!((face.flux[2] - sign * port.momentum_rate[1] / width).abs() <= 1e-12);
        assert!(face.volume > 0.0);
        assert_eq!(face.pressure_corrections, [0.0; 2]);
    }
    assert!((solver.ledger().boundary_in - solver.ledger().boundary_out).abs() <= 1e-12);
    Ok(())
}

#[test]
fn competing_interior_and_port_exports_share_one_donor_budget() -> TestResult {
    let domain = flat_domain(2, 1, 1.0, 1.0)?;
    let port = FluxPort {
        face: BoundaryFace { axis: Axis::X, column: 0, row: 0 },
        water_rate: 10.0,
        momentum_rate: [-104.905, 0.0],
        signal_speed: 14.0,
        provenance: "Manufactured westward h=1 m, normal velocity=-10 m/s full flux with speed bound above 10+sqrt(9.81)".into(),
    };
    let mut solver = solver(
        domain,
        vec![
            CellState {
                h: 1.0,
                ..CellState::default()
            },
            CellState {
                h: 0.25,
                ..CellState::default()
            },
        ],
        vec![port],
        0.0,
        SolverConfig::default(),
    )?;
    let report = solver.advance_to(1.0, 1)?;
    assert_eq!(report.status, AdvanceStatus::StepLimit);
    assert_eq!(report.accepted_steps, 1);
    let step = solver.last_step()?.ok_or("accepted step is absent")?;
    assert!(step.retries > 0);
    let interior = step
        .faces
        .iter()
        .find(|face| {
            face.face
                == BoundaryFace {
                    axis: Axis::X,
                    column: 1,
                    row: 0,
                }
        })
        .ok_or("interior face diagnostic is absent")?;
    assert!(interior.flux[0] > 0.0);
    let outgoing_rates: Vec<_> = step
        .faces
        .iter()
        .filter_map(|face| {
            if face.left == Some(0) && face.flux[0] > 0.0 {
                Some(face.flux[0])
            } else if face.right == Some(0) && face.flux[0] < 0.0 {
                Some(-face.flux[0])
            } else {
                None
            }
        })
        .collect();
    assert_eq!(outgoing_rates.len(), 2);
    let combined_rate: f64 = outgoing_rates.iter().sum();
    let combined_limit = 1.0 / combined_rate;
    let individual_limit = 1.0 / outgoing_rates.iter().copied().fold(0.0_f64, f64::max);
    assert!((step.depletion_limit - combined_limit).abs() <= 1e-12);
    assert!(step.depletion_limit < individual_limit);
    assert!(step.dt <= combined_limit);
    assert!(combined_rate * step.dt <= 1.0);
    assert!((step.boundary_out - 10.0 * step.dt).abs() <= 1e-12);
    assert!(solver.state()?.cells().iter().all(|cell| cell.h >= 0.0));
    assert_eq!(solver.ledger().water_adjustment, 0.0);
    Ok(())
}

#[test]
fn land_withdrawal_removes_proportional_donor_momentum() -> TestResult {
    let domain = flat_domain(7, 1, 10.0, 10.0)?;
    let cell = CellState {
        h: 2.0,
        qx: 1.0,
        qy: 0.0,
    };
    let mut solver = solver(
        domain,
        vec![cell; 7],
        vec![],
        0.1,
        SolverConfig {
            max_step: 0.1,
            ..SolverConfig::default()
        },
    )?;
    solver.advance_to(0.1, 1)?;
    let step = solver.last_step()?.ok_or("accepted step is absent")?;
    assert_eq!(step.dt, 0.1);
    let center = solver.state()?.cells()[3];
    assert!((center.h - 1.99).abs() <= 1e-12);
    assert!((center.qx - 0.995).abs() <= 1e-12);
    assert_eq!(center.qy, 0.0);
    assert!((step.withdrawal_volume - 7.0).abs() <= 1e-12);
    assert_eq!(solver.ledger().momentum_removed, [0.0; 2]);
    assert_eq!(solver.ledger().water_adjustment, 0.0);
    Ok(())
}

#[test]
fn film_resets_record_signed_momentum_without_removing_water() -> TestResult {
    let domain = flat_domain(3, 2, 2.0, 3.0)?;
    let cell = CellState {
        h: 0.001,
        qx: 0.002,
        qy: -0.003,
    };
    let ports = full_ports(&domain, cell);
    let mut solver = solver(
        domain,
        vec![cell; 6],
        ports,
        0.0,
        SolverConfig {
            max_step: 1e-5,
            momentum_film_depth: 0.01,
            ..SolverConfig::default()
        },
    )?;
    solver.advance_to(0.1, 1)?;
    let step = solver.last_step()?.ok_or("accepted step is absent")?;
    for actual in solver.state()?.cells() {
        assert!((actual.h - cell.h).abs() <= 1e-15);
        assert!(actual.h > 0.0);
        assert_eq!([actual.qx, actual.qy], [0.0; 2]);
    }
    let expected = [cell.qx * 36.0, cell.qy * 36.0];
    for (axis, value) in expected.iter().enumerate() {
        assert!((step.momentum_removed[axis] - value).abs() <= 1e-12);
        assert!((solver.ledger().momentum_removed[axis] - value).abs() <= 1e-12);
    }
    assert!(step.momentum_removed[0] > 0.0);
    assert!(step.momentum_removed[1] < 0.0);
    assert!((solver.ledger().storage - solver.ledger().initial_storage).abs() <= 1e-12);
    assert_eq!(solver.ledger().water_adjustment, 0.0);
    Ok(())
}
