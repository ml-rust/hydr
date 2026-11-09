//! Wet-subcritical stage/discharge, orientation, accounting, and transactional boundary checks.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, DischargeBoundary,
    ForcingInterval, ForcingSchedule, Solver, SolverConfig, StageBoundary,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn domain(nx: usize, ny: usize, bed: Vec<f64>) -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(nx, ny, 1.0, 1.0)?,
        bed,
        vec![0.0; nx * ny],
        vec![true; nx * ny],
        SpatialReference::new(
            "synthetic projected metres".into(),
            "synthetic SI datum".into(),
            [0.0; 2],
        )?,
    )?)
}

fn all_faces(nx: usize, ny: usize) -> Vec<BoundaryFace> {
    let mut faces = Vec::new();
    for row in 0..ny {
        for column in [0, nx] {
            faces.push(BoundaryFace {
                axis: Axis::X,
                column,
                row,
            });
        }
    }
    for column in 0..nx {
        for row in [0, ny] {
            faces.push(BoundaryFace {
                axis: Axis::Y,
                column,
                row,
            });
        }
    }
    faces
}

fn solver(
    domain: Domain,
    cells: Vec<CellState>,
    intervals: Vec<ForcingInterval>,
    config: SolverConfig,
) -> Result<Solver, Box<dyn std::error::Error>> {
    let faces = intervals
        .first()
        .ok_or("missing forcing interval")?
        .boundaries
        .iter()
        .map(BoundaryForcing::face)
        .collect();
    let boundaries = BoundaryConditions::with_open_faces(&domain, faces)?;
    let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
    let state = FlowState::new(&domain, cells)?;
    Ok(Solver::new(
        domain, state, boundaries, forcing, config, 0.0,
    )?)
}

fn interval(
    start: f64,
    end: f64,
    count: usize,
    boundaries: Vec<BoundaryForcing>,
) -> ForcingInterval {
    ForcingInterval {
        start,
        end,
        rain: vec![0.0; count],
        withdrawal: vec![0.0; count],
        boundaries,
    }
}

fn stage(face: BoundaryFace, elevation: f64, tangent: Option<f64>) -> BoundaryForcing {
    BoundaryForcing::Stage(StageBoundary {
        face,
        elevation,
        inflow_tangential_velocity: tangent,
    })
}

fn discharge(face: BoundaryFace, rate: f64, tangent: Option<f64>) -> BoundaryForcing {
    BoundaryForcing::Discharge(DischargeBoundary {
        face,
        outward_rate: rate,
        inflow_tangential_velocity: tangent,
    })
}

#[test]
fn matching_stage_preserves_a_stepped_lake_with_open_faces() -> TestResult {
    let bed = vec![0.0, 0.2, 0.5, 0.1];
    let cells: Vec<_> = bed
        .iter()
        .map(|bed| CellState {
            h: 1.0 - bed,
            qx: 0.0,
            qy: 0.0,
        })
        .collect();
    let mut engine = solver(
        domain(2, 2, bed)?,
        cells.clone(),
        vec![interval(
            0.0,
            1.0,
            4,
            all_faces(2, 2)
                .into_iter()
                .map(|face| stage(face, 1.0, Some(0.0)))
                .collect(),
        )],
        SolverConfig {
            max_step: 0.01,
            ..SolverConfig::default()
        },
    )?;
    assert_eq!(
        engine.advance_to(0.2, 10_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    for (actual, initial) in engine.state().cells().iter().zip(cells) {
        assert!((actual.h - initial.h).abs() <= 1e-11);
        assert!(actual.qx.abs() <= 1e-11 && actual.qy.abs() <= 1e-11);
    }
    assert!(engine.ledger().residual.abs() <= 1e-9 + 1e-12 * engine.ledger().storage.max(1.0));
    assert_eq!(
        engine
            .last_step()
            .ok_or("missing accepted step")?
            .boundary_traces
            .len(),
        8
    );
    Ok(())
}

fn uniform(discharge_mode: bool, u: f64, v: f64) -> Result<Solver, Box<dyn std::error::Error>> {
    let faces = all_faces(1, 1);
    let data = faces
        .into_iter()
        .map(|face| {
            let (normal, tangent, positive) = if face.axis == Axis::X {
                (u, v, face.column == 1)
            } else {
                (v, u, face.row == 1)
            };
            if discharge_mode {
                discharge(face, if positive { normal } else { -normal }, Some(tangent))
            } else {
                stage(face, 1.0, Some(tangent))
            }
        })
        .collect();
    solver(
        domain(1, 1, vec![0.0])?,
        vec![CellState {
            h: 1.0,
            qx: u,
            qy: v,
        }],
        vec![interval(0.0, 1.0, 1, data)],
        SolverConfig {
            max_step: 0.01,
            ..SolverConfig::default()
        },
    )
}

#[test]
fn four_orientations_preserve_uniform_flow_and_exact_discharge_budgets() -> TestResult {
    for discharge_mode in [false, true] {
        let mut original = uniform(discharge_mode, 0.5, -0.2)?;
        let mut rotated = uniform(discharge_mode, 0.2, 0.5)?;
        original.advance_to(0.2, 10_000)?;
        rotated.advance_to(0.2, 10_000)?;
        let cell = original.state().cells()[0];
        let turn = rotated.state().cells()[0];
        assert!((cell.h - 1.0).abs() <= 1e-11);
        assert!((cell.qx - 0.5).abs() <= 1e-11 && (cell.qy + 0.2).abs() <= 1e-11);
        assert!(
            (turn.h - cell.h).abs() <= 1e-11
                && (turn.qx + cell.qy).abs() <= 1e-11
                && (turn.qy - cell.qx).abs() <= 1e-11
        );
        for engine in [&original, &rotated] {
            assert!((engine.ledger().boundary_in - 0.14).abs() <= 1e-12);
            assert!((engine.ledger().boundary_out - 0.14).abs() <= 1e-12);
            assert_eq!(engine.ledger().water_adjustment, 0.0);
        }
        let data = original.snapshot().data();
        let mut restored = uniform(discharge_mode, 0.5, -0.2)?;
        restored.restore_data(&data)?;
        assert_eq!(restored.snapshot(), original.snapshot());
    }
    Ok(())
}

#[test]
fn stage_reversal_switches_tangent_and_splits_forcing_intervals() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 0,
    };
    let initial = CellState {
        h: 1.0,
        qx: 0.0,
        qy: -0.3,
    };
    let mut engine = solver(
        domain(1, 1, vec![0.0])?,
        vec![initial],
        vec![
            interval(0.0, 0.02, 1, vec![stage(face, 1.2, Some(0.4))]),
            interval(0.02, 0.04, 1, vec![stage(face, 0.8, Some(-1.0))]),
        ],
        SolverConfig {
            max_step: 0.015,
            ..SolverConfig::default()
        },
    )?;
    let mut previous = initial;
    let mut incoming = false;
    let mut outgoing = false;
    let mut split = false;
    let report = engine.advance_controlled(
        0.04,
        10_000,
        || false,
        |step, cells| {
            assert!(!(step.start < 0.02 && step.end > 0.02));
            split |= step.end == 0.02;
            let trace = &step.boundary_traces[0];
            if trace.outward_water_rate < 0.0 {
                incoming = true;
                assert!((trace.tangential_velocity - 0.4).abs() <= 1e-12);
            } else if trace.outward_water_rate > 0.0 {
                outgoing = true;
                assert!((trace.tangential_velocity - previous.qy / previous.h).abs() <= 1e-12);
            }
            previous = cells[0];
        },
    )?;
    assert_eq!(report.status, AdvanceStatus::ReachedTarget);
    assert!(incoming && outgoing && split);
    Ok(())
}

#[test]
fn inadmissible_boundary_trials_preserve_the_accepted_snapshot() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    for (cell, request) in [
        (
            CellState {
                h: 0.0,
                qx: 0.0,
                qy: 0.0,
            },
            stage(face, 1.0, Some(0.0)),
        ),
        (
            CellState {
                h: 1.0,
                qx: 4.0,
                qy: 0.0,
            },
            stage(face, 1.0, Some(0.0)),
        ),
        (
            CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0,
            },
            stage(face, 1.2, None),
        ),
        (
            CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0,
            },
            discharge(face, 1000.0, None),
        ),
    ] {
        let mut engine = solver(
            domain(1, 1, vec![0.0])?,
            vec![cell],
            vec![interval(0.0, 1.0, 1, vec![request])],
            SolverConfig {
                max_step: 0.01,
                ..SolverConfig::default()
            },
        )?;
        let before = engine.snapshot();
        assert!(engine.advance_to(0.01, 1000).is_err());
        assert_eq!(engine.snapshot(), before);
    }
    Ok(())
}

#[test]
fn boundary_source_retries_reproduce_direct_accepted_transport() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    let create = |max_step| -> Result<Solver, Box<dyn std::error::Error>> {
        let mut forcing = interval(0.0, 1.0, 1, vec![discharge(face, 0.8, None)]);
        forcing.withdrawal[0] = 1.0;
        solver(
            domain(1, 1, vec![0.0])?,
            vec![CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0,
            }],
            vec![forcing],
            SolverConfig {
                max_step,
                ..SolverConfig::default()
            },
        )
    };
    let mut retried = create(1.0)?;
    let report = retried.advance_to(1.0, 1)?;
    assert_eq!(report.status, AdvanceStatus::StepLimit);
    let step = retried.last_step().ok_or("missing accepted retry")?;
    assert!(step.retries > 0 && step.dt < 1.0);
    let dt = step.dt;
    let mut direct = create(dt)?;
    assert_eq!(
        direct.advance_to(dt, 1)?.status,
        AdvanceStatus::ReachedTarget
    );
    assert_eq!(retried.state(), direct.state());
    assert_eq!(retried.ledger(), direct.ledger());
    let direct_step = direct.last_step().ok_or("missing direct step")?;
    assert_eq!(step.faces, direct_step.faces);
    assert_eq!(step.boundary_traces, direct_step.boundary_traces);
    Ok(())
}

#[test]
fn physical_boundary_restart_and_trace_import_remain_complete() -> TestResult {
    let mut source = uniform(true, 0.5, -0.2)?;
    source.advance_to(0.04, 10_000)?;
    let checkpoint = source.snapshot().data();
    let mut restored = uniform(true, 0.5, -0.2)?;
    restored.restore_data(&checkpoint)?;
    source.advance_to(0.08, 10_000)?;
    restored.advance_to(0.08, 10_000)?;
    assert_eq!(source.snapshot(), restored.snapshot());
    let accepted = restored.snapshot();
    let valid = accepted.data();
    let mut wrong_rate = valid.clone();
    wrong_rate
        .last_step
        .as_mut()
        .ok_or("missing last step")?
        .boundary_traces[0]
        .outward_water_rate += 0.01;
    let mut duplicate = valid.clone();
    let last = duplicate.last_step.as_mut().ok_or("missing last step")?;
    last.boundary_traces.push(last.boundary_traces[0].clone());
    let mut high_speed = valid.clone();
    high_speed
        .last_step
        .as_mut()
        .ok_or("missing last step")?
        .boundary_traces[0]
        .signal_speed = 1e100;
    let error = restored
        .restore_data(&high_speed)
        .err()
        .ok_or("contradictory boundary CFL was admitted")?;
    assert!(error.to_string().contains("boundary CFL"));
    assert_eq!(restored.snapshot(), accepted);
    let mut wrong_config = valid.clone();
    wrong_config.config.boundary.discharge_absolute =
        f64::from_bits(wrong_config.config.boundary.discharge_absolute.to_bits() + 1);
    for altered in [wrong_rate, duplicate, wrong_config] {
        assert!(restored.restore_data(&altered).is_err());
        assert_eq!(restored.snapshot(), accepted);
    }
    let source = uniform(false, 0.0, 0.0)?;
    let mut signed_zero = uniform(false, 0.0, -0.0)?;
    let before = signed_zero.snapshot();
    assert!(signed_zero.restore(&source.snapshot()).is_err());
    assert!(signed_zero.restore_data(&source.snapshot().data()).is_err());
    assert_eq!(signed_zero.snapshot(), before);
    Ok(())
}

#[test]
fn positive_film_inflow_preserves_primitive_tangent_in_checkpoint_diagnostics() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 0,
    };
    let create = || -> Result<Solver, Box<dyn std::error::Error>> {
        solver(
            domain(1, 1, vec![0.0])?,
            vec![CellState {
                h: 1e-200,
                qx: 0.0,
                qy: 0.0,
            }],
            vec![interval(
                0.0,
                1.0,
                1,
                vec![stage(face, 1.2e-200, Some(1e-200))],
            )],
            SolverConfig {
                max_step: 0.01,
                ..SolverConfig::default()
            },
        )
    };
    let mut source = create()?;
    assert_eq!(
        source.advance_to(0.01, 100)?.status,
        AdvanceStatus::ReachedTarget
    );
    assert!(source.state().cells()[0].h > 0.0);
    let step = source.last_step().ok_or("missing accepted film step")?;
    let trace = &step.boundary_traces[0];
    assert!(trace.outward_water_rate < 0.0);
    assert_eq!(trace.tangential_velocity.to_bits(), 1e-200_f64.to_bits());
    assert_eq!(trace.depth * trace.tangential_velocity, 0.0);
    let transfer = step
        .faces
        .iter()
        .find(|transfer| transfer.face == face)
        .ok_or("missing boundary transfer")?;
    assert_eq!(transfer.flux[2], 0.0);
    let checkpoint = source.snapshot();
    let mut trusted = create()?;
    trusted.restore(&checkpoint)?;
    assert_eq!(trusted.snapshot(), checkpoint);
    let mut packaged = create()?;
    packaged.restore_data(&checkpoint.data())?;
    assert_eq!(packaged.snapshot(), checkpoint);
    Ok(())
}
