//! Exterior admission, transmissive witnesses, retry, and checkpoint contracts.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/exterior_fixture.rs"]
mod fixture;

use fixture::{TestResult, create, interval, through, uniform, wet};
use hydr::model::CellState;
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryFace, BoundaryForcing, DischargeBoundary, FluxPort, SolverConfig,
    StageBoundary,
};

#[test]
fn dry_transmissive_states_preserve_zero_flux_and_unrestricted_checkpoint_cfl() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    let make = || {
        create(
            1,
            1,
            vec![0.0],
            vec![CellState::default()],
            vec![interval(
                0.0,
                1.0,
                1,
                vec![BoundaryForcing::Transmissive(face)],
            )],
            SolverConfig {
                max_step: 0.1,
                ..SolverConfig::default()
            },
        )
    };
    let mut source = make()?;
    source.advance_to(0.1, 100)?;
    let step = source.last_step()?.ok_or("missing accepted dry step")?;
    assert_eq!(step.cfl_limit, f64::INFINITY);
    assert!(
        step.faces
            .iter()
            .all(|face| face.flux == [0.0; 3] && face.signal_speed == 0.0)
    );
    assert_eq!(step.transmissive_states.len(), 1);
    assert_eq!(step.transmissive_states[0].depth, 0.0);
    let checkpoint = source.snapshot();
    let mut restored = make()?;
    restored.restore_data(&checkpoint.data()?)?;
    assert_eq!(restored.snapshot().data()?, checkpoint.data()?);
    Ok(())
}

#[test]
fn transmissive_incoming_or_critical_characteristics_reject_without_reflection() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    for velocity in [0.0, 9.81_f64.sqrt(), -5.0] {
        let mut engine = create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: velocity,
                qy: 0.0,
            }],
            vec![interval(
                0.0,
                1.0,
                1,
                vec![BoundaryForcing::Transmissive(face)],
            )],
            SolverConfig {
                max_step: 0.01,
                ..SolverConfig::default()
            },
        )?;
        let before = engine.snapshot();
        assert!(engine.advance_to(0.01, 100).is_err());
        assert_eq!(engine.snapshot().data()?, before.data()?);
    }
    Ok(())
}

#[test]
fn post_source_transmissive_retries_match_direct_accepted_execution() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    let make = |max_step| {
        let mut forcing = interval(0.0, 1.0, 1, vec![BoundaryForcing::Transmissive(face)]);
        forcing.rain[0] = 10.0;
        create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: 5.0,
                qy: 0.0,
            }],
            vec![forcing],
            SolverConfig {
                max_step,
                ..SolverConfig::default()
            },
        )
    };
    let mut retried = make(0.1)?;
    assert_eq!(retried.advance_to(0.1, 1)?.status, AdvanceStatus::StepLimit);
    let step = retried.last_step()?.ok_or("missing accepted retry")?;
    assert!(step.retries >= 2 && step.dt <= 0.025);
    let mut direct = make(step.dt)?;
    direct.advance_to(step.dt, 1)?;
    let direct_step = direct.last_step()?.ok_or("missing direct step")?;
    assert_eq!(retried.state()?, direct.state()?);
    assert_eq!(retried.ledger(), direct.ledger());
    assert_eq!(step.faces, direct_step.faces);
    assert_eq!(step.transmissive_states, direct_step.transmissive_states);
    Ok(())
}

#[test]
fn exterior_and_transmissive_restart_reject_contradictory_speed_or_state_witnesses() -> TestResult {
    let mut source = through(Axis::X, 1.0)?;
    source.advance_to(0.04, 1000)?;
    let checkpoint = source.snapshot();
    let mut restored = through(Axis::X, 1.0)?;
    restored.restore(&checkpoint)?;
    let mut packaged = through(Axis::X, 1.0)?;
    packaged.restore_data(&checkpoint.data()?)?;
    for engine in [&mut source, &mut restored, &mut packaged] {
        engine.advance_to(0.08, 1000)?;
    }
    assert_eq!(source.snapshot().data()?, restored.snapshot().data()?);
    assert_eq!(source.snapshot().data()?, packaged.snapshot().data()?);
    let accepted = packaged.snapshot();
    let valid = accepted.data()?;
    let mut speed = valid.clone();
    speed.last_step.as_mut().ok_or("missing last step")?.faces[0].signal_speed *= 2.0;
    let mut cfl = valid.clone();
    cfl.last_step.as_mut().ok_or("missing last step")?.cfl_limit *= 0.5;
    let mut witness = valid.clone();
    witness
        .last_step
        .as_mut()
        .ok_or("missing last step")?
        .transmissive_states[0]
        .outward_discharge_per_width += 0.25;
    for altered in [speed, cfl, witness] {
        assert!(packaged.restore_data(&altered).is_err());
        assert_eq!(packaged.snapshot().data()?, accepted.data()?);
    }
    let source = uniform(1, 1, vec![0.0], 1.0, [0.0, 0.0])?;
    let mut signed = uniform(1, 1, vec![0.0], 1.0, [0.0, -0.0])?;
    let before = signed.snapshot();
    assert!(signed.restore(&source.snapshot()).is_err());
    assert!(signed.restore_data(&source.snapshot().data()?).is_err());
    assert_eq!(signed.snapshot().data()?, before.data()?);
    Ok(())
}

#[test]
fn exterior_data_require_complete_unique_representable_states() -> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    for (bed, elevation, normal, tangent) in [
        (0.0, 1.0, f64::NAN, 0.0),
        (-f64::MAX, f64::MAX, 0.0, 0.0),
        (0.0, 1e200, 1e200, 0.0),
        (0.0, 1e-200, 1e-200, 0.0),
        (0.0, 1e-200, 0.0, 1e-200),
        (0.0, 0.0, 0.0, 0.0),
    ] {
        assert!(
            create(
                1,
                1,
                vec![bed],
                vec![CellState {
                    h: 1.0,
                    qx: 0.0,
                    qy: 0.0
                }],
                vec![interval(
                    0.0,
                    1.0,
                    1,
                    vec![wet(face, elevation, normal, tangent)]
                )],
                SolverConfig::default()
            )
            .is_err()
        );
    }
    let data = wet(face, 1.0, 0.0, 0.0);
    assert!(
        create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0
            }],
            vec![interval(0.0, 1.0, 1, vec![data.clone(), data.clone()])],
            SolverConfig::default()
        )
        .is_err()
    );
    assert!(
        create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0
            }],
            vec![
                interval(0.0, 0.5, 1, vec![data]),
                interval(0.5, 1.0, 1, vec![])
            ],
            SolverConfig::default()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn all_five_boundary_policies_coexist_with_independent_trace_and_witness_sets() -> TestResult {
    let face = |axis, column, row| BoundaryFace { axis, column, row };
    let mut boundaries = vec![
        wet(face(Axis::X, 0, 0), 1.0, -5.0, 0.0),
        BoundaryForcing::Flux(FluxPort {
            face: face(Axis::X, 0, 1),
            water_rate: -5.0,
            momentum_rate: [-29.905, 0.0],
            signal_speed: 9.0,
            provenance: "Uniform h=1, qx=5, qy=0 full flux, |u|+sqrt(gh)<9".into(),
        }),
        BoundaryForcing::Transmissive(face(Axis::X, 2, 0)),
        BoundaryForcing::Transmissive(face(Axis::X, 2, 1)),
    ];
    for row in [0, 2] {
        boundaries.push(BoundaryForcing::Stage(StageBoundary {
            face: face(Axis::Y, 0, row),
            elevation: 1.0,
            inflow_tangential_velocity: Some(5.0),
        }));
        boundaries.push(BoundaryForcing::Discharge(DischargeBoundary {
            face: face(Axis::Y, 1, row),
            outward_rate: 0.0,
            inflow_tangential_velocity: Some(5.0),
        }));
    }
    let mut engine = create(
        2,
        2,
        vec![0.0; 4],
        vec![
            CellState {
                h: 1.0,
                qx: 5.0,
                qy: 0.0
            };
            4
        ],
        vec![interval(0.0, 1.0, 4, boundaries)],
        SolverConfig {
            max_step: 0.01,
            ..SolverConfig::default()
        },
    )?;
    engine.advance_to(0.02, 1000)?;
    for cell in engine.state()?.cells() {
        assert!(
            (cell.h - 1.0).abs() <= 1e-11
                && (cell.qx - 5.0).abs() <= 1e-11
                && cell.qy.abs() <= 1e-11
        );
    }
    assert!(
        (engine.ledger().boundary_in - 0.2).abs() <= 1e-12
            && (engine.ledger().boundary_out - 0.2).abs() <= 1e-12
    );
    let step = engine.last_step()?.ok_or("missing mixed-boundary step")?;
    assert_eq!(step.boundary_traces.len(), 4);
    assert_eq!(step.transmissive_states.len(), 2);
    assert_eq!(
        step.faces
            .iter()
            .filter(|face| face.signal_speed == 9.0)
            .count(),
        1
    );
    let accepted = engine.snapshot();
    engine.restore_data(&accepted.data()?)?;
    assert_eq!(engine.snapshot().data()?, accepted.data()?);
    Ok(())
}
