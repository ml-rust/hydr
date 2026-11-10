//! Complete reservoir Riemann states, dry fronts, strong shocks, and orientation checks.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "support/exterior_fixture.rs"]
mod fixture;

use fixture::{TestResult, create, interval, through, uniform, wet};
use hydr::model::CellState;
use hydr::solver::{
    AdvanceStatus, Axis, BoundaryFace, BoundaryForcing, ExteriorBoundary, ExteriorState,
    SolverConfig,
};

#[test]
fn complete_uniform_exteriors_preserve_rotation_and_interior_hll_wiring() -> TestResult {
    let mut source = uniform(2, 2, vec![0.0; 4], 1.0, [0.5, 0.2])?;
    let mut rotated = uniform(2, 2, vec![0.0; 4], 1.0, [-0.2, 0.5])?;
    source.advance_to(0.1, 10_000)?;
    rotated.advance_to(0.1, 10_000)?;
    for cell in source.state()?.cells() {
        assert!(
            (cell.h - 1.0).abs() <= 1e-11
                && (cell.qx - 0.5).abs() <= 1e-11
                && (cell.qy - 0.2).abs() <= 1e-11
        );
    }
    for row in 0..2 {
        for column in 0..2 {
            let before = source.state()?.cells()[row * 2 + column];
            let after = rotated.state()?.cells()[column * 2 + (1 - row)];
            assert!(
                (after.h - before.h).abs() <= 1e-11
                    && (after.qx + before.qy).abs() <= 1e-11
                    && (after.qy - before.qx).abs() <= 1e-11
            );
        }
    }
    let step = source.last_step()?.ok_or("missing accepted uniform step")?;
    let interior = step
        .faces
        .iter()
        .find(|face| face.face.axis == Axis::X && face.face.column == 1 && face.face.row == 0)
        .ok_or("missing interior face")?;
    for face in &step.faces {
        let (mass, x, y) = if face.face.axis == Axis::X {
            (0.5, 5.155, 0.1)
        } else {
            (0.2, 0.1, 4.945)
        };
        for (actual, expected) in face.flux.iter().zip([mass, x, y]) {
            assert!((*actual - expected).abs() <= 1e-12);
        }
        if face.face.axis == Axis::X && (face.left.is_none() || face.right.is_none()) {
            for (actual, expected) in face.flux.iter().zip(interior.flux) {
                assert!((*actual - expected).abs() <= 1e-12);
            }
            assert!((face.signal_speed - interior.signal_speed).abs() <= 1e-12);
        }
    }
    assert!(step.boundary_traces.is_empty() && step.transmissive_states.is_empty());
    Ok(())
}

#[test]
fn matching_reservoir_elevation_preserves_a_nonzero_stepped_bed_lake() -> TestResult {
    let bed = vec![10.0, 10.2, 10.4, 10.6];
    let initial: Vec<_> = bed.iter().map(|bed| 11.0 - bed).collect();
    let mut engine = uniform(2, 2, bed, 11.0, [0.0; 2])?;
    assert_eq!(
        engine.advance_to(0.1, 10_000)?.status,
        AdvanceStatus::ReachedTarget
    );
    for (cell, h) in engine.state()?.cells().iter().zip(initial) {
        assert!((cell.h - h).abs() <= 1e-11 && cell.qx.abs() <= 1e-11 && cell.qy.abs() <= 1e-11);
    }
    Ok(())
}

#[test]
fn complete_inflow_and_transmissive_outflow_preserve_four_supercritical_directions() -> TestResult {
    for axis in [Axis::X, Axis::Y] {
        for direction in [-1.0, 1.0] {
            let mut engine = through(axis, direction)?;
            let initial = engine.state()?.cells()[0];
            assert_eq!(
                engine.advance_to(0.1, 10_000)?.status,
                AdvanceStatus::ReachedTarget
            );
            let cell = engine.state()?.cells()[0];
            assert!(
                (cell.h - initial.h).abs() <= 1e-11
                    && (cell.qx - initial.qx).abs() <= 1e-11
                    && (cell.qy - initial.qy).abs() <= 1e-11
            );
            assert!(
                (engine.ledger().boundary_in - 0.5).abs() <= 1e-12
                    && (engine.ledger().boundary_out - 0.5).abs() <= 1e-12
            );
        }
    }
    Ok(())
}

#[test]
fn complete_wet_and_dry_reservoirs_rewet_or_drain_without_prescribing_face_discharge() -> TestResult
{
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    for (h, exterior, inflow) in [
        (0.0, wet(face, 1.0, 0.0, 0.0), true),
        (
            1.0,
            BoundaryForcing::Exterior(ExteriorBoundary {
                face,
                state: ExteriorState::Dry,
            }),
            false,
        ),
    ] {
        let mut engine = create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h,
                qx: 0.0,
                qy: 0.0,
            }],
            vec![interval(0.0, 1.0, 1, vec![exterior])],
            SolverConfig {
                max_step: 0.01,
                ..SolverConfig::default()
            },
        )?;
        engine.advance_to(0.01, 100)?;
        assert!(engine.state()?.cells()[0].h > 0.0);
        let ledger = engine.ledger();
        if inflow {
            assert!(ledger.boundary_in > 0.0 && ledger.storage > 0.0);
        } else {
            assert!(ledger.boundary_out > 0.0 && ledger.storage < 1.0);
        }
        assert!(ledger.residual.abs() <= 1e-9 + 1e-12 * ledger.storage.max(1.0));
        assert_eq!(ledger.water_adjustment, 0.0);
    }
    Ok(())
}

#[test]
fn reservoir_riemann_data_handle_vacuum_and_incoming_shocks_against_supercritical_outflow()
-> TestResult {
    let face = BoundaryFace {
        axis: Axis::X,
        column: 1,
        row: 0,
    };
    for (u, h, exterior_u, shock) in [(-8.0, 1.0, 8.0, false), (5.0, 100.0, 0.0, true)] {
        let mut engine = create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: u,
                qy: 0.0,
            }],
            vec![interval(0.0, 1.0, 1, vec![wet(face, h, exterior_u, 0.0)])],
            SolverConfig {
                max_step: 1e-4,
                ..SolverConfig::default()
            },
        )?;
        engine.advance_to(1e-4, 100)?;
        let step = engine.last_step()?.ok_or("missing reservoir step")?;
        let boundary = step
            .faces
            .iter()
            .find(|transfer| transfer.face == face)
            .ok_or("missing exterior face")?;
        assert!(
            boundary.flux.iter().all(|value| value.is_finite())
                && boundary.signal_speed.is_finite()
        );
        assert!(engine.state()?.cells()[0].h >= 0.0);
        if shock {
            assert!(boundary.flux[0] < 0.0 && engine.ledger().boundary_in > 0.0);
        }
    }
    Ok(())
}

#[test]
fn complete_reservoir_reversal_splits_time_and_rotates_tangent_flux() -> TestResult {
    let make = |axis| -> Result<_, Box<dyn std::error::Error>> {
        let face = BoundaryFace {
            axis,
            column: if axis == Axis::X { 1 } else { 0 },
            row: if axis == Axis::Y { 1 } else { 0 },
        };
        let sign = if axis == Axis::X { 1.0 } else { -1.0 };
        create(
            1,
            1,
            vec![0.0],
            vec![CellState {
                h: 1.0,
                qx: 0.0,
                qy: 0.0,
            }],
            vec![
                interval(0.0, 0.02, 1, vec![wet(face, 1.0, -0.5, sign * 0.3)]),
                interval(0.02, 0.04, 1, vec![wet(face, 1.0, 0.5, sign * (-0.4))]),
            ],
            SolverConfig {
                max_step: 0.015,
                ..SolverConfig::default()
            },
        )
    };
    let mut source = make(Axis::X)?;
    let mut rotated = make(Axis::Y)?;
    let mut split = false;
    source.advance_controlled(
        0.04,
        1000,
        || false,
        |step, _| {
            assert!(!(step.start < 0.02 && step.end > 0.02));
            split |= step.end == 0.02;
        },
    )?;
    rotated.advance_to(0.04, 1000)?;
    assert!(split && source.ledger().boundary_in > 0.0 && source.ledger().boundary_out > 0.0);
    let a = source.state()?.cells()[0];
    let b = rotated.state()?.cells()[0];
    assert!(
        (a.h - b.h).abs() <= 1e-11 && (a.qx - b.qy).abs() <= 1e-11 && (a.qy + b.qx).abs() <= 1e-11
    );
    Ok(())
}
