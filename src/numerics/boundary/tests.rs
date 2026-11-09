use super::{TensorBoundaryRequest, TensorBoundaryTrace, characteristic_trace};
use crate::cpu::CpuTensor;
use crate::numerics::TensorFaceState;
use crate::solver::{Axis, BoundaryFace, EngineError, SolverConfig};
use numr::ops::BinaryOps;
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

type TestResult = Result<(), Box<dyn std::error::Error>>;
const G: f64 = 9.81;

#[derive(Clone, Copy)]
struct Lane {
    h: f64,
    u: f64,
    v: f64,
    stage: f64,
    discharge: Option<f64>,
    incoming: Option<f64>,
}

fn run(lanes: &[Lane]) -> Result<TensorBoundaryTrace, EngineError> {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let make = |values: Vec<f64>| CpuTensor::from_slice(&values, &[lanes.len()], &device);
    let interior = TensorFaceState {
        h: make(lanes.iter().map(|lane| lane.h).collect())?,
        q_normal: make(lanes.iter().map(|lane| lane.h * lane.u).collect())?,
        q_tangent: make(lanes.iter().map(|lane| lane.h * lane.v).collect())?,
        bed: make(vec![0.0; lanes.len()])?,
    };
    let request = TensorBoundaryRequest {
        stage_depth: make(lanes.iter().map(|lane| lane.stage).collect())?,
        outward_discharge: make(
            lanes
                .iter()
                .map(|lane| lane.discharge.unwrap_or(0.0))
                .collect(),
        )?,
        discharge_mask: make(
            lanes
                .iter()
                .map(|lane| if lane.discharge.is_some() { 1.0 } else { 0.0 })
                .collect(),
        )?,
        inflow_tangent: make(
            lanes
                .iter()
                .map(|lane| lane.incoming.unwrap_or(0.0))
                .collect(),
        )?,
        has_inflow_tangent: make(
            lanes
                .iter()
                .map(|lane| if lane.incoming.is_some() { 1.0 } else { 0.0 })
                .collect(),
        )?,
    };
    let faces: Vec<_> = (0..lanes.len())
        .map(|row| BoundaryFace {
            axis: Axis::X,
            column: 0,
            row,
        })
        .collect();
    characteristic_trace(
        &client,
        &interior,
        &request,
        &faces,
        G,
        &SolverConfig::default(),
    )
}

fn lane() -> Lane {
    Lane {
        h: 1.0,
        u: 0.5,
        v: 0.2,
        stage: 1.0,
        discharge: None,
        incoming: Some(-0.4),
    }
}

fn velocity(interior: Lane, depth: f64) -> f64 {
    let wave = if depth <= interior.h {
        2.0 * ((G * depth).sqrt() - (G * interior.h).sqrt())
    } else {
        (depth - interior.h) * (0.5 * G * (1.0 / depth + 1.0 / interior.h)).sqrt()
    };
    interior.u - wave
}

#[test]
fn uniform_stage_and_discharge_use_the_same_complete_physical_flux() -> TestResult {
    let result = run(&[
        lane(),
        Lane {
            discharge: Some(0.5),
            ..lane()
        },
    ])?;
    for value in result.q_normal.try_to_vec::<f64>()? {
        assert_eq!(value.to_bits(), 0.5_f64.to_bits());
    }
    for value in result.flux[1].try_to_vec::<f64>()? {
        assert!((value - 5.155).abs() <= 1e-11);
    }
    for value in result.flux[2].try_to_vec::<f64>()? {
        assert!((value - 0.1).abs() <= 1e-11);
    }
    Ok(())
}

#[test]
fn still_water_has_zero_mass_flux_and_single_hydrostatic_pressure() -> TestResult {
    let rest = Lane { u: 0.0, ..lane() };
    let result = run(&[
        rest,
        Lane {
            discharge: Some(0.0),
            ..rest
        },
    ])?;
    for value in result.flux[0].try_to_vec::<f64>()? {
        assert_eq!(value, 0.0);
    }
    for value in result.flux[1].try_to_vec::<f64>()? {
        assert!((value - 4.905).abs() <= 1e-11);
    }
    for value in result.flux[2].try_to_vec::<f64>()? {
        assert_eq!(value, 0.0);
    }
    Ok(())
}

#[test]
fn mixed_shock_and_rarefaction_requests_recover_compatible_traces() -> TestResult {
    let initial = lane();
    let mut lanes = Vec::new();
    for depth in [0.8, 1.2] {
        let stage = Lane {
            stage: depth,
            ..initial
        };
        lanes.push(stage);
        lanes.push(Lane {
            discharge: Some(depth * velocity(initial, depth)),
            ..stage
        });
    }
    let result = run(&lanes)?;
    let depths = result.h.try_to_vec::<f64>()?;
    let q = result.q_normal.try_to_vec::<f64>()?;
    let normal = result.flux[1].try_to_vec::<f64>()?;
    let tangent = result.flux[2].try_to_vec::<f64>()?;
    for (index, request) in lanes.iter().enumerate() {
        let u = velocity(initial, request.stage);
        let expected_q = request.stage * u;
        let v = if expected_q < 0.0 { -0.4 } else { initial.v };
        assert!((depths[index] / request.stage - 1.0).abs() <= 1e-11);
        assert!((q[index] - expected_q).abs() <= 1e-11);
        assert!(
            (normal[index] - (expected_q * u + 0.5 * G * request.stage.powi(2))).abs() <= 1e-11
        );
        assert!((tangent[index] - expected_q * v).abs() <= 1e-11);
        if request.stage > initial.h {
            let shock = (q[index] - initial.h * initial.u) / (depths[index] - initial.h);
            let initial_momentum = initial.h * initial.u.powi(2) + 0.5 * G * initial.h.powi(2);
            assert!(shock < 0.0);
            assert!(
                (normal[index] - initial_momentum - shock * (q[index] - initial.h * initial.u))
                    .abs()
                    <= 1e-11
            );
        } else {
            assert!(u - (G * request.stage).sqrt() < 0.0);
        }
        if let Some(target) = request.discharge {
            assert_eq!(q[index].to_bits(), target.to_bits());
        }
    }
    Ok(())
}

#[test]
fn inflow_requires_explicit_tangent_while_outflow_carries_the_interior_tangent() -> TestResult {
    assert!(matches!(
        run(&[Lane {
            stage: 1.2,
            incoming: None,
            ..lane()
        }]),
        Err(EngineError::Boundary { .. })
    ));
    let result = run(&[Lane {
        incoming: None,
        ..lane()
    }])?;
    assert!((result.flux[2].try_to_vec::<f64>()?[0] - 0.1).abs() <= 1e-11);
    Ok(())
}

#[test]
fn thin_wet_traces_control_characteristic_error_without_a_unit_floor() -> TestResult {
    let mut lanes = Vec::new();
    for h in [1e-20, 1e-100] {
        let initial = Lane {
            h,
            u: 0.1 * (G * h).sqrt(),
            v: 0.0,
            stage: 0.8 * h,
            discharge: None,
            incoming: Some(0.0),
        };
        lanes.push(initial);
        lanes.push(Lane {
            discharge: Some(initial.stage * velocity(initial, initial.stage)),
            ..initial
        });
    }
    let result = run(&lanes)?;
    let depths = result.h.try_to_vec::<f64>()?;
    let residuals = result.characteristic_residual.try_to_vec::<f64>()?;
    let q = result.q_normal.try_to_vec::<f64>()?;
    for (index, request) in lanes.iter().enumerate() {
        assert!((depths[index] / request.stage - 1.0).abs() <= 1e-11);
        assert!(residuals[index].abs() / (G * request.h).sqrt() <= 1e-12);
        if let Some(target) = request.discharge {
            assert_eq!(q[index].to_bits(), target.to_bits());
        }
    }
    Ok(())
}

#[test]
fn dry_critical_supercritical_and_inadmissible_targets_reject() {
    for request in [
        Lane {
            h: 0.0,
            u: 0.0,
            ..lane()
        },
        Lane {
            u: G.sqrt(),
            ..lane()
        },
        Lane { u: 4.0, ..lane() },
        Lane { u: -4.0, ..lane() },
        Lane {
            stage: 0.0,
            ..lane()
        },
        Lane {
            stage: 0.01,
            ..lane()
        },
        Lane {
            stage: 10.0,
            ..lane()
        },
        Lane {
            discharge: Some(100.0),
            ..lane()
        },
    ] {
        assert!(
            matches!(run(&[request]), Err(EngineError::Boundary { .. })),
            "h={}, u={}, stage={}, q={:?}",
            request.h,
            request.u,
            request.stage,
            request.discharge
        );
    }
}

#[test]
fn sonic_discharge_limit_rejects_but_a_resolved_subcritical_rate_is_admitted() -> TestResult {
    let rest = Lane { u: 0.0, ..lane() };
    let sonic_depth = 4.0 / 9.0;
    let maximum = sonic_depth * (G * sonic_depth).sqrt();
    for target in [maximum, maximum * (1.0 + 1e-6)] {
        assert!(
            matches!(
                run(&[Lane {
                    discharge: Some(target),
                    ..rest
                }]),
                Err(EngineError::Boundary { .. })
            ),
            "target={target:.17e}"
        );
    }
    let chosen_depth = sonic_depth * (1.0 + 1e-6);
    let target = chosen_depth * velocity(rest, chosen_depth);
    let result = run(&[Lane {
        discharge: Some(target),
        ..rest
    }])?;
    let depth = result.h.try_to_vec::<f64>()?[0];
    let q = result.q_normal.try_to_vec::<f64>()?[0];
    let u = q / depth;
    let c = (G * depth).sqrt();
    assert!(depth > sonic_depth);
    assert!(u - c < 0.0 && u + c > 0.0);
    assert_eq!(q.to_bits(), target.to_bits());
    // Rate inversion is ill-conditioned at the sonic maximum. Check the characteristic relation directly.
    assert!((u - velocity(rest, depth)).abs() / G.sqrt() <= 1e-12);
    Ok(())
}

#[test]
fn a_bad_second_lane_identifies_its_face_in_mixed_boundary_batches() {
    let expected = BoundaryFace {
        axis: Axis::X,
        column: 0,
        row: 1,
    };
    for bad in [
        Lane {
            discharge: Some(100.0),
            ..lane()
        },
        Lane {
            stage: 0.01,
            ..lane()
        },
    ] {
        assert!(
            matches!(run(&[lane(), bad]), Err(EngineError::Boundary { face, .. }) if face == expected)
        );
    }
}

#[test]
fn incoming_tangent_remains_available_when_its_discharge_underflows() -> TestResult {
    let incoming = 1e-200;
    let result = run(&[Lane {
        h: 1e-200,
        u: 0.0,
        stage: 1.2e-200,
        incoming: Some(incoming),
        ..lane()
    }])?;
    let normal = result.q_normal.try_to_vec::<f64>()?[0];
    assert!(normal.is_finite() && normal < 0.0);
    let client = CpuClient::new(result.h.device().clone())
        .with_parallelism(ParallelismConfig::new(Some(1), None));
    assert_eq!(
        client
            .mul(&result.h, &result.tangential_velocity)?
            .item::<f64>()?,
        0.0
    );
    assert_eq!(
        result.tangential_velocity.try_to_vec::<f64>()?[0].to_bits(),
        incoming.to_bits()
    );
    Ok(())
}
