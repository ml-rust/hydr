use super::admission::{check_inputs, reject_mask};
use super::discharge;
use super::types::{TensorBoundaryRequest, TensorBoundaryTrace};
use crate::contract::{BoundaryFace, EngineError, SolverConfig};
use crate::numerics::tensor_types::{TensorFaceState, check_finite, constant};
use crate::numerics::tensor_wave::{bounds, curve, shock_speed};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

pub(crate) fn characteristic_trace<R, C>(
    client: &C,
    interior: &TensorFaceState<R>,
    request: &TensorBoundaryRequest<R>,
    faces: &[BoundaryFace],
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorBoundaryTrace<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    if !gravity.is_finite() || gravity <= 0.0 {
        return Err(EngineError::rejected(
            "boundary trace gravity",
            format!("gravity={gravity}: supply a finite positive gravity"),
        ));
    }
    check_inputs(client, interior, request, faces)?;
    let zero = constant(&interior.h, 0.0)?;
    let ui = client.div(&interior.q_normal, &interior.h)?;
    let vi = client.div(&interior.q_tangent, &interior.h)?;
    check_finite(client, &ui, "boundary interior velocity")?;
    check_finite(client, &vi, "boundary interior tangent")?;
    let ci = subcritical(
        client,
        &interior.h,
        &ui,
        faces,
        gravity,
        config.boundary.critical_relative,
        "Interior is dry, critical, or supercritical. Supply a supported complete exterior-state boundary",
    )?;
    let discharge_depth =
        discharge::depth(client, &interior.h, &ui, request, faces, gravity, config)?;
    let h = client.where_cond(
        &request.discharge_mask,
        &discharge_depth,
        &request.stage_depth,
    )?;
    check_finite(client, &h, "boundary trace depth")?;
    let curve_velocity = client.sub(&ui, &curve(client, &h, &interior.h, gravity)?)?;
    let curve_discharge = client.mul(&h, &curve_velocity)?;
    let q_normal = client.where_cond(
        &request.discharge_mask,
        &request.outward_discharge,
        &curve_discharge,
    )?;
    // Prescribed discharge defines the conserved state. Momentum uses this same corrected velocity.
    let ub = client.div(&q_normal, &h)?;
    for value in [&curve_velocity, &curve_discharge, &q_normal, &ub] {
        check_finite(client, value, "boundary trace velocity")?;
    }
    let cb = subcritical(
        client,
        &h,
        &ub,
        faces,
        gravity,
        config.boundary.critical_relative,
        "Requested trace is critical or supercritical. Supply an admissible target or complete exterior state",
    )?;
    let discharge_residual = client.where_cond(
        &request.discharge_mask,
        &client.sub(&curve_discharge, &request.outward_discharge)?,
        &zero,
    )?;
    let discharge_scale = client.maximum(
        &client.abs(&curve_discharge)?,
        &client.abs(&request.outward_discharge)?,
    )?;
    let discharge_tolerance = client.add_scalar(
        &client.mul_scalar(&discharge_scale, config.boundary.discharge_relative)?,
        config.boundary.discharge_absolute,
    )?;
    reject_mask(
        client,
        &client.gt(&client.abs(&discharge_residual)?, &discharge_tolerance)?,
        faces,
        "Discharge compatibility exceeds tolerance. Tighten root tolerances or inspect the boundary target",
        &[("discharge residual", &discharge_residual), ("depth", &h)],
    )?;
    let characteristic_residual = client.sub(&ub, &curve_velocity)?;
    let mut scale = client.maximum(&client.abs(&ui)?, &client.abs(&ub)?)?;
    scale = client.maximum(&scale, &client.maximum(&ci, &cb)?)?;
    reject_mask(
        client,
        &client.gt(
            &client.abs(&characteristic_residual)?,
            &client.mul_scalar(&scale, config.boundary.characteristic_relative)?,
        )?,
        faces,
        "Characteristic compatibility exceeds the local velocity tolerance. Tighten root tolerances or increase the depth scale",
        &[
            ("velocity residual", &characteristic_residual),
            ("local speed scale", &scale),
        ],
    )?;
    let shock = client.sub(&ui, &shock_speed(client, &h, &interior.h, gravity)?)?;
    let incoming_wave =
        client.where_cond(&client.gt(&h, &interior.h)?, &shock, &client.sub(&ub, &cb)?)?;
    reject_mask(
        client,
        &client.gt(&incoming_wave, &zero)?,
        faces,
        "Connecting wave leaves the domain. Supply an admissible trace instead of a sonic substitution",
        &[("connecting wave speed", &incoming_wave)],
    )?;
    let inflow = client.lt(&q_normal, &zero)?;
    reject_mask(
        client,
        &client.mul(
            &inflow,
            &client.rsub_scalar(&request.has_inflow_tangent, 1.0)?,
        )?,
        faces,
        "Inflow lacks tangential velocity. Supply the exterior incoming tangential velocity",
        &[("outward discharge", &q_normal)],
    )?;
    let vb = client.where_cond(&inflow, &request.inflow_tangent, &vi)?;
    let q_tangent = client.mul(&h, &vb)?;
    let pressure = client.mul(&client.mul_scalar(&h, 0.5 * gravity)?, &h)?;
    let flux = [
        q_normal.clone(),
        client.add(&client.mul(&q_normal, &ub)?, &pressure)?,
        client.mul(&q_normal, &vb)?,
    ];
    for value in [&q_tangent, &flux[1], &flux[2]] {
        check_finite(client, value, "boundary physical flux")?;
    }
    let trace = TensorFaceState {
        h: h.clone(),
        q_normal: q_normal.clone(),
        q_tangent: q_tangent.clone(),
        bed: interior.bed.clone(),
    };
    let (sl, sr) = bounds(client, interior, &trace, gravity, config)?;
    let speed = client.maximum(
        &client.maximum(&client.abs(&sl)?, &client.abs(&sr)?)?,
        &client.maximum(
            &client.add(&client.abs(&ui)?, &ci)?,
            &client.add(&client.abs(&ub)?, &cb)?,
        )?,
    )?;
    let speed = client.mul_scalar(&speed, 1.0 + 16.0 * f64::EPSILON)?;
    check_finite(client, &speed, "boundary signal speed")?;
    Ok(TensorBoundaryTrace {
        h,
        q_normal,
        tangential_velocity: vb,
        flux,
        speed,
        discharge_residual,
        characteristic_residual,
    })
}

fn subcritical<R, C>(
    client: &C,
    h: &Tensor<R>,
    velocity: &Tensor<R>,
    faces: &[BoundaryFace],
    gravity: f64,
    relative: f64,
    message: &str,
) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    let celerity = client.mul_scalar(&client.sqrt(h)?, gravity.sqrt())?;
    check_finite(client, &celerity, "boundary celerity")?;
    let margin = client.mul_scalar(
        &client.maximum(&client.abs(velocity)?, &celerity)?,
        relative,
    )?;
    let minus = client.sub(velocity, &celerity)?;
    let plus = client.add(velocity, &celerity)?;
    let invalid = client.maximum(
        &client.ge(&minus, &client.neg(&margin)?)?,
        &client.le(&plus, &margin)?,
    )?;
    reject_mask(
        client,
        &invalid,
        faces,
        message,
        &[
            ("depth", h),
            ("normal velocity", velocity),
            ("celerity", &celerity),
        ],
    )?;
    Ok(celerity)
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{G, Lane, TestResult, lane, run, velocity};
    use super::*;
    use numr::runtime::cpu::{CpuClient, ParallelismConfig};

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
}
