use super::admission::{check_inputs, reject_mask};
use super::discharge;
use super::types::{TensorBoundaryRequest, TensorBoundaryTrace};
use crate::numerics::tensor_types::{CpuTensor, TensorFaceState, check_finite, constant};
use crate::numerics::tensor_wave::{bounds, curve, shock_speed};
use crate::solver::{BoundaryFace, EngineError, SolverConfig};
use numr::prelude::{BinaryOps, CompareOps, ConditionalOps, CpuClient, ScalarOps, UnaryOps};

pub(crate) fn characteristic_trace(
    client: &CpuClient,
    interior: &TensorFaceState,
    request: &TensorBoundaryRequest,
    faces: &[BoundaryFace],
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorBoundaryTrace, EngineError> {
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

fn subcritical(
    client: &CpuClient,
    h: &CpuTensor,
    velocity: &CpuTensor,
    faces: &[BoundaryFace],
    gravity: f64,
    relative: f64,
    message: &str,
) -> Result<CpuTensor, EngineError> {
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
