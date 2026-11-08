//! Batched normal Riemann brackets with outward floating-point margins.

use super::tensor_types::{CpuTensor, TensorFaceState, any, check_finite, constant};
use crate::solver::{EngineError, SolverConfig};
use numr::prelude::{BinaryOps, CompareOps, ConditionalOps, CpuClient, ScalarOps, UnaryOps};
use solvr::optimize::OptimizeError;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};

pub(super) fn bounds(
    client: &CpuClient,
    left: &TensorFaceState,
    right: &TensorFaceState,
    gravity: f64,
    config: &SolverConfig,
) -> Result<(CpuTensor, CpuTensor), EngineError> {
    let zero = constant(&left.h, 0.0)?;
    let one = constant(&left.h, 1.0)?;
    let lw = client.gt(&left.h, &zero)?;
    let rw = client.gt(&right.h, &zero)?;
    let both = client.mul(&lw, &rw)?;
    let ul = client.div(&left.q_normal, &client.where_cond(&lw, &left.h, &one)?)?;
    let ur = client.div(&right.q_normal, &client.where_cond(&rw, &right.h, &one)?)?;
    let cl = client.mul_scalar(&client.sqrt(&left.h)?, gravity.sqrt())?;
    let cr = client.mul_scalar(&client.sqrt(&right.h)?, gravity.sqrt())?;
    let jump = client.sub(&ur, &ul)?;
    let vacuum = client.ge(&jump, &client.mul_scalar(&client.add(&cl, &cr)?, 2.0)?)?;
    // If f(min(hL,hR)) >= 0, both waves are rarefactions or zero-strength waves.
    // Their outer heads bound the fan without calculating its intermediate depth.
    let rarefaction = client.ge(
        &jump,
        &client.mul_scalar(&client.abs(&client.sub(&cl, &cr)?)?, 2.0)?,
    )?;
    let solve = client.mul(
        &both,
        &client.sub(&one, &client.maximum(&vacuum, &rarefaction)?)?,
    )?;
    for value in [&ul, &ur, &cl, &cr, &jump] {
        check_finite(client, value, "normal wave inputs")?;
    }
    // Inactive entries use benign wet states before evaluating root expressions.
    let hl = client.where_cond(&solve, &left.h, &one)?;
    let hr = client.where_cond(&solve, &right.h, &one)?;
    let du = client.where_cond(&solve, &jump, &zero)?;
    let upper = root_upper(client, &hl, &hr, &du, &solve, gravity, config)?;
    let shock_l = shock_speed(client, &upper, &hl, gravity)?;
    let shock_r = shock_speed(client, &upper, &hr, gravity)?;
    let wet_sl = client.where_cond(&solve, &client.sub(&ul, &shock_l)?, &client.sub(&ul, &cl)?)?;
    let wet_sr = client.where_cond(&solve, &client.add(&ur, &shock_r)?, &client.add(&ur, &cr)?)?;
    let left_dry_sl = client.sub(&ur, &client.mul_scalar(&cr, 2.0)?)?;
    let right_dry_sr = client.add(&ul, &client.mul_scalar(&cl, 2.0)?)?;
    let sl = client.where_cond(&lw, &wet_sl, &left_dry_sl)?;
    let sr = client.where_cond(&rw, &wet_sr, &right_dry_sr)?;
    // Padding enlarges each bound. Its scale follows local speeds, without a unit floor.
    let mut scale = client.maximum(&client.abs(&ul)?, &client.abs(&ur)?)?;
    for value in [&cl, &cr, &sl, &sr] {
        scale = client.maximum(&scale, &client.abs(value)?)?;
    }
    let padding = client.mul_scalar(&scale, 16.0 * f64::EPSILON)?;
    let sl = client.sub(&sl, &padding)?;
    let sr = client.add(&sr, &padding)?;
    check_finite(client, &sl, "left normal wave bound")?;
    check_finite(client, &sr, "right normal wave bound")?;
    if any(client, &client.gt(&sl, &sr)?)? {
        return Err(EngineError::rejected(
            "normal wave bounds",
            "outward bounds are reversed: inspect face states and bracket arithmetic",
        ));
    }
    Ok((sl, sr))
}

fn shock_speed(
    client: &CpuClient,
    upper: &CpuTensor,
    h: &CpuTensor,
    gravity: f64,
) -> Result<CpuTensor, EngineError> {
    // Equivalent to c*sqrt(.5*(H/h)*(1+H/h)), without forming H/h.
    let maximum = client.maximum(upper, h)?;
    let factor = client
        .sqrt(&client.mul_scalar(&client.add_scalar(&client.div(h, &maximum)?, 1.0)?, 0.5)?)?;
    let speed = client.mul(
        &client.mul_scalar(&client.div(&maximum, &client.sqrt(h)?)?, gravity.sqrt())?,
        &factor,
    )?;
    check_finite(client, &speed, "shock wave speed")?;
    Ok(speed)
}

fn curve(
    client: &CpuClient,
    h: &CpuTensor,
    reference: &CpuTensor,
    gravity: f64,
) -> numr::error::Result<CpuTensor> {
    let rarefaction = client.mul_scalar(
        &client.sub(&client.sqrt(h)?, &client.sqrt(reference)?)?,
        2.0 * gravity.sqrt(),
    )?;
    let shock_h = client.maximum(h, reference)?;
    let factor = client.sqrt(&client.mul_scalar(
        &client.add_scalar(&client.div(reference, &shock_h)?, 1.0)?,
        0.5,
    )?)?;
    let shock = client.mul(
        &client.mul_scalar(
            &client.div(&client.sub(&shock_h, reference)?, &client.sqrt(reference)?)?,
            gravity.sqrt(),
        )?,
        &factor,
    )?;
    client.where_cond(&client.le(h, reference)?, &rarefaction, &shock)
}

fn residual(
    client: &CpuClient,
    h: &CpuTensor,
    hl: &CpuTensor,
    hr: &CpuTensor,
    jump: &CpuTensor,
    gravity: f64,
) -> numr::error::Result<CpuTensor> {
    let result = client.add(
        &client.add(
            &curve(client, h, hl, gravity)?,
            &curve(client, h, hr, gravity)?,
        )?,
        jump,
    )?;
    Ok(result)
}

fn root_upper(
    client: &CpuClient,
    hl: &CpuTensor,
    hr: &CpuTensor,
    jump: &CpuTensor,
    solve: &CpuTensor,
    gravity: f64,
    config: &SolverConfig,
) -> Result<CpuTensor, EngineError> {
    let zero = constant(hl, 0.0)?;
    let maximum = client.maximum(hl, hr)?;
    let coefficient_scale = gravity.sqrt() * std::f64::consts::FRAC_1_SQRT_2;
    let coefficient = client.add(
        &client.div(&constant(hl, coefficient_scale)?, &client.sqrt(hl)?)?,
        &client.div(&constant(hr, coefficient_scale)?, &client.sqrt(hr)?)?,
    )?;
    check_finite(client, &coefficient, "compression bracket coefficient")?;
    // For H>=M, fL(H)+fR(H)>=(H-M)*sum(sqrt(g/(2ha))).
    let compression = client.maximum(&client.neg(jump)?, &zero)?;
    let seed = client.add(&maximum, &client.div(&compression, &coefficient)?)?;
    let hi = client.add_scalar(
        &client.mul_scalar(&seed, 1.0 + 16.0 * f64::EPSILON)?,
        f64::from_bits(1),
    )?;
    check_finite(client, &hi, "compression bracket seed")?;
    // Film-depth wave speeds require relative root accuracy beyond an absolute depth ceiling.
    let options = BatchedBisectionOptions {
        atol: f64::from_bits(1),
        rtol: config.root_relative_depth.min(1e-12),
        max_iter: config.root_iterations,
        max_expansions: config.root_iterations,
        expansion_factor: 2.0,
    };
    let result = client
        .bisect_batched(
            |depth| {
                client.where_cond(
                    solve,
                    &residual(client, depth, hl, hr, jump, gravity)?,
                    &zero,
                )
            },
            &zero,
            &hi,
            &options,
        )
        .map_err(|error| {
            let reason = match &error {
                OptimizeError::DidNotConverge { .. } => format!("iteration limit reached: {error}"),
                _ => error.to_string(),
            };
            EngineError::rejected("Riemann root", reason)
        })?;
    Ok(result.upper)
}
