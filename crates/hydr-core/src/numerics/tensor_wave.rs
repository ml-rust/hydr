//! Batched normal Riemann brackets with outward floating-point margins.

use super::tensor_types::{TensorFaceState, any, check_finite, constant};
use crate::contract::{EngineError, SolverConfig};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

#[cfg(test)]
mod fixture;
mod root;

use root::{Rooted, root_upper, rooted_curve};

pub fn bounds<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
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

/// Padded rarefaction-to-dry speeds `(u - 2c, u + 2c)` of each state, zero where `h = 0`.
///
/// `2c = 2 sqrt(g h)` is the rectangular dry-front invariant term `φ(h)`. The padding follows
/// the face bounds: it enlarges each speed by `16 ε` of the local speed scale.
pub fn dry_front_bounds<R, C>(
    client: &C,
    depth: &Tensor<R>,
    velocity: &Tensor<R>,
    gravity: f64,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let zero = constant(depth, 0.0)?;
    let wet = client.gt(depth, &zero)?;
    let sound = client.mul_scalar(
        &client.sqrt(&client.where_cond(&wet, depth, &zero)?)?,
        gravity.sqrt(),
    )?;
    let front = client.mul_scalar(&sound, 2.0)?;
    let lower = client.sub(velocity, &front)?;
    let upper = client.add(velocity, &front)?;
    let scale = client.maximum(
        &client.abs(velocity)?,
        &client.maximum(&client.abs(&lower)?, &client.abs(&upper)?)?,
    )?;
    let padding = client.mul_scalar(&scale, 16.0 * f64::EPSILON)?;
    let lower = client.where_cond(&wet, &client.sub(&lower, &padding)?, &zero)?;
    let upper = client.where_cond(&wet, &client.add(&upper, &padding)?, &zero)?;
    check_finite(client, &lower, "lower dry-front speed")?;
    check_finite(client, &upper, "upper dry-front speed")?;
    Ok((lower, upper))
}

pub fn shock_speed<R, C>(
    client: &C,
    upper: &Tensor<R>,
    h: &Tensor<R>,
    gravity: f64,
) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
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

pub fn curve<R, C>(
    client: &C,
    h: &Tensor<R>,
    reference: &Tensor<R>,
    gravity: f64,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let root = Rooted::new(client, reference)?;
    rooted_curve(client, h, &client.sqrt(h)?, &root, gravity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::tensor_wave::fixture::{
        DEPTHS, GRAVITY, TestResult, physical_upper, setup,
    };

    #[test]
    fn unequal_positive_films_enclose_the_physical_shock_in_both_orientations() -> TestResult {
        let (client, left, right) = setup()?;
        let (sl, sr) = bounds(&client, &left, &right, GRAVITY, &SolverConfig::default())?;
        for ((&(hl, hr), sl), sr) in DEPTHS
            .iter()
            .zip(sl.try_to_vec::<f64>()?)
            .zip(sr.try_to_vec::<f64>()?)
        {
            let deep = hl.max(hr);
            let shallow = hl.min(hr);
            let star = physical_upper(deep, shallow);
            let head = GRAVITY.sqrt() * deep.sqrt();
            let shock =
                GRAVITY.sqrt() * (star / shallow.sqrt()) * (0.5 * (1.0 + shallow / star)).sqrt();
            let (physical_sl, physical_sr) = if hl > hr {
                (-head, shock)
            } else {
                (-shock, head)
            };
            let rounding = 32.0 * f64::EPSILON * shock;
            assert!(sl.is_finite() && sr.is_finite());
            assert!(
                sl <= physical_sl + rounding,
                "sl={sl}, physical_sl={physical_sl}"
            );
            assert!(
                sr >= physical_sr - rounding,
                "sr={sr}, physical_sr={physical_sr}"
            );
            assert!(sl < 0.0 && sr > 0.0);
        }
        Ok(())
    }
}
