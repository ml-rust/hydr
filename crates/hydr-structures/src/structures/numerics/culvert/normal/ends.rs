//! End classification of the normal domain `[2^-30, top]` in `s = y/D`, before any solver call.
//!
//! `top` is `s_N = 0.93818` for a circle and the crown `1` for a box. `f` is the residual
//! enclosure, `kappa` the row slope floor and `s_K` the circle conveyance peak.
//!
//! | Rows | Status | Enclosure in `s` |
//! | --- | --- | --- |
//! | `S <= 0` | `NonPositiveSlope` | none |
//! | `Q = 0` | `Dry` | `[0, 0]` |
//! | `f(floor).lower > 0` | `BelowFloor` | `[0, floor]` |
//! | `f(floor)` holds a positive value and a nonpositive lower end | `BelowFloor` | `[0, floor - f(floor).lower / kappa]` |
//! | `f(floor) <= 0 <= f(top)` | `Solved` | from the solver |
//! | `f(floor) <= 0`, `f(top)` holds a negative value and a nonnegative upper end | `NearCapacity` | circle `[top - f(top).upper / kappa, s_K]`, box `[top - f(top).upper / kappa, 1]` |
//! | `f(top).upper < 0`, circle target not above `3 ln K_max` | `NearCapacity` | `[s_N, s_K]` |
//! | `f(top).upper < 0`, box, or circle target above `3 ln K_max` | `AboveCapacity` | none |
//!
//! A `NearCapacity` row holds its rising-limb root in the enclosure when the root exists.
//! A circle has one up to the peak `K_max`. A box has one up to the crown limit `K(D^-)`.

use super::residual::{residual, target};
use crate::structures::numerics::culvert::constants::{
    CONVEYANCE_PEAK, CRITICAL_FLOOR, LOG_PEAK_CONVEYANCE,
};
use crate::structures::numerics::culvert::roots::{self, Ends, Signs};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Classifies every row from the residual at both domain ends and the circle peak.
pub(super) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    roughness: &Tensor<R>,
    slope: &Tensor<R>,
) -> ArithmeticResult<Ends<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let top = &barrel.normal_top;
    let row_target = target(a, barrel, discharge, roughness, slope)?;
    let bottom = residual(a, barrel, unit, &row_target, &floor)?;
    let crown = residual(a, barrel, unit, &row_target, top)?;
    let c = a.client();
    let one = &barrel.one;
    let zero = &barrel.zero;

    let sloped = c.gt(slope, zero)?;
    let flat = c.sub(one, &sloped)?;
    let wet = c.mul(&sloped, &c.gt(discharge, zero)?)?;
    let Signs {
        below,
        straddle,
        solved,
        near_top,
        above,
        past,
        start,
    } = roots::classify(
        a,
        barrel,
        &wet,
        [&bottom, &crown],
        top,
        &barrel.normal_slope,
    )?;

    // 3 ln k_max - target encloses the circle residual at the conveyance peak.
    let [log_lower, log_upper] =
        LOG_PEAK_CONVEYANCE.map(|bits| barrel.constant(f64::from_bits(bits)));
    let log_peak = a.bounds(&log_lower?, &log_upper?)?;
    let three = a.point(&barrel.constant(3.0)?)?;
    let thrice_peak = a.mul(&three, &log_peak)?;
    let at_peak = a.sub(&thrice_peak, &row_target)?;
    let beyond = c.lt(at_peak.upper(), zero)?;
    let over = c.maximum(&barrel.boxed, &beyond)?;
    let above_capacity = c.mul(&above, &over)?;
    let below_peak = c.sub(&above, &above_capacity)?;
    let near_capacity = c.add(&near_top, &below_peak)?;

    // A rising-limb root above s_N lies below the peak. A box root lies at or below the crown.
    let peak = barrel.constant(f64::from_bits(CONVEYANCE_PEAK[1]))?;
    let capacity_top = c.where_cond(&barrel.circular, &peak, one)?;

    let mut lower = zero.clone();
    let mut upper = c.where_cond(&below, &floor, zero)?;
    upper = c.where_cond(&straddle, &past, &upper)?;
    lower = c.where_cond(&near_top, &start, &lower)?;
    upper = c.where_cond(&near_capacity, &capacity_top, &upper)?;
    lower = c.where_cond(&below_peak, top, &lower)?;
    lower = c.where_cond(&solved, &floor, &lower)?;
    upper = c.where_cond(&solved, top, &upper)?;

    let code = |value: f64| barrel.constant(value);
    let below_floor = code(codes::NORMAL_BELOW_FLOOR)?;
    let mut status = code(codes::NORMAL_DRY)?;
    status = c.where_cond(&below, &below_floor, &status)?;
    status = c.where_cond(&straddle, &below_floor, &status)?;
    status = c.where_cond(&near_capacity, &code(codes::NORMAL_NEAR_CAPACITY)?, &status)?;
    status = c.where_cond(
        &above_capacity,
        &code(codes::NORMAL_ABOVE_CAPACITY)?,
        &status,
    )?;
    status = c.where_cond(&solved, &code(codes::NORMAL_SOLVED)?, &status)?;
    status = c.where_cond(&flat, &code(codes::NORMAL_NON_POSITIVE_SLOPE)?, &status)?;

    Ok(Ends {
        target: [row_target.lower().clone(), row_target.upper().clone()],
        status,
        enclosed: c.sub(&c.sub(one, &above_capacity)?, &flat)?,
        solver_top: c.where_cond(&solved, top, &floor)?,
        solved,
        fraction: [lower, upper],
    })
}
