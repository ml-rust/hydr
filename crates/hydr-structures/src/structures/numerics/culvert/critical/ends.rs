//! End classification of the critical domain `[2^-30, top]` in `s = y/D`, before any solver call.
//!
//! `f` is the residual enclosure and `kappa` the row slope floor. The classes partition the rows.
//!
//! | Rows | Status | Enclosure in `s` |
//! | --- | --- | --- |
//! | `Q = 0` | `Dry` | `[0, 0]` |
//! | `f(floor).lower > 0` | `BelowFloor` | `[0, floor]` |
//! | `f(floor)` holds a positive value and a nonpositive lower end | `BelowFloor` | `[0, floor - f(floor).lower / kappa]` |
//! | `f(floor) <= 0 <= f(top)` | `Solved` | from the solver |
//! | `f(floor) <= 0`, `f(top)` holds a negative value and a nonnegative upper end | `AtCrown` | `[top - f(top).upper / kappa, 1]` |
//! | `f(top).upper < 0` | `AboveCrown` | circle `[top, 1]`, box none |
//!
//! The solved class matches the solver's own bracket test, so every solved row is bracketed.
//! A circle root always lies in `(0, 1)`. Above `top` it lies within `2^-53 D` of the crown.
//! A box root exists only for `Q <= sqrt(g) B D^(3/2)`. An `AtCrown` box row holds the root
//! when one exists.

use super::residual::{residual, target};
use crate::structures::numerics::culvert::constants::CRITICAL_FLOOR;
use crate::structures::numerics::culvert::roots::{self, Ends, Signs};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Classifies every row from the residual at both domain ends.
///
/// Every row has an enclosure except box rows above the crown.
pub(super) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    gravity: &Tensor<R>,
) -> ArithmeticResult<Ends<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let top = &barrel.critical_top;
    let slope = &barrel.critical_slope;
    let row_target = target(a, barrel, discharge, gravity)?;
    let bottom = residual(a, barrel, unit, &row_target, &floor)?;
    let crown = residual(a, barrel, unit, &row_target, top)?;
    let c = a.client();
    let one = &barrel.one;
    let zero = &barrel.zero;

    let wet = c.gt(discharge, zero)?;
    let Signs {
        below,
        straddle,
        solved,
        near_top: at_crown,
        above,
        past,
        start,
    } = roots::classify(a, barrel, &wet, [&bottom, &crown], top, slope)?;

    let mut lower = zero.clone();
    let mut upper = c.where_cond(&below, &floor, zero)?;
    upper = c.where_cond(&straddle, &past, &upper)?;
    lower = c.where_cond(&at_crown, &start, &lower)?;
    upper = c.where_cond(&at_crown, one, &upper)?;
    lower = c.where_cond(&above, top, &lower)?;
    upper = c.where_cond(&above, one, &upper)?;
    lower = c.where_cond(&solved, &floor, &lower)?;
    upper = c.where_cond(&solved, top, &upper)?;

    let code = |value: f64| barrel.constant(value);
    let mut status = code(codes::CRITICAL_DRY)?;
    status = c.where_cond(&below, &code(codes::CRITICAL_BELOW_FLOOR)?, &status)?;
    status = c.where_cond(&straddle, &code(codes::CRITICAL_BELOW_FLOOR)?, &status)?;
    status = c.where_cond(&at_crown, &code(codes::CRITICAL_AT_CROWN)?, &status)?;
    status = c.where_cond(&above, &code(codes::CRITICAL_ABOVE_CROWN)?, &status)?;
    status = c.where_cond(&solved, &code(codes::CRITICAL_SOLVED)?, &status)?;

    Ok(Ends {
        target: [row_target.lower().clone(), row_target.upper().clone()],
        status,
        enclosed: c.sub(one, &c.mul(&above, &barrel.boxed)?)?,
        solver_top: c.where_cond(&solved, top, &floor)?,
        solved,
        fraction: [lower, upper],
    })
}
