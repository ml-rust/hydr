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
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Per-row classification and the enclosures that need no solver call.
pub(super) struct Ends<R: Runtime<DType = DType>> {
    /// Endpoints of the row target, re-entered by the solver callback.
    pub(super) target: Bounds<R>,
    /// Critical status code per row. Bracketed rows hold `CRITICAL_SOLVED`.
    pub(super) status: Tensor<R>,
    /// 1 on bracketed rows, 0 elsewhere.
    pub(super) solved: Tensor<R>,
    /// 1 on rows with a depth enclosure, 0 on box rows above the crown.
    pub(super) enclosed: Tensor<R>,
    /// Enclosure in `s` of every row the solver does not serve. Bracketed rows hold the domain.
    pub(super) fraction: Bounds<R>,
    /// Solver upper end: `top` on bracketed rows, the floor elsewhere, so other rows stop at once.
    pub(super) solver_top: Tensor<R>,
}

/// Classifies every row from the residual at both domain ends.
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
    let below = c.mul(&wet, &c.gt(bottom.lower(), zero)?)?;
    let rising = c.mul(&wet, &c.le(bottom.upper(), zero)?)?;
    let straddle = c.sub(&c.sub(&wet, &below)?, &rising)?;
    let above = c.mul(&rising, &c.lt(crown.upper(), zero)?)?;
    let solved = c.mul(&rising, &c.ge(crown.lower(), zero)?)?;
    let at_crown = c.sub(&c.sub(&rising, &above)?, &solved)?;

    // f rises by at least kappa per unit s: a root past the floor lies within -f(floor)/kappa of it.
    let deficit = a.point(&c.neg(bottom.lower())?)?;
    let reach = a.div_positive(&deficit, slope)?;
    let floor_point = a.point(&floor)?;
    let past = a.add(&floor_point, &reach)?;
    // A root below the top lies within f(top)/kappa of it.
    let excess = a.point(crown.upper())?;
    let drop = a.div_positive(&excess, slope)?;
    let top_point = a.point(top)?;
    let start = a.sub(&top_point, &drop)?;
    let start = c.maximum(start.lower(), &floor)?;

    let mut lower = zero.clone();
    let mut upper = c.where_cond(&below, &floor, zero)?;
    upper = c.where_cond(&straddle, past.upper(), &upper)?;
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
