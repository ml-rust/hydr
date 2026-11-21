//! Sign classes of a rising residual at both ends of the domain `[2^-30, top]` in `s = y/D`.
//!
//! `f` is the residual enclosure and `kappa` the row slope floor. Over the wet rows the classes
//! partition the rows.
//!
//! | Class | Test | Root in `s` |
//! | --- | --- | --- |
//! | `below` | `f(floor).lower > 0` | below the floor |
//! | `straddle` | `f(floor)` holds a positive value and a nonpositive lower end | at most `floor - f(floor).lower / kappa` |
//! | `solved` | `f(floor) <= 0 <= f(top)` | bracketed |
//! | `near_top` | `f(floor) <= 0`, `f(top)` holds a negative value and a nonnegative upper end | at least `top - f(top).upper / kappa`, if it lies below `top` |
//! | `above` | `f(floor) <= 0`, `f(top).upper < 0` | above `top`, if any |
//!
//! The solved class matches the solver's own bracket test, so every solved row is bracketed.

use crate::structures::numerics::culvert::constants::CRITICAL_FLOOR;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// 0 or 1 masks of each class, with the bounds the slope floor gives near each end.
pub(crate) struct Signs<R: Runtime<DType = DType>> {
    pub(crate) below: Tensor<R>,
    pub(crate) straddle: Tensor<R>,
    pub(crate) solved: Tensor<R>,
    pub(crate) near_top: Tensor<R>,
    pub(crate) above: Tensor<R>,
    /// `floor - f(floor).lower / kappa`, rounded up.
    pub(crate) past: Tensor<R>,
    /// `max(floor, top - f(top).upper / kappa)`, rounded down.
    pub(crate) start: Tensor<R>,
}

/// Classifies the `wet` rows (mask 1) from the residual at the floor and at `top`.
///
/// Rows outside `wet` belong to no class.
pub(crate) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    wet: &Tensor<R>,
    [bottom, crown]: [&Interval<R>; 2],
    top: &Tensor<R>,
    slope: &Tensor<R>,
) -> ArithmeticResult<Signs<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let floor = barrel.constant(CRITICAL_FLOOR)?;
    let c = a.client();
    let zero = &barrel.zero;
    let below = c.mul(wet, &c.gt(bottom.lower(), zero)?)?;
    let rising = c.mul(wet, &c.le(bottom.upper(), zero)?)?;
    let straddle = c.sub(&c.sub(wet, &below)?, &rising)?;
    let above = c.mul(&rising, &c.lt(crown.upper(), zero)?)?;
    let solved = c.mul(&rising, &c.ge(crown.lower(), zero)?)?;
    let near_top = c.sub(&c.sub(&rising, &above)?, &solved)?;

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
    Ok(Signs {
        below,
        straddle,
        solved,
        near_top,
        above,
        past: past.upper().clone(),
        start,
    })
}
