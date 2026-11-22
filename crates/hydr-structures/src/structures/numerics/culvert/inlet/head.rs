//! Critical head `Hc = dc + A(dc) / (2 T(dc))` from a critical-depth enclosure.
//!
//! `h(y) = y + A(y) / (2 T(y))` rises in `y`. A box has `h = 3y/2`. Circle and arc sections
//! are convex. Below the widest depth `T` is concave and rising with `T(0) >= 0`, so
//! `T' <= T/y` and `A <= y T`. That gives `d(A/T)/dy = 1 - A T' / T^2 >= 0`. Above the widest
//! depth `T' < 0`. So `Hc` lies in `[h(lower), h(upper)]` of the depth enclosure.
//!
//! Circle and arc rows read `a(s)` and `t(s)` of the unit-rise section at `s = y/D`, so tiny
//! rises never underflow the area. `h(0) = 0`.

use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, BarrelDepths, section};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Critical head per row, with a mask of the rows whose depth enclosure lies below the crown.
pub(super) struct CriticalHead<R: Runtime<DType = DType>> {
    /// `Hc` in metres. Rows outside `valid` hold an exact zero.
    pub(super) head: Interval<R>,
    /// 1 where the depth status is `Dry`, `Solved` or `BelowFloor`, 0 elsewhere.
    pub(super) valid: Tensor<R>,
}

/// Encloses the critical head of every row from the depth enclosure `lower` to `upper`.
///
/// `status` holds the critical-depth status codes. `unit` holds unit-diameter circles of the
/// same width as `barrel`.
pub(super) fn critical_head<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    depths: &BarrelDepths<R>,
) -> ArithmeticResult<CriticalHead<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = &barrel.zero;
    let code = |value: f64| barrel.constant(value);
    let status = &depths.status;
    let valid = c.add(
        &c.add(
            &c.eq(status, &code(codes::CRITICAL_DRY)?)?,
            &c.eq(status, &code(codes::CRITICAL_SOLVED)?)?,
        )?,
        &c.eq(status, &code(codes::CRITICAL_BELOW_FLOOR)?)?,
    )?;
    // Rejected rows take a dry depth, and NaN endpoints of rows without enclosure vanish.
    let lower = c.where_cond(&valid, &depths.lower, zero)?;
    let upper = c.where_cond(&valid, &depths.upper, zero)?;
    let low = a.point(&lower)?;
    let high = a.point(&upper)?;
    let s_low = a.div_positive(&low, &barrel.rise)?.lower().clone();
    let s_high = c.minimum(a.div_positive(&high, &barrel.rise)?.upper(), &barrel.one)?;

    let bottom = unit_head(a, barrel, unit, &s_low)?;
    let top = unit_head(a, barrel, unit, &s_high)?;
    let lower = c.where_cond(&c.gt(&s_low, zero)?, bottom.lower(), zero)?;
    let upper = c.where_cond(&c.gt(&s_high, zero)?, top.upper(), zero)?;
    let unit_bounds = a.bounds(&lower, &upper)?;
    let d = a.point(&barrel.rise)?;
    let head = a.mul(&d, &unit_bounds)?;
    Ok(CriticalHead { head, valid })
}

/// Encloses `h(s) / D` at the depth fractions `s` (`[m]`, `0 <= s < 1`).
///
/// Rows with `s = 0` hold a harmless value. The caller replaces it by zero.
pub(crate) fn unit_head<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    s: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let half = barrel.constant(0.5)?;
    let wet = c.gt(s, &barrel.zero)?;
    let curved = c.sub(&barrel.one, &barrel.boxed)?;
    let curved_rows = c.mul(&wet, &curved)?;
    let fraction = c.where_cond(&curved_rows, s, &half)?;
    let geometry = section::at_depth(a, unit, &fraction)?.geometry;
    let two = a.point(&barrel.constant(2.0)?)?;
    let width = a.mul(&two, &geometry.top_width)?;
    let depth = a.div_positive_interval(&geometry.area, &width)?;
    let point = a.point(&fraction)?;
    let section_head = a.add(&point, &depth)?;
    let box_fraction = a.point(s)?;
    let three_halves = a.point(&barrel.constant(1.5)?)?;
    let rectangle = a.mul(&three_halves, &box_fraction)?;
    super::super::choose(a, &curved, &section_head, &rectangle)
}
