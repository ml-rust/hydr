//! Critical head `Hc = dc + A(dc) / (2 T(dc))` from a critical-depth enclosure.
//!
//! `h(y) = y + A(y) / (2 T(y))` rises in `y`. A box has `h = 3y/2`. Circle and arc sections
//! are convex. Below the widest depth `T` is concave and rising with `T(0) >= 0`, so
//! `T' <= T/y` and `A <= y T`. That gives `d(A/T)/dy = 1 - A T' / T^2 >= 0`. Above the widest
//! depth `T' < 0`. So `Hc` lies in `[h(lower), h(upper)]` of the depth enclosure.
//!
//! Circle and arc rows read `a(s)` and `t(s)` of the unit-rise section at `s = y/D`, so tiny
//! rises never underflow the area. `h(0) = 0`.
//!
//! Above a floor `s = y/D'` and `h` is the head of the open section. The open section is convex,
//! and below its widest depth `T` is concave and rising with `T(0) = b >= 0`. So `A <= y T` and
//! `T' <= T/y` again, and `h` rises in `y`. A floored circle or arc row reads its frame
//! geometry: `h / D' = s + A_f / (2 T_f D'_f)`. A floored box is the box on `D'`. Below the
//! domain floor `s = 2^-30` the head lies in `[y, h(2^-30 D')]`, because `A / (2 T) >= 0` and
//! `h` rises.

use crate::structures::numerics::culvert::constants::{CRITICAL_FLOOR, FLOOR_CRITICAL_TOP};
use crate::structures::numerics::culvert::{
    Arithmetic, BarrelClient, BarrelDepths, floor, section,
};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{FloorRows, TensorBarrelContext};
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
    let rise = barrel.open_rise();
    let s_low = a.div_positive(&low, rise)?.lower().clone();
    let s_high = c.minimum(a.div_positive(&high, rise)?.upper(), &barrel.one)?;

    let bottom = unit_head(a, barrel, unit, &s_low)?;
    let top = unit_head(a, barrel, unit, &s_high)?;
    let mut lower = c.where_cond(&c.gt(&s_low, zero)?, bottom.lower(), zero)?;
    let mut upper = c.where_cond(&c.gt(&s_high, zero)?, top.upper(), zero)?;
    if let Some(rows) = barrel.floor.as_deref() {
        let open = c.sub(&rows.floored, &rows.boxed)?;
        let [open_lower, open_upper] = open_heads(a, barrel, rows, [&s_low, &s_high])?;
        lower = c.where_cond(&open, &open_lower, &lower)?;
        upper = c.where_cond(&open, &open_upper, &upper)?;
    }
    let unit_bounds = a.bounds(&lower, &upper)?;
    let d = a.point(rise)?;
    let head = a.mul(&d, &unit_bounds)?;
    Ok(CriticalHead { head, valid })
}

/// Lower bound of `h / D'` at `low` and upper bound at `high` on the open sections of `rows`,
/// with `0 <= low <= high <= 1`. Other rows hold harmless values.
fn open_heads<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    rows: &FloorRows<R>,
    [low, high]: [&Tensor<R>; 2],
) -> ArithmeticResult<[Tensor<R>; 2]>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = &barrel.zero;
    let floor_value = barrel.constant(CRITICAL_FLOOR)?;
    // A root never passes the domain top, so a rounded fraction above it stands for the top.
    let top = barrel.constant(FLOOR_CRITICAL_TOP)?;
    let clamp = |s: &Tensor<R>| c.minimum(&c.maximum(s, &floor_value)?, &top);
    let bottom = open_head(a, barrel, rows, &clamp(low)?)?;
    let crest = open_head(a, barrel, rows, &clamp(high)?)?;
    // Below the domain floor the head is at least the depth.
    let shallow = c.lt(low, &floor_value)?;
    let lower = c.where_cond(&shallow, low, bottom.lower())?;
    let upper = c.where_cond(&c.gt(high, zero)?, crest.upper(), zero)?;
    Ok([lower, upper])
}

/// Encloses `h(s) / D' = s + A_f / (2 T_f D'_f)` at the fractions `s` in `[2^-30, 1 - 2^-52]`
/// of the open sections, from frame geometry.
fn open_head<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    rows: &FloorRows<R>,
    s: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let framed = barrel.frame.as_ref().map_or(barrel, |frame| &frame.context);
    let Some(open) = framed.floor.as_deref() else {
        return Err(numr::error::Error::InvalidArgument {
            arg: "barrel",
            reason: "the barrel frame lost its floor rows".into(),
        }
        .into());
    };
    let geometry = floor::at_fraction(a, barrel, rows, s)?.geometry;
    let two = a.point(&barrel.constant(2.0)?)?;
    let width = a.mul(&two, &geometry.top_width)?;
    let depth = a.div_positive_interval(&geometry.area, &width)?;
    let scaled = a.div_positive(&depth, &open.top)?;
    let point = a.point(s)?;
    a.add(&point, &scaled)
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
