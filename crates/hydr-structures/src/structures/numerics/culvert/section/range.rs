//! Geometry over depth ranges `[lower, upper]` inside `[0, D)`, from the point enclosures at both ends.
//!
//! - `A`, `P` and `M` increase in `y`, so each range runs from the lower end's lower bound to the
//!   upper end's upper bound.
//! - `T` is unimodal: its least value lies at an end. A circle range holding `D/2` reaches `D`.
//!   Otherwise `T` is monotone on the range and its largest value lies at an end.
//! - `R` divides the `A` range by the `P` range. A circle range from `y = 0` has `P = 0` at its
//!   lower end. It uses `0 <= R <= y`: below `D/2`, `A <= y T` and `P = D phi >= D sin(phi) = T`.
//!   Above it, `A <= y D` and `P >= pi D / 2`.
//!
//! A range below 0, reversed, NaN or reaching the crown yields inverted bounds, which reject the
//! enclosing call at its completion check. Split an integral at the crown instead.

use super::{Quantities, SectionIntervals, at_depth};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the geometry over every depth in `[lower, upper]` per row.
pub(crate) fn over_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<SectionIntervals<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let valid = c.mul(
        &c.mul(&c.ge(lower, &barrel.zero)?, &c.le(lower, upper)?)?,
        &c.lt(upper, &barrel.rise)?,
    )?;
    // Rejected rows evaluate the dry point at both ends, then receive inverted bounds.
    let lower = c.where_cond(&valid, lower, &barrel.zero)?;
    let upper = c.where_cond(&valid, upper, &barrel.zero)?;
    let low = at_depth(a, barrel, &lower)?.geometry;
    let high = at_depth(a, barrel, &upper)?.geometry;

    let spans = |a: &mut Arithmetic<'_, R>, pick: fn(&Quantities<R>) -> &Interval<R>| {
        let [bottom, top] = [pick(&low).lower(), pick(&high).upper()];
        reject(a, barrel, &valid, bottom, top)
    };
    let area = spans(a, |q| &q.area)?;
    let perimeter = spans(a, |q| &q.perimeter)?;
    let moment = spans(a, |q| &q.moment)?;

    let width_floor = c.minimum(low.top_width.lower(), high.top_width.lower())?;
    let width_end = c.maximum(low.top_width.upper(), high.top_width.upper())?;
    let holds_middle = c.mul(
        &c.mul(
            &c.le(&lower, &barrel.half_rise)?,
            &c.ge(&upper, &barrel.half_rise)?,
        )?,
        &barrel.circular,
    )?;
    let width_top = c.where_cond(&holds_middle, &barrel.rise, &width_end)?;
    let top_width = reject(a, barrel, &valid, &width_floor, &width_top)?;

    let wetted = c.gt(perimeter.lower(), &barrel.zero)?;
    let divisor_lower = c.where_cond(&wetted, perimeter.lower(), &barrel.one)?;
    let divisor_upper = c.where_cond(&wetted, perimeter.upper(), &barrel.one)?;
    let divisor = a.bounds(&divisor_lower, &divisor_upper)?;
    let quotient = a.div_positive_interval(&area, &divisor)?;
    let radius_lower = c.where_cond(&wetted, quotient.lower(), &barrel.zero)?;
    let radius_upper = c.where_cond(&wetted, quotient.upper(), &upper)?;
    let radius = reject(a, barrel, &valid, &radius_lower, &radius_upper)?;

    let state = c.where_cond(
        &c.gt(&upper, &barrel.zero)?,
        &barrel.constant(codes::BARREL_FREE)?,
        &barrel.constant(codes::BARREL_DRY)?,
    )?;
    Ok(SectionIntervals {
        state,
        geometry: Quantities {
            area,
            top_width,
            perimeter,
            radius,
            moment,
        },
    })
}

/// Encloses `[lower, upper]` on valid rows and inverts it on the others.
fn reject<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    valid: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.where_cond(valid, lower, &barrel.one)?;
    let upper = c.where_cond(valid, upper, &barrel.zero)?;
    a.bounds(&lower, &upper)
}
