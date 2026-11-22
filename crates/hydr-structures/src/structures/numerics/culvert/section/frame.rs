//! Point geometry in each row's frame, scaled back to metres.
//!
//! The frame depth is `y 2^-e`, exact unless it falls below the normal range. The frame
//! enclosures times `2^e`, `2^2e` or `2^3e` give the metre enclosures, each bound rounded
//! outward. So a barrel and its power-of-two multiples answer alike, and a metre value outside
//! the F64 range rounds outward instead of failing inside a product.
//!
//! | Frame depth | Metre enclosure |
//! | --- | --- |
//! | exact | the frame enclosure scaled back |
//! | inexact, below `2^-1022` | upper bounds from the next F64 above it, lower bounds the dry values |
//! | overflows | the metre full state |
//!
//! An inexact frame depth lies far below the widest depth, where `A`, `T`, `P`, `R` and `M`
//! rise with depth. The scaled operations commute with every closed form, so a row of scale 1
//! returns the bits of the metre evaluation.

use super::{Quantities, SectionIntervals, direct, full};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{BarrelFrame, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the geometry at the represented depths `depth` (`[m]`, metres) through `frame`.
pub(super) fn at_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    frame: &BarrelFrame<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<SectionIntervals<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let scaled = c.mul(depth, &frame.unscale)?;
    let exact = c.eq(&c.mul(&scaled, &frame.scale)?, depth)?;
    let lifted = c.where_cond(&exact, &scaled, &c.next_up(&scaled)?)?;
    // NaN and overflowing frame depths take the frame crown. The metre checks below decide them.
    let finite = c.lt(&scaled, &barrel.constant(f64::INFINITY)?)?;
    let operand = c.where_cond(&finite, &lifted, &frame.context.rise)?;
    let inner = direct(a, &frame.context, &operand)?;

    let full_rows = c.ge(depth, &barrel.rise)?;
    let metre_full = full::at_depth(a, barrel, &c.where_cond(&full_rows, depth, &barrel.rise)?)?;
    let mut floor = c.mul(&barrel.boxed, &barrel.span)?;
    if let Some(rows) = &barrel.arcs {
        floor = c.add(&floor, &rows.floor_width)?;
    }
    let back = Back {
        frame,
        exact: &exact,
        finite: &finite,
        valid: &c.ge(depth, &barrel.zero)?,
        zero: &barrel.zero,
        one: &barrel.one,
    };
    let g = &inner.geometry;
    let geometry = Quantities {
        area: back.value(a, &g.area, 2, &barrel.zero, &metre_full.area)?,
        top_width: back.value(a, &g.top_width, 1, &floor, &metre_full.top_width)?,
        perimeter: back.value(a, &g.perimeter, 1, &floor, &metre_full.perimeter)?,
        radius: back.value(a, &g.radius, 1, &barrel.zero, &metre_full.radius)?,
        moment: back.value(a, &g.moment, 3, &barrel.zero, &metre_full.moment)?,
    };
    Ok(SectionIntervals {
        state: inner.state,
        geometry,
    })
}

/// Masks and constants of one scale-back.
struct Back<'b, R: Runtime<DType = DType>> {
    frame: &'b BarrelFrame<R>,
    /// 1 where the frame depth is exact.
    exact: &'b Tensor<R>,
    /// 1 where the frame depth is finite.
    finite: &'b Tensor<R>,
    /// 1 where the metre depth is nonnegative.
    valid: &'b Tensor<R>,
    zero: &'b Tensor<R>,
    one: &'b Tensor<R>,
}

impl<R: Runtime<DType = DType>> Back<'_, R> {
    /// `x` times `2^(power e)` per row, `dry` as the lower bound where the frame depth is not
    /// exact, `overflow` where it is not finite, and inverted bounds where the depth is invalid.
    fn value(
        &self,
        a: &mut Arithmetic<'_, R>,
        x: &Interval<R>,
        power: usize,
        dry: &Tensor<R>,
        overflow: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let [lower, upper] = self.frame.times([x.lower(), x.upper()], power)?;
        let c = a.client();
        let lower = c.where_cond(self.exact, &lower, dry)?;
        let lower = c.where_cond(self.finite, &lower, overflow.lower())?;
        let upper = c.where_cond(self.finite, &upper, overflow.upper())?;
        // Inverted bounds on rejected rows fail the completion check.
        let lower = c.where_cond(self.valid, &lower, self.one)?;
        let upper = c.where_cond(self.valid, &upper, self.zero)?;
        a.bounds(&lower, &upper)
    }
}
