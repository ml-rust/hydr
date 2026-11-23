//! Open-section geometry in each row's frame, scaled back to metres.
//!
//! The rules of the barrel frame hold: the frame depth is `y 2^-e`, the frame enclosures times
//! `2^e`, `2^2e` or `2^3e` give the metre enclosures, an inexact frame depth below `2^-1022`
//! takes the dry values as lower bounds, and an overflowing one takes the metre full state.
//! A row of scale 1 returns the bits of the metre evaluation.

use super::point::{Core, core as direct, full_state};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::{BarrelFrame, Bounds, FloorRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// The open-section core of `floor` at metre depths `depth`, evaluated in `frame`.
pub(super) fn core<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    frame: &BarrelFrame<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<(Tensor<R>, Core<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(inner) = &frame.context.floor else {
        return Err(numr::error::Error::InvalidArgument {
            arg: "frame",
            reason: "the barrel frame lost its floor rows".into(),
        }
        .into());
    };
    let c = a.client();
    let scaled = c.mul(depth, &frame.unscale)?;
    let exact = c.eq(&c.mul(&scaled, &frame.scale)?, depth)?;
    let lifted = c.where_cond(&exact, &scaled, &c.next_up(&scaled)?)?;
    // NaN and overflowing frame depths take the frame crown. The metre checks below decide them.
    let finite = c.lt(&scaled, &barrel.constant(f64::INFINITY)?)?;
    let operand = c.where_cond(&finite, &lifted, &inner.top)?;
    let (state, framed) = direct(a, &frame.context, inner, &operand)?;

    let full_rows = c.ge(depth, &floor.top)?;
    let metre_full = full_state(a, floor, &c.where_cond(&full_rows, depth, &floor.top)?)?;
    let back = Back {
        frame,
        exact: &exact,
        finite: &finite,
        valid: &c.ge(depth, &barrel.zero)?,
        zero: &barrel.zero,
        one: &barrel.one,
    };
    let nothing = [barrel.zero.clone(), barrel.zero.clone()];
    let geometry = Core {
        area: back.value(a, &framed.area, 2, &nothing, &metre_full.area)?,
        top_width: back.value(
            a,
            &framed.top_width,
            1,
            &floor.bed_width,
            &metre_full.top_width,
        )?,
        wall: back.value(a, &framed.wall, 1, &nothing, &metre_full.wall)?,
        moment: back.value(a, &framed.moment, 3, &nothing, &metre_full.moment)?,
    };
    Ok((state, geometry))
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
    /// `x` times `2^(power e)` per row, joined with the bed value `dry` where the frame depth is
    /// not exact, `overflow` where it is not finite, and inverted bounds where the depth is
    /// invalid.
    ///
    /// The frame evaluates an inexact depth at the next F64 above it. Each quantity is monotone
    /// between the bed and that depth, so the hull of both ends holds it, whichever way `T` runs.
    fn value(
        &self,
        a: &mut Arithmetic<'_, R>,
        x: &Interval<R>,
        power: usize,
        [dry_lower, dry_upper]: &Bounds<R>,
        overflow: &Interval<R>,
    ) -> ArithmeticResult<Interval<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let [lower, upper] = self.frame.times([x.lower(), x.upper()], power)?;
        let c = a.client();
        let joined_lower = c.minimum(&lower, dry_lower)?;
        let joined_upper = c.maximum(&upper, dry_upper)?;
        let lower = c.where_cond(self.exact, &lower, &joined_lower)?;
        let upper = c.where_cond(self.exact, &upper, &joined_upper)?;
        let lower = c.where_cond(self.finite, &lower, overflow.lower())?;
        let upper = c.where_cond(self.finite, &upper, overflow.upper())?;
        // Inverted bounds on rejected rows fail the completion check.
        let lower = c.where_cond(self.valid, &lower, self.one)?;
        let upper = c.where_cond(self.valid, &upper, self.zero)?;
        a.bounds(&lower, &upper)
    }
}
