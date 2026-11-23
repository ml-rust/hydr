//! Open-section geometry over depth ranges `[lower, upper]` inside `[0, D')`.
//!
//! - `A`, `P_w`, `P`, `P_n` and `M` increase in `y`. Each range holds both end enclosures whole.
//! - `T` is unimodal on the convex open section, so its least value lies at an end. A range
//!   holding the widest depth reaches the widest width. A bed at or above the widest depth
//!   has `T` falling from `b`.
//! - `b` does not change with depth.
//! - `R` divides the `A` range by the `P` range. `P >= b > 0` at every depth.
//!
//! A range below 0, reversed, NaN or reaching the crown yields inverted bounds, which reject the
//! enclosing call at its completion check.

use super::point::{FloorQuantities, FloorSection, at_depth};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{FloorRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the open-section geometry over every depth in `[lower, upper]` per row.
pub(crate) fn over_depth<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    floor: &FloorRows<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<FloorSection<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let valid = c.mul(
        &c.mul(&c.ge(lower, &barrel.zero)?, &c.le(lower, upper)?)?,
        &c.lt(upper, &floor.top)?,
    )?;
    // Rejected rows evaluate the dry point at both ends, then receive inverted bounds.
    let lower = c.where_cond(&valid, lower, &barrel.zero)?;
    let upper = c.where_cond(&valid, upper, &barrel.zero)?;
    let low = at_depth(a, barrel, floor, &lower)?.geometry;
    let high = at_depth(a, barrel, floor, &upper)?.geometry;
    let ends = Ends {
        low: &low,
        high: &high,
        valid: &valid,
        barrel,
    };

    let area = ends.span(a, |q| &q.area)?;
    let wall = ends.span(a, |q| &q.wall)?;
    let bed = ends.span(a, |q| &q.bed)?;
    let perimeter = ends.span(a, |q| &q.perimeter)?;
    let weighted = ends.span(a, |q| &q.weighted)?;
    let moment = ends.span(a, |q| &q.moment)?;

    let width_floor = c.minimum(low.top_width.lower(), high.top_width.lower())?;
    let width_end = c.maximum(low.top_width.upper(), high.top_width.upper())?;
    let [first, last] = &floor.widest_depth;
    let holds_widest = c.mul(&c.le(&lower, last)?, &c.ge(&upper, first)?)?;
    // The widest width holds every interior point; both end enclosures stay whole.
    let widest = c.maximum(&floor.widest_width, &width_end)?;
    let width_top = c.where_cond(&holds_widest, &widest, &width_end)?;
    let top_width = reject(a, barrel, &valid, &width_floor, &width_top)?;

    // P >= b > 0, so the quotient of the ranges holds R.
    let radius = a.div_positive_interval(&area, &perimeter)?;
    let state = c.where_cond(
        &c.gt(&upper, &barrel.zero)?,
        &barrel.constant(codes::BARREL_FREE)?,
        &barrel.constant(codes::BARREL_DRY)?,
    )?;
    Ok(FloorSection {
        state,
        geometry: FloorQuantities {
            area,
            top_width,
            wall,
            bed,
            perimeter,
            weighted,
            radius,
            moment,
        },
    })
}

/// Both end enclosures of a range with the rows that admit it.
struct Ends<'e, R: Runtime<DType = DType>> {
    low: &'e FloorQuantities<R>,
    high: &'e FloorQuantities<R>,
    valid: &'e Tensor<R>,
    barrel: &'e TensorBarrelContext<R>,
}

impl<R: Runtime<DType = DType>> Ends<'_, R> {
    /// The hull of both end enclosures of `pick`, inverted on rejected rows.
    fn span(
        &self,
        a: &mut Arithmetic<'_, R>,
        pick: fn(&FloorQuantities<R>) -> &Interval<R>,
    ) -> ArithmeticResult<Interval<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let c = a.client();
        let (low, high) = (pick(self.low), pick(self.high));
        let least = c.minimum(low.lower(), high.lower())?;
        let most = c.maximum(low.upper(), high.upper())?;
        reject(a, self.barrel, self.valid, &least, &most)
    }
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
