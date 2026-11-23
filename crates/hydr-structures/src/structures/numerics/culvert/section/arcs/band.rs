//! One band of an arc barrel at a depth, from its walls' chords and circular segments.
//!
//! The band runs from the start normal `a` to the normal `psi` at depth `y`. With
//! `dy = y - y_k^0` and `r sin psi = sqrt((y - (y_k - r)) (y_k + r - y))`:
//!
//! | Quantity | Form |
//! | --- | --- |
//! | `T` | `2 (x_k + r sin psi)` |
//! | `A`, `M` | the stretch from the band start, see `arcs::cap::stretch` |
//! | `P` | `P_k + 2 r t` |
//!
//! Every term of `A` is nonnegative, and the segment meets its series and trigonometric forms,
//! so the band keeps its relative precision near its start. On a flat-floor arch that start is
//! the floor.

use super::super::Quantities;
use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::arcs::cap::{Shared, Stretch, stretch};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// The part of `x` at or above zero, for a value known to be nonnegative.
fn nonnegative<R>(
    a: &mut Arithmetic<'_, R>,
    zero: &Tensor<R>,
    x: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.maximum(x.lower(), zero)?;
    let upper = c.maximum(x.upper(), zero)?;
    a.bounds(&lower, &upper)
}

/// Encloses the band geometry at `depth` for rows whose depth lies at or above the band start.
///
/// `start_width` is the width at the band start. The caller gives every other row a depth
/// inside the band's circle with `r sin psi + r sin a > 0`.
pub(crate) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    shared: &Shared<'_, R>,
    band: &ArcBand<R>,
    start_width: &Bounds<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    let zero = &shared.zero;
    let x = enter(a, &band.centre_x)?;
    let r = enter(a, &band.radius)?;
    let sa = enter(a, &band.start_sin)?;
    let ca = enter(a, &band.start_cos)?;
    let start = enter(a, &band.start_depth)?;
    let start_area = enter(a, &band.start_area)?;
    let start_perimeter = enter(a, &band.start_perimeter)?;
    let start_moment = enter(a, &band.start_moment)?;
    let low = enter(a, &band.circle_low)?;
    let high = enter(a, &band.circle_high)?;
    let start_width = enter(a, start_width)?;

    let y = a.point(depth)?;
    let rise = a.sub(&y, &start)?;
    let dy = nonnegative(a, zero, &rise)?;
    let above = a.sub(&y, &low)?;
    let above = nonnegative(a, zero, &above)?;
    let below = a.sub(&high, &y)?;
    let below = nonnegative(a, zero, &below)?;
    let product = a.mul(&above, &below)?;
    let reach = a.sqrt_nonnegative(&product)?;
    let half_width = a.add(&x, &reach)?;
    let top_width = a.mul(&shared.two, &half_width)?;
    let start_reach = a.mul(&r, &sa)?;
    let lever = a.add(&reach, &start_reach)?;

    let walls = stretch(
        a,
        shared,
        &Stretch {
            radius: &r,
            start_sin: &sa,
            start_cos: &ca,
            lever: &lever,
            rise: &dy,
            start_width: &start_width,
            end_width: &top_width,
            start_area: &start_area,
            start_moment: &start_moment,
        },
    )?;
    let arcs = a.mul(&shared.two, &walls.arc)?;
    let perimeter = a.add(&start_perimeter, &arcs)?;
    let radius = a.div_positive_interval(&walls.area, &perimeter)?;
    Ok(Quantities {
        area: walls.area,
        top_width,
        perimeter,
        radius,
        moment: walls.moment,
    })
}
