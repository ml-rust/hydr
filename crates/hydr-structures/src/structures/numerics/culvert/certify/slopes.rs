//! Bounds of `d r_c / ds` and `d r_n / ds` over depth pieces of unit-rise arc rows.
//!
//! In unit geometry `s = y`. On an arc with centre height `y_k` and radius `r`, with
//! `h = r sin psi = sqrt((y - (y_k - r)) (y_k + r - y))`:
//!
//! - `dT/ds = 2 cot psi = -2 (y - y_k) / h` and `dP/ds = 2 / sin psi = 2 r / h`
//! - `d r_c / ds = 3 T / A + 2 (y - y_k) / (h T)`
//! - `d r_n / ds = 5 T / A - 4 r / (h P)`
//!
//! Every term is a ratio of positive enclosures, so a piece at the crown, where `T` reaches 0,
//! still gets a finite bound. `A`, `T` and `P` come from range geometry over the whole piece.
//! The arc terms come from the corner arc on the part of the piece at or below the top
//! junction enclosure's upper end, and from the top arc on the part at or above its lower end.
//! Each depth of the piece lies on one of those arcs, so the least lower and greatest upper
//! bound of the parts hold for the piece.

use crate::structures::numerics::culvert::arcs::ArcBand;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, section};
use crate::structures::tensor::culvert::{ArcRows, TensorBarrelContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Per-piece bounds. `known` is 1 where every enclosure the bounds divide by is positive.
pub(super) struct PieceSlopes<R: Runtime<DType = DType>> {
    pub(super) critical_lower: Tensor<R>,
    pub(super) normal_lower: Tensor<R>,
    pub(super) normal_upper: Tensor<R>,
    pub(super) known: Tensor<R>,
}

/// `A`, `T` and `P` over a piece.
struct Flow<'f, R: Runtime<DType = DType>> {
    area: &'f Interval<R>,
    width: &'f Interval<R>,
    perimeter: &'f Interval<R>,
}

/// Lower and upper bounds of one part, with its known mask.
type Part<R> = (Tensor<R>, Tensor<R>, Tensor<R>, Tensor<R>);

/// Encloses both slopes over the pieces `[lower, upper]` of the unit-rise rows `unit`.
///
/// Each piece lies in `[2^-30, 1)`.
pub(super) fn over_pieces<R>(
    a: &mut Arithmetic<'_, R>,
    unit: &TensorBarrelContext<R>,
    rows: &ArcRows<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<PieceSlopes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let geometry = section::over_depth(a, unit, lower, upper)?.geometry;
    let zero = &unit.zero;
    let positive = c.mul(
        &c.gt(geometry.area.lower(), zero)?,
        &c.gt(geometry.top_width.lower(), zero)?,
    )?;
    let positive = c.mul(&positive, &c.gt(geometry.perimeter.lower(), zero)?)?;
    let one = a.point(&unit.one)?;
    let area = guard(a, &positive, &geometry.area, &one)?;
    let width = guard(a, &positive, &geometry.top_width, &one)?;
    let perimeter = guard(a, &positive, &geometry.perimeter, &one)?;
    let flow = Flow {
        area: &area,
        width: &width,
        perimeter: &perimeter,
    };

    let [_, corner, top] = &rows.bands;
    let [_, corner_middle, top_middle] = &rows.middles;
    let [junction_low, junction_high] = &top.start_depth;
    let on_corner = c.le(lower, junction_high)?;
    let corner_top = c.minimum(upper, junction_high)?;
    let corner_part = part(
        a,
        corner,
        &flow,
        &on_corner,
        [
            &c.where_cond(&on_corner, lower, corner_middle)?,
            &c.where_cond(&on_corner, &corner_top, corner_middle)?,
        ],
    )?;
    let on_top = c.ge(upper, junction_low)?;
    let top_bottom = c.maximum(lower, junction_low)?;
    let top_part = part(
        a,
        top,
        &flow,
        &on_top,
        [
            &c.where_cond(&on_top, &top_bottom, top_middle)?,
            &c.where_cond(&on_top, upper, top_middle)?,
        ],
    )?;

    // A part that does not apply holds +inf lower and -inf upper bounds.
    let (c_low, n_low, n_high, c_known) = corner_part;
    let (t_c_low, t_n_low, t_n_high, t_known) = top_part;
    let known = c.mul(&c.mul(&c_known, &t_known)?, &positive)?;
    Ok(PieceSlopes {
        critical_lower: c.minimum(&c_low, &t_c_low)?,
        normal_lower: c.minimum(&n_low, &t_n_low)?,
        normal_upper: c.maximum(&n_high, &t_n_high)?,
        known,
    })
}

/// `x` where `mask` is 1 and `fill` elsewhere.
fn guard<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    x: &Interval<R>,
    fill: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let lower = c.where_cond(mask, x.lower(), fill.lower())?;
    let upper = c.where_cond(mask, x.upper(), fill.upper())?;
    a.bounds(&lower, &upper)
}

/// Both slopes from the arc `band` over the depths `[from, to]` of the rows `applies` marks.
///
/// Rows outside `applies` must carry a depth inside the band's circle.
fn part<R>(
    a: &mut Arithmetic<'_, R>,
    band: &ArcBand<R>,
    flow: &Flow<'_, R>,
    applies: &Tensor<R>,
    [from, to]: [&Tensor<R>; 2],
) -> ArithmeticResult<Part<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let constant =
        |value: f64| Tensor::<R>::full_scalar(from.shape(), DType::F64, value, from.device());
    let zero = constant(0.0)?;
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &[Tensor<R>; 2]| a.bounds(lower, upper);
    let centre = enter(a, &band.centre_y)?;
    let radius = enter(a, &band.radius)?;
    let low = enter(a, &band.circle_low)?;
    let high = enter(a, &band.circle_high)?;
    let y = a.bounds(from, to)?;
    let above = a.sub(&y, &low)?;
    let below = a.sub(&high, &y)?;
    let above = a.bounds(
        &c.maximum(above.lower(), &zero)?,
        &c.maximum(above.upper(), &zero)?,
    )?;
    let below = a.bounds(
        &c.maximum(below.lower(), &zero)?,
        &c.maximum(below.upper(), &zero)?,
    )?;
    let product = a.mul(&above, &below)?;
    let reach = a.sqrt_nonnegative(&product)?;
    let open = c.gt(reach.lower(), &zero)?;
    let unit = constant(1.0)?;
    let one = a.point(&unit)?;
    let reach = guard(a, &open, &reach, &one)?;
    let lift = a.sub(&y, &centre)?;

    let three = a.point(&constant(3.0)?)?;
    let two = a.point(&constant(2.0)?)?;
    let five = a.point(&constant(5.0)?)?;
    let four = a.point(&constant(4.0)?)?;
    let fill = a.div_positive_interval(flow.width, flow.area)?;
    let filled = a.mul(&three, &fill)?;
    let turn = a.mul(&reach, flow.width)?;
    let narrowing = a.div_positive_interval(&lift, &turn)?;
    let narrowing = a.mul(&two, &narrowing)?;
    let critical = a.add(&filled, &narrowing)?;
    let gain = a.mul(&five, &fill)?;
    let wall = a.mul(&reach, flow.perimeter)?;
    let wetting = a.div_positive_interval(&radius, &wall)?;
    let wetting = a.mul(&four, &wetting)?;
    let normal = a.sub(&gain, &wetting)?;

    let inf = constant(f64::INFINITY)?;
    let minus_inf = constant(f64::NEG_INFINITY)?;
    let known = c.maximum(&c.sub(&unit, applies)?, &open)?;
    Ok((
        c.where_cond(applies, critical.lower(), &inf)?,
        c.where_cond(applies, normal.lower(), &inf)?,
        c.where_cond(applies, normal.upper(), &minus_inf)?,
        known,
    ))
}
