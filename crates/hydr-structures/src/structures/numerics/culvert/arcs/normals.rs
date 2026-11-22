//! Derived radii, corner centres and junction normals of each arc family.
//!
//! The three-radius family has a bottom arc centred at `(0, r_b)`, corner arcs centred at
//! `(±x_c, y_c)` and a top arc centred at `(0, D - r_t)`. The junction normals are
//! `sin psi_1 = x_c / (r_b - r_c)`, `cos psi_1 = (r_b - y_c) / (r_b - r_c)`,
//! `sin psi_2 = x_c / (r_t - r_c)` and `cos psi_2 = (D - r_t - y_c) / (r_t - r_c)`.
//!
//! Each form below keeps its differences between positive terms. The pipe-arch and ellipse
//! radii repeat the admission expressions in the same order, so admission certifies the signs.

use super::inputs::ArcInputs;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// One three-radius section per row. `minus_cos2 = -cos psi_2 >= 0`.
pub(super) struct ThreeRadius<R: Runtime<DType = DType>> {
    pub(super) rise: Interval<R>,
    /// `x_c`, negative for a vertical ellipse.
    pub(super) offset: Interval<R>,
    /// `y_c`, the corner-centre height and the widest depth.
    pub(super) height: Interval<R>,
    pub(super) bottom: Interval<R>,
    pub(super) corner: Interval<R>,
    pub(super) top: Interval<R>,
    pub(super) sin1: Interval<R>,
    pub(super) cos1: Interval<R>,
    pub(super) sin2: Interval<R>,
    pub(super) minus_cos2: Interval<R>,
}

/// One flat-floor arch per row. `cap_cos = -cos psi_0`, the cosine of the cap angle `pi - psi_0`.
pub(super) struct FlatArch<R: Runtime<DType = DType>> {
    pub(super) span: Interval<R>,
    pub(super) rise: Interval<R>,
    pub(super) radius: Interval<R>,
    pub(super) sin0: Interval<R>,
    pub(super) cap_cos: Interval<R>,
}

/// Exact constant `value` on every row of `like`.
fn constant<R>(
    a: &mut Arithmetic<'_, R>,
    like: &Tensor<R>,
    value: f64,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let tensor = Tensor::<R>::full_scalar(like.shape(), DType::F64, value, like.device())?;
    a.point(&tensor)
}

/// A second interval with the endpoints of `x`.
pub(super) fn copy<R>(a: &mut Arithmetic<'_, R>, x: &Interval<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    a.bounds(x.lower(), x.upper())
}

/// The part of `x` at or above zero, for a value known to be nonnegative.
pub(super) fn nonnegative<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = Tensor::<R>::zeros(x.lower().shape(), DType::F64, x.lower().device())?;
    let lower = c.maximum(x.lower(), &zero)?;
    let upper = c.maximum(x.upper(), &zero)?;
    a.bounds(&lower, &upper)
}

/// `(x^2 + y^2) / (2 y)` for `y > 0`.
fn tangency<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    y: &Interval<R>,
    two: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let xx = a.mul(x, x)?;
    let yy = a.mul(y, y)?;
    let sum = a.add(&xx, &yy)?;
    let twice = a.mul(two, y)?;
    a.div_positive_interval(&sum, &twice)
}

/// Pipe-arch from span, rise, corner radius and bottom radius.
fn pipe_arch<R>(
    a: &mut Arithmetic<'_, R>,
    [b, d, rc, rb]: &[Tensor<R>; 4],
) -> ArithmeticResult<ThreeRadius<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let point_half = constant(a, b, 0.5)?;
    let two = constant(a, b, 2.0)?;
    let span = a.point(b)?;
    let rise = a.point(d)?;
    let corner = a.point(rc)?;
    let bottom = a.point(rb)?;
    let half = a.mul(&span, &point_half)?;
    let offset = a.sub(&half, &corner)?;
    let doubled = a.mul(&two, &corner)?;
    let below = a.sub(&bottom, &half)?;
    let wide = a.add(&bottom, &half)?;
    let beside = a.sub(&wide, &doubled)?;
    let lift = a.mul(&below, &beside)?;
    let lift = nonnegative(a, &lift)?;
    let rise_to_centre = a.sqrt_nonnegative(&lift)?;
    let height = a.sub(&bottom, &rise_to_centre)?;
    let above = a.sub(&rise, &height)?;
    let gap = a.sub(&above, &corner)?;
    let reach = a.sub(&half, &above)?;
    let reach = nonnegative(a, &reach)?;
    let spread = tangency(a, &offset, &gap, &two)?;
    let top = a.add(&corner, &spread)?;
    let inner = a.sub(&bottom, &corner)?;
    let sin1 = a.div_positive_interval(&offset, &inner)?;
    let cos1 = a.div_positive_interval(&rise_to_centre, &inner)?;
    let sin2 = a.div_positive_interval(&offset, &spread)?;
    // r_t - (D - y_c) = (B/2 - u) (x_c + g) / (2 g), with u = D - y_c.
    let widened = a.add(&gap, &offset)?;
    let product = a.mul(&reach, &widened)?;
    let twice = a.mul(&two, &gap)?;
    let overhang = a.div_positive_interval(&product, &twice)?;
    let minus_cos2 = a.div_positive_interval(&overhang, &spread)?;
    Ok(ThreeRadius {
        rise,
        offset,
        height,
        bottom,
        corner,
        top,
        sin1,
        cos1,
        sin2,
        minus_cos2,
    })
}

/// Horizontal ellipse from span, rise and side radius.
fn horizontal<R>(
    a: &mut Arithmetic<'_, R>,
    [b, d, rc]: &[Tensor<R>; 3],
) -> ArithmeticResult<ThreeRadius<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let point_half = constant(a, b, 0.5)?;
    let two = constant(a, b, 2.0)?;
    let span = a.point(b)?;
    let rise = a.point(d)?;
    let corner = a.point(rc)?;
    let half = a.mul(&span, &point_half)?;
    let half_rise = a.mul(&rise, &point_half)?;
    let offset = a.sub(&half, &corner)?;
    let lift = a.sub(&half_rise, &corner)?;
    let spread = tangency(a, &offset, &lift, &two)?;
    let radius = a.add(&corner, &spread)?;
    // R - D/2 = (B/2 - D/2) (B/2 + D/2 - 2 r_c) / (2 e), with e = D/2 - r_c.
    let narrow = a.sub(&half, &half_rise)?;
    let sum = a.add(&half, &half_rise)?;
    let doubled = a.mul(&two, &corner)?;
    let wide = a.sub(&sum, &doubled)?;
    let product = a.mul(&narrow, &wide)?;
    let twice = a.mul(&two, &lift)?;
    let rise_to_centre = a.div_positive_interval(&product, &twice)?;
    let sin1 = a.div_positive_interval(&offset, &spread)?;
    let cos1 = a.div_positive_interval(&rise_to_centre, &spread)?;
    Ok(ThreeRadius {
        rise,
        offset,
        height: half_rise,
        bottom: copy(a, &radius)?,
        corner,
        top: radius,
        sin2: copy(a, &sin1)?,
        minus_cos2: copy(a, &cos1)?,
        sin1,
        cos1,
    })
}

/// Vertical ellipse from span, rise and end radius.
fn vertical<R>(
    a: &mut Arithmetic<'_, R>,
    [b, d, r]: &[Tensor<R>; 3],
) -> ArithmeticResult<ThreeRadius<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let point_half = constant(a, b, 0.5)?;
    let two = constant(a, b, 2.0)?;
    let zero = constant(a, b, 0.0)?;
    let span = a.point(b)?;
    let rise = a.point(d)?;
    let end = a.point(r)?;
    let half = a.mul(&span, &point_half)?;
    let half_rise = a.mul(&rise, &point_half)?;
    // The section turned a quarter turn is a horizontal ellipse of side radius `r`.
    let offset_turned = a.sub(&half_rise, &end)?;
    let lift_turned = a.sub(&half, &end)?;
    let spread = tangency(a, &offset_turned, &lift_turned, &two)?;
    let corner = a.add(&end, &spread)?;
    // -x_c = (D/2 - B/2) (D/2 + B/2 - 2 r) / (2 f), with f = B/2 - r.
    let narrow = a.sub(&half_rise, &half)?;
    let sum = a.add(&half_rise, &half)?;
    let doubled = a.mul(&two, &end)?;
    let wide = a.sub(&sum, &doubled)?;
    let product = a.mul(&narrow, &wide)?;
    let twice = a.mul(&two, &lift_turned)?;
    let minus_offset = a.div_positive_interval(&product, &twice)?;
    let offset = a.sub(&zero, &minus_offset)?;
    let sin1 = a.div_positive_interval(&minus_offset, &spread)?;
    let cos1 = a.div_positive_interval(&offset_turned, &spread)?;
    Ok(ThreeRadius {
        rise,
        offset,
        height: half_rise,
        bottom: copy(a, &end)?,
        corner,
        top: end,
        sin2: copy(a, &sin1)?,
        minus_cos2: copy(a, &cos1)?,
        sin1,
        cos1,
    })
}

/// The three-radius section of each row: pipe-arch, horizontal ellipse, else vertical ellipse.
pub(super) fn three_radius<R>(
    a: &mut Arithmetic<'_, R>,
    inputs: &ArcInputs<R>,
) -> ArithmeticResult<ThreeRadius<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let pipe = pipe_arch(a, &inputs.pipe_arch_operands)?;
    let wide = horizontal(a, &inputs.horizontal_operands)?;
    let tall = vertical(a, &inputs.vertical_operands)?;
    let mut pick = |p: &Interval<R>, h: &Interval<R>, v: &Interval<R>| {
        let ellipse = choose(a, &inputs.horizontal, h, v)?;
        choose(a, &inputs.pipe_arch, p, &ellipse)
    };
    Ok(ThreeRadius {
        rise: pick(&pipe.rise, &wide.rise, &tall.rise)?,
        offset: pick(&pipe.offset, &wide.offset, &tall.offset)?,
        height: pick(&pipe.height, &wide.height, &tall.height)?,
        bottom: pick(&pipe.bottom, &wide.bottom, &tall.bottom)?,
        corner: pick(&pipe.corner, &wide.corner, &tall.corner)?,
        top: pick(&pipe.top, &wide.top, &tall.top)?,
        sin1: pick(&pipe.sin1, &wide.sin1, &tall.sin1)?,
        cos1: pick(&pipe.cos1, &wide.cos1, &tall.cos1)?,
        sin2: pick(&pipe.sin2, &wide.sin2, &tall.sin2)?,
        minus_cos2: pick(&pipe.minus_cos2, &wide.minus_cos2, &tall.minus_cos2)?,
    })
}

/// Flat-floor arch from span and rise: `r = (B^2/4 + D^2) / (2 D)`,
/// `sin psi_0 = B D / (B^2/4 + D^2)` and `-cos psi_0 = (B/2 - D) (B/2 + D) / (B^2/4 + D^2)`.
pub(super) fn flat_arch<R>(
    a: &mut Arithmetic<'_, R>,
    [b, d]: &[Tensor<R>; 2],
) -> ArithmeticResult<FlatArch<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let point_half = constant(a, b, 0.5)?;
    let two = constant(a, b, 2.0)?;
    let span = a.point(b)?;
    let rise = a.point(d)?;
    let half = a.mul(&span, &point_half)?;
    let half_square = a.mul(&half, &half)?;
    let rise_square = a.mul(&rise, &rise)?;
    let sum = a.add(&half_square, &rise_square)?;
    let twice = a.mul(&two, &rise)?;
    let radius = a.div_positive_interval(&sum, &twice)?;
    let area = a.mul(&span, &rise)?;
    let sin0 = a.div_positive_interval(&area, &sum)?;
    let narrow = a.sub(&half, &rise)?;
    let wide = a.add(&half, &rise)?;
    let product = a.mul(&narrow, &wide)?;
    let cap_cos = a.div_positive_interval(&product, &sum)?;
    Ok(FlatArch {
        span,
        rise,
        radius,
        sin0,
        cap_cos,
    })
}
