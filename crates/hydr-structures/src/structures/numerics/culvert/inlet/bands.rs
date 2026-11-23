//! Band membership of each discharge and the headwater hull over the bands it may lie in.
//!
//! The limit discharges `3.5 scale` and `4 scale` are enclosures, so a discharge near a limit can
//! lie in two bands. Its headwater then takes the hull of both band values. The exact headwater
//! equals one of them, so the hull holds it.

use super::head::critical_head;
use super::power::power;
use crate::structures::numerics::culvert::constants::{SUBMERGED_LIMIT, UNSUBMERGED_LIMIT};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, BarrelDepths, choose};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorInletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// 0 or 1 masks of the bands each row may lie in. Every row lies in at least one.
pub(crate) struct Bands<R: Runtime<DType = DType>> {
    /// `X <= 3.5` is possible.
    pub(crate) unsubmerged: Tensor<R>,
    /// `3.5 < X < 4` is possible.
    pub(crate) transition: Tensor<R>,
    /// `X >= 4` is possible.
    pub(crate) submerged: Tensor<R>,
}

/// Places each discharge enclosure `[lower, upper]` against the limit-discharge enclosures. No
/// interval call is needed.
pub(crate) fn classify<R>(
    inlet: &TensorInletContext<R>,
    barrel: &TensorBarrelContext<R>,
    [lower, upper]: &Bounds<R>,
) -> numr::error::Result<Bands<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = &barrel.client;
    let [low_lower, low_upper] = &inlet.unsubmerged_discharge;
    let [high_lower, high_upper] = &inlet.submerged_discharge;
    Ok(Bands {
        unsubmerged: c.le(lower, low_upper)?,
        transition: c.mul(&c.gt(upper, low_lower)?, &c.lt(lower, high_upper)?)?,
        submerged: c.ge(upper, high_lower)?,
    })
}

/// Status code per row: one band, or a limit when two bands are possible.
pub(crate) fn status<R>(
    barrel: &TensorBarrelContext<R>,
    bands: &Bands<R>,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = &barrel.client;
    let code = |value: f64| barrel.constant(value);
    let low_limit = c.mul(&bands.unsubmerged, &bands.transition)?;
    let high_limit = c.mul(&bands.transition, &bands.submerged)?;
    let mut status = code(codes::INLET_TRANSITION)?;
    status = c.where_cond(
        &bands.unsubmerged,
        &code(codes::INLET_UNSUBMERGED)?,
        &status,
    )?;
    status = c.where_cond(&bands.submerged, &code(codes::INLET_SUBMERGED)?, &status)?;
    status = c.where_cond(&low_limit, &code(codes::INLET_UNSUBMERGED_LIMIT)?, &status)?;
    c.where_cond(&high_limit, &code(codes::INLET_SUBMERGED_LIMIT)?, &status)
}

/// Inputs of one headwater evaluation that the caller prepared outside the interval call.
pub(crate) struct Operands<'o, R: Runtime<DType = DType>> {
    pub(crate) barrel: &'o TensorBarrelContext<R>,
    pub(crate) unit: &'o TensorBarrelContext<R>,
    pub(crate) inlet: &'o TensorInletContext<R>,
    /// Enclosure of the unblocked discharge.
    pub(crate) discharge: &'o Bounds<R>,
    pub(crate) bands: &'o Bands<R>,
    /// Critical depth over `discharge` on form-1 rows that may be unsubmerged, dry elsewhere.
    pub(crate) critical: &'o BarrelDepths<R>,
}

/// Encloses the inlet-control headwater of every row, in metres above the inlet invert, or
/// above the bed of a barrel with a floor.
///
/// Each band runs on every row. Rows outside a band evaluate it at a harmless `X`.
/// A form-1 row that may be unsubmerged without a critical depth below the crown gets inverted
/// bounds, which reject the enclosing call.
pub(crate) fn headwater<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &Operands<'_, R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Operands {
        barrel,
        unit,
        inlet,
        discharge,
        bands,
        critical,
    } = *operands;
    let c = a.client();
    let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
    let q = enter(a, discharge)?;
    let scale = enter(a, &inlet.scale)?;
    let x = a.div_positive_interval(&q, &scale)?;
    let d = a.point(barrel.open_rise())?;
    let slope_term = enter(a, &inlet.slope_term)?;

    let within = |a: &mut Arithmetic<'_, R>, mask: &Tensor<R>, harmless: f64| {
        let other = barrel.constant(harmless)?;
        let lower = c.where_cond(mask, x.lower(), &other)?;
        let upper = c.where_cond(mask, x.upper(), &other)?;
        a.bounds(&lower, &upper)
    };

    // Unsubmerged: form 1 `Hc + D (K X^M + Ks S)`, form 2 `D K X^M`.
    let x_free = within(a, &bands.unsubmerged, 1.0)?;
    let k = a.point(&inlet.k)?;
    let raised = power(a, barrel, &x_free, &inlet.m)?;
    let scaled = a.mul(&k, &raised)?;
    let form_two = a.mul(&d, &scaled)?;
    let sloped = a.add(&scaled, &slope_term)?;
    let partial = a.mul(&d, &sloped)?;
    let head = critical_head(a, barrel, unit, critical)?;
    let form_one = a.add(&head.head, &partial)?;
    let free = choose(a, &inlet.form_one, &form_one, &form_two)?;

    // Submerged: `D (c X^2 + Y + Ks S)`.
    let x_full = within(a, &bands.submerged, SUBMERGED_LIMIT)?;
    let square = a.mul(&x_full, &x_full)?;
    let coefficient = a.point(&inlet.c)?;
    let quadratic = a.mul(&coefficient, &square)?;
    let offset = a.point(&inlet.y)?;
    let shifted = a.add(&quadratic, &offset)?;
    let total = a.add(&shifted, &slope_term)?;
    let full = a.mul(&d, &total)?;

    // Transition: `low + (high - low) (X - 3.5) / 0.5`.
    let x_mid = within(
        a,
        &bands.transition,
        0.5 * (UNSUBMERGED_LIMIT + SUBMERGED_LIMIT),
    )?;
    let low = enter(a, &inlet.low)?;
    let high = enter(a, &inlet.high)?;
    let limit = a.point(&barrel.constant(UNSUBMERGED_LIMIT)?)?;
    let past = a.sub(&x_mid, &limit)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let fraction = a.mul(&past, &two)?;
    let rise = a.sub(&high, &low)?;
    let gain = a.mul(&rise, &fraction)?;
    let middle = a.add(&low, &gain)?;

    let mut lower = barrel.constant(f64::INFINITY)?;
    let mut upper = barrel.constant(f64::NEG_INFINITY)?;
    for (mask, value) in [
        (&bands.unsubmerged, &free),
        (&bands.transition, &middle),
        (&bands.submerged, &full),
    ] {
        lower = c.where_cond(mask, &c.minimum(&lower, value.lower())?, &lower)?;
        upper = c.where_cond(mask, &c.maximum(&upper, value.upper())?, &upper)?;
    }
    // A form-1 row that may be unsubmerged needs a critical depth below the crown.
    let needs = c.mul(&inlet.form_one, &bands.unsubmerged)?;
    let missing = c.mul(&needs, &c.sub(&barrel.one, &head.valid)?)?;
    let lower = c.where_cond(&missing, &barrel.one, &lower)?;
    let upper = c.where_cond(&missing, &barrel.zero, &upper)?;
    a.bounds(&lower, &upper)
}
