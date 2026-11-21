//! Outlet classes of each row and the subcritical profile that the march from the outlet follows.
//!
//! | Rows | Class |
//! | --- | --- |
//! | `Q = 0` | no flow, level pool at `max(TW - S L, 0)` |
//! | `TW >= D`, `h_out + (Sf_full - S) L >= D` | full to the inlet |
//! | `TW >= D`, head falls below the crown inside the barrel | full, then a falling profile from the crown |
//! | `TW < D`, `Sf(h0) > S` at `h0 = max(TW, yc)` | rising profile toward `yn` or the crown |
//! | `TW < D`, `Sf(h0) < S`, `TW > yc` | falling profile toward `yn` (mild) or `yc` (steep) |
//! | `TW < D`, `Sf(h0) < S`, `TW <= yc` | critical depth at the inlet |
//!
//! Critical depth is capped at the crown, as HDS-5 caps `dc` at `D`. A box above its critical
//! capacity `sqrt(g) B D^(3/2)` takes `yc = D`. Its free profile starts at the crown, where `Sf`
//! takes the free-surface perimeter. Its falling profile from the crown is empty, so it leaves
//! critical depth at the inlet.
//!
//! Every comparison is certified. A row whose comparison is not, or whose critical depth lies
//! below the search floor, is unresolved.

use super::integrand::{ProfileParams, coordinate, integrand, shape, slopes};
use super::jet::{self, exp_interval};
use super::partition::{ProfileSpec, guarded};
use super::trig::fill;
use crate::structures::numerics::culvert::constants::PI;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, BarrelDepths, choose};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Query operands of one classification.
pub(crate) struct Operands<'o, R: Runtime<DType = DType>> {
    pub(crate) barrel: &'o TensorBarrelContext<R>,
    pub(crate) outlet: &'o TensorOutletContext<R>,
    pub(crate) discharge: &'o Tensor<R>,
    pub(crate) tailwater: &'o Tensor<R>,
    pub(crate) critical: &'o BarrelDepths<R>,
    pub(crate) normal: &'o BarrelDepths<R>,
}

/// Row classes, the quantities later stages read, and the subcritical profile.
pub(crate) struct Classes<R: Runtime<DType = DType>> {
    pub(crate) dry: Tensor<R>,
    pub(crate) full_inlet: Tensor<R>,
    pub(crate) critical_now: Tensor<R>,
    /// Rows with critical depth at the crown whose crown comparison is not certified.
    pub(crate) tie: Tensor<R>,
    /// Inlet head hull of a tie row: capped critical depth to the full-barrel head.
    pub(crate) tie_head: Bounds<R>,
    /// Rows with a subcritical profile.
    pub(crate) active: Tensor<R>,
    /// Profile rows that reach the crown going upstream.
    pub(crate) crown_up: Tensor<R>,
    /// Profile rows that reach critical depth going upstream.
    pub(crate) critical_down: Tensor<R>,
    /// Profile rows whose outlet segment flows full.
    pub(crate) falling: Tensor<R>,
    /// `Sf_full - S`.
    pub(crate) rate: Bounds<R>,
    /// `h_out + (Sf_full - S) L`.
    pub(crate) head_full: Bounds<R>,
    /// Barrel length the free profile covers.
    pub(crate) target: Bounds<R>,
    /// Distance from the represented limit to the exact limit of a finite profile.
    pub(crate) end_error: Bounds<R>,
    /// Coordinate enclosure of the exact limit of a finite profile, harmless elsewhere.
    pub(crate) end: Bounds<R>,
    /// Critical depth, harmless on rows without one.
    pub(crate) critical: Bounds<R>,
    /// `max(TW - S L, 0)`.
    pub(crate) still: Bounds<R>,
    /// `(1 + ke) Q^2 / (2 g)`.
    pub(crate) velocity: Bounds<R>,
    pub(crate) spec: ProfileSpec<R>,
}

fn ends<R: Runtime<DType = DType>>(value: &Interval<R>) -> Bounds<R> {
    [value.lower().clone(), value.upper().clone()]
}

/// Selects `on` where `mask` is 1, else `off`, for endpoint pairs.
fn pick<R>(
    c: &R::Client,
    mask: &Tensor<R>,
    on: &Bounds<R>,
    off: &Bounds<R>,
) -> numr::error::Result<Bounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok([
        c.where_cond(mask, &on[0], &off[0])?,
        c.where_cond(mask, &on[1], &off[1])?,
    ])
}

/// `[min(0, w lower), max(0, w upper)]`: the integral of a function enclosed by `value` over a
/// span of length at most `width`.
fn sliver<R>(
    a: &mut Arithmetic<'_, R>,
    width: &Tensor<R>,
    value: &Interval<R>,
) -> ArithmeticResult<Bounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let w = a.point(width)?;
    let scaled = a.mul(&w, value)?;
    let zero = fill(width, 0.0)?;
    Ok([
        c.minimum(scaled.lower(), &zero)?,
        c.maximum(scaled.upper(), &zero)?,
    ])
}

/// Classifies every row and describes its subcritical profile.
pub(crate) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &Operands<'_, R>,
) -> ArithmeticResult<Classes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Operands {
        barrel,
        outlet,
        discharge,
        tailwater,
        critical,
        normal,
    } = *operands;
    let c = a.client();
    let zero = &barrel.zero;
    let one = &barrel.one;
    let rise = &barrel.rise;
    let not = |mask: &Tensor<R>| c.sub(one, mask);
    let and = |x: &Tensor<R>, y: &Tensor<R>| c.mul(x, y);
    let or = |x: &Tensor<R>, y: &Tensor<R>| c.maximum(x, y);
    let is = |status: &Tensor<R>, code: f64| c.eq(status, &fill(status, code)?);

    let dry = c.eq(discharge, zero)?;
    let wet = not(&dry)?;
    let crit_solved = and(&is(&critical.status, codes::CRITICAL_SOLVED)?, &wet)?;
    let crit_crown = and(
        &or(
            &is(&critical.status, codes::CRITICAL_AT_CROWN)?,
            &is(&critical.status, codes::CRITICAL_ABOVE_CROWN)?,
        )?,
        &wet,
    )?;
    let n_solved = is(&normal.status, codes::NORMAL_SOLVED)?;
    let n_root = or(
        &or(&n_solved, &is(&normal.status, codes::NORMAL_NEAR_CAPACITY)?)?,
        &is(&normal.status, codes::NORMAL_BELOW_FLOOR)?,
    )?;
    let n_floor = is(&normal.status, codes::NORMAL_BELOW_FLOOR)?;
    let n_none = or(
        &is(&normal.status, codes::NORMAL_ABOVE_CAPACITY)?,
        &is(&normal.status, codes::NORMAL_NON_POSITIVE_SLOPE)?,
    )?;
    let half = &barrel.half_rise;
    // Critical depth cannot exceed the crown. A box above its critical capacity has no root, so
    // its capped critical depth is the crown itself.
    let crit_known = or(&crit_solved, &crit_crown)?;
    let box_above = and(
        &is(&critical.status, codes::CRITICAL_ABOVE_CROWN)?,
        &barrel.boxed,
    )?;
    let capped = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        let below = c.where_cond(&box_above, rise, value)?;
        c.where_cond(&crit_known, &c.minimum(&below, rise)?, half)
    };
    let yc = [capped(&critical.lower)?, capped(&critical.upper)?];
    let yn = [
        c.where_cond(&n_root, &normal.lower, half)?,
        c.where_cond(&n_root, &normal.upper, half)?,
    ];

    // Flow scales: F = Q^2 / (g D^5), ln K = 2 ln n + 2 ln Q - 16 ln D / 3, full friction.
    let flow = a.point(&c.where_cond(&dry, one, discharge)?)?;
    let log_q = a.log_positive(&flow)?;
    let g = a.point(&outlet.gravity)?;
    let log_g = a.log_positive(&g)?;
    let d = a.point(rise)?;
    let log_d = a.log_positive(&d)?;
    let n = a.point(&outlet.manning)?;
    let log_n = a.log_positive(&n)?;
    let constant = |a: &mut Arithmetic<'_, R>, value: f64| a.point(&fill(rise, value)?);
    let two = constant(a, 2.0)?;
    let five = constant(a, 5.0)?;
    let sixteen = constant(a, 16.0)?;
    let twice_q = a.mul(&two, &log_q)?;
    let twice_n = a.mul(&two, &log_n)?;
    let gravity_part = a.sub(&twice_q, &log_g)?;
    let size = a.mul(&five, &log_d)?;
    let log_froude = a.sub(&gravity_part, &size)?;
    let froude = exp_interval(a, &log_froude)?;
    let roughness = a.add(&twice_n, &twice_q)?;
    let sized = a.mul(&sixteen, &log_d)?;
    let sized = a.div_positive(&sized, &fill(rise, 3.0)?)?;
    let log_friction = a.sub(&roughness, &sized)?;

    let full_area = a.bounds(&barrel.full_area[0], &barrel.full_area[1])?;
    let full_radius = a.bounds(&barrel.full_radius[0], &barrel.full_radius[1])?;
    let log_area = a.log_positive(&full_area)?;
    let log_radius = a.log_positive(&full_radius)?;
    let four = constant(a, 4.0)?;
    let area_part = a.mul(&two, &log_area)?;
    let radius_part = a.mul(&four, &log_radius)?;
    let radius_part = a.div_positive(&radius_part, &fill(rise, 3.0)?)?;
    let shrink = a.add(&area_part, &radius_part)?;
    let log_full = a.sub(&roughness, &shrink)?;
    let full_friction = exp_interval(a, &log_full)?;
    let slope = a.point(&outlet.slope)?;
    let rate = a.sub(&full_friction, &slope)?;
    let length = a.point(&outlet.length)?;
    let ke = a.point(&outlet.entrance)?;
    let unit = a.point(one)?;
    let loss = a.add(&unit, &ke)?;
    let flow_square = a.mul(&flow, &flow)?;
    let doubled_g = a.mul(&two, &g)?;
    let kinetic = a.div_positive_interval(&flow_square, &doubled_g)?;
    let velocity = a.mul(&loss, &kinetic)?;

    // Still water: max(TW - S L, 0).
    let tw = a.point(tailwater)?;
    let drop = a.mul(&slope, &length)?;
    let level = a.sub(&tw, &drop)?;
    let still = [
        c.maximum(level.lower(), zero)?,
        c.maximum(level.upper(), zero)?,
    ];

    // Outlet full.
    let crit_floor = and(&is(&critical.status, codes::CRITICAL_BELOW_FLOOR)?, &wet)?;
    let outlet_full = and(&and(&c.ge(tailwater, rise)?, &wet)?, &not(&crit_floor)?)?;
    let outlet_head = a.point(&c.maximum(tailwater, rise)?)?;
    let gain = a.mul(&rate, &length)?;
    let head_full = a.add(&outlet_head, &gain)?;
    let full_inlet = and(&outlet_full, &c.ge(head_full.lower(), rise)?)?;
    let falling = and(
        &and(&outlet_full, &c.lt(head_full.upper(), rise)?)?,
        &crit_known,
    )?;
    let negative = super::trig::negate(a, &rate)?;
    let decline = a.bounds(
        &c.where_cond(&falling, negative.lower(), one)?,
        &c.where_cond(&falling, negative.upper(), one)?,
    )?;
    let excess = a.sub(&outlet_head, &d)?;
    let reach = a.div_positive_interval(&excess, &decline)?;
    let remaining = a.sub(&length, &reach)?;
    let target = pick(c, &falling, &ends(&remaining), &ends(&length))?;

    // Free outlet: direction from Sf(h0) against S.
    let free = and(&and(&crit_known, &c.lt(tailwater, rise)?)?, &wet)?;
    let h0 = a.bounds(
        &c.maximum(tailwater, &yc[0])?,
        &c.maximum(tailwater, &yc[1])?,
    )?;
    let v0 = coordinate(a, &barrel.circular, rise, &h0)?;
    let base = ProfileParams {
        circular: barrel.circular.clone(),
        beta: outlet.beta.clone(),
        rise: rise.clone(),
        froude: ends(&froude),
        log_friction: ends(&log_friction),
        slope: outlet.slope.clone(),
        numerator: one.clone(),
        denominator: one.clone(),
    };
    let section = shape(a, &barrel.circular, &outlet.beta, &v0, 1)?;
    let at_start = slopes(a, &base, &section)?;
    let start_friction = jet::coefficient(a, &at_start.friction, 0, rise)?;
    let up = and(&free, &c.gt(start_friction.lower(), &outlet.slope)?)?;
    let down = and(&free, &c.lt(start_friction.upper(), &outlet.slope)?)?;
    let critical_now = and(&down, &c.le(tailwater, &yc[1])?)?;
    let down_free = c.sub(&down, &critical_now)?;

    // A capped row whose crown comparison is not certified: the free-surface friction at the
    // crown against the slope, or the full-barrel head at the inlet against the crown. Its inlet
    // head lies between the capped critical depth and the full-barrel head.
    let capped_row = and(&crit_known, &c.ge(&yc[1], rise)?)?;
    let tie_free = and(&free, &not(&or(&up, &down)?)?)?;
    let tie_full = and(&outlet_full, &not(&or(&full_inlet, &falling)?)?)?;
    let tie = and(&capped_row, &or(&tie_free, &tie_full)?)?;
    let tie_head = [
        c.minimum(&yc[0], head_full.lower())?,
        c.maximum(rise, head_full.upper())?,
    ];
    let descending = or(&down_free, &falling)?;

    // Limits.
    let asym_up = and(&and(&up, &n_solved)?, &c.gt(&yn[0], h0.upper())?)?;
    let below_start = and(&n_root, &c.lt(&yn[1], h0.lower())?)?;
    let crown_up = and(&up, &or(&n_none, &below_start)?)?;
    let start_depth = c.where_cond(&falling, rise, tailwater)?;
    let mild = and(
        &and(&n_root, &c.gt(&yn[0], &yc[1])?)?,
        &c.lt(&yn[1], &start_depth)?,
    )?;
    let asym_down = and(&descending, &mild)?;
    let steep = or(&n_floor, &and(&n_root, &c.lt(&yn[1], &yc[0])?)?)?;
    let critical_down = and(&descending, &steep)?;

    // Coordinates of the start and the limit.
    let crown = a.bounds(
        &fill(rise, f64::from_bits(PI[0]))?,
        &fill(rise, f64::from_bits(PI[1]))?,
    )?;
    let crown = choose(a, &barrel.circular, &crown, &unit)?;
    let v_tw = coordinate(a, &barrel.circular, rise, &tw)?;
    let normal_depth = a.bounds(&yn[0], &yn[1])?;
    let v_yn = coordinate(a, &barrel.circular, rise, &normal_depth)?;
    let critical_depth = a.bounds(&yc[0], &yc[1])?;
    let v_yc = coordinate(a, &barrel.circular, rise, &critical_depth)?;
    let start = pick(
        c,
        &up,
        &ends(&v0),
        &pick(c, &falling, &ends(&crown), &ends(&v_tw))?,
    )?;
    let harmless = [fill(rise, 0.5)?, fill(rise, 0.5)?];
    let active_raw = or(&or(&asym_up, &crown_up)?, &or(&asym_down, &critical_down)?)?;
    let start = pick(c, &active_raw, &start, &harmless)?;
    let represented = c.where_cond(&up, &start[1], &start[0])?;
    let far = c.where_cond(&up, &start[0], &start[1])?;
    let limit_up = c.where_cond(&asym_up, v_yn.lower(), crown.lower())?;
    let limit_down = c.where_cond(&asym_down, v_yn.upper(), v_yc.upper())?;
    let limit = c.where_cond(&up, &limit_up, &limit_down)?;
    let limit = c.where_cond(&active_raw, &limit, &represented)?;
    let sign = c.where_cond(&up, one, &fill(rise, -1.0)?)?;
    let asymptotic = or(&asym_up, &asym_down)?;
    let room = or(
        &not(&asymptotic)?,
        &guarded(c, &sign, &represented, &limit)?,
    )?;
    let end = pick(c, &crown_up, &ends(&crown), &ends(&v_yc))?;
    let finite = and(&active_raw, &not(&asymptotic)?)?;
    let end = pick(c, &finite, &end, &harmless)?;
    // A finite limit at or behind the start leaves an empty profile.
    let ahead = c.gt(&c.mul(&sign, &c.sub(&limit, &represented)?)?, zero)?;
    let limit = c.where_cond(&and(&finite, &not(&ahead)?)?, &represented, &limit)?;
    let params = ProfileParams {
        numerator: one.clone(),
        denominator: c.mul(&sign, &fill(rise, -1.0)?)?,
        ..base
    };
    let start_interval = a.bounds(&start[0], &start[1])?;
    let start_point = a.point(&start[0])?;
    let start_width = a.sub(&start_interval, &start_point)?;
    let at_start = integrand(a, &params, &start_interval, 1)?;
    let start_value = jet::coefficient(a, &at_start.value, 0, rise)?;
    let start_error = sliver(a, start_width.upper(), &start_value)?;
    let end_interval = a.bounds(&end[0], &end[1])?;
    let end_point = a.point(&end[0])?;
    let end_width = a.sub(&end_interval, &end_point)?;
    let at_end = integrand(a, &params, &end_interval, 1)?;
    let end_value = jet::coefficient(a, &at_end.value, 0, rise)?;
    let end_error = sliver(a, end_width.upper(), &end_value)?;
    let end_ok = or(&not(&finite)?, &at_end.valid)?;
    let active = and(&and(&active_raw, &room)?, &and(&at_start.valid, &end_ok)?)?;

    let spec = ProfileSpec {
        params,
        active: active.clone(),
        asymptotic: and(&asymptotic, &active)?,
        sign,
        start: c.where_cond(&active, &represented, &harmless[0])?,
        start_far: c.where_cond(&active, &far, &harmless[0])?,
        start_error: pick(c, &active, &start_error, &[zero.clone(), zero.clone()])?,
        limit: c.where_cond(&active, &limit, &harmless[0])?,
        scale: c.where_cond(&active, &target[1], one)?,
    };
    Ok(Classes {
        dry,
        full_inlet,
        critical_now,
        tie,
        tie_head,
        crown_up: and(&crown_up, &active)?,
        critical_down: and(&critical_down, &active)?,
        falling: and(&falling, &active)?,
        active,
        rate: ends(&rate),
        head_full: ends(&head_full),
        target,
        end_error,
        end,
        critical: yc,
        still,
        velocity: ends(&velocity),
        spec,
    })
}
