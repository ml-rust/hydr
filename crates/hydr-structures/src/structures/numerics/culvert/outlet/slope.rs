//! Certified slope bounds of the outlet headwater `HW_o(Q, TW)` at one march.
//!
//! Each row gets a lower bound on `E = Q dHW_o/dQ` and an upper bound on `|dHW_o/dTW|`, both
//! at the marched discharge and tailwater. The bounds use the march's class, inlet depth and
//! start depth, and closed forms at those depths. They need no second march.
//!
//! # Notation
//!
//! `xi` runs upstream from the start of the free profile. `Es = y + Q^2 / (2 g A^2)` is the
//! specific energy, `sigma = Sf - S` and `v = Q / (g A^2)`. Along a profile at fixed `Q`:
//!
//! - `dEs/dxi = sigma` and `dy/dxi = sigma / (1 - Fr^2)`. A rising profile has `sigma > 0`.
//! - `d sigma / dxi = a sigma` with `a = (dSf/dy) / (1 - Fr^2)`. So `sigma` keeps its sign.
//!
//! # Discharge
//!
//! Let `w = dEs/dQ` at fixed `xi`, and `u = w - v`. Differentiating the profile gives
//! `u' = a u + F` with `F = (2 / Q) (Sf - S Fr^2) / (1 - Fr^2)`. A start at the outlet has
//! `u = 0`. At a fixed tailwater depth `dEs/dQ = v`. At critical depth `dEs/dy = 0`, which gives
//! the same. A start at the crown after a full segment has `u = |sigma_s| dX/dQ >= 0`, with
//! `X = (TW - D) / (S - Sf_full)`.
//!
//! The solution is `u_e = (sigma_e / sigma_s) u_s + (2 sigma_e / Q) K`, with
//! `K = int (Sf / sigma + Fr^2 / (1 - Fr^2)) dxi` over the length `l` the profile covers.
//!
//! - Rising: `F > 0`, so `u_e >= 0` and `w_e >= v_e`. Where `Sf` falls with depth, `sigma` is
//!   least at the inlet. Then `u_e <= (2 / Q) (y_e - y_s + max(S, 0) l)`.
//! - Falling below the circle conveyance peak `s_K`, or in a box: `dSf/dy <= 0`, so `a <= 0`.
//!   Then `w' = |a| (v - w) + 2 Sf / Q`, and `w` never falls below `min(w_s, min v)`. That is
//!   `v` at the deepest point, the start.
//! - Falling from above `s_K` in a circle: `Sf` rises and `Fr^2` falls with depth there. So
//!   `Sf >= S Fr^2` at `y_lo = max(y_e, s_K D)` gives `F >= 0` above `s_K`. Then `w_e` is at
//!   least `v` at `y_lo`.
//! - Falling, any shape: `Sf` has one minimum, so `Sf` is largest and `|sigma|` least at an
//!   end. Then
//!   `u_e <= |sigma_e| dX/dQ + (2 / Q) l max(1, |sigma_e / sigma_s|) max(Sf_s, Sf_e)`.
//!
//! The headwater is `HW = Es + ke Q^2 / (2 g A^2)` at the inlet. At fixed `Es` the depth falls
//! with `Q`, so `E = Q (d_e w_e + ke v_e / (1 - Fr_e^2))` with
//! `d_e = 1 - ke Fr_e^2 / (1 - Fr_e^2)`. A negative `d_e` takes the upper bound of `w_e`. A row
//! without one is unknown.
//!
//! # Tailwater
//!
//! `z = dEs/dTW` obeys `z' = a z`, so `z_e = z_s sigma_e / sigma_s`. An outlet start at the
//! tailwater has `z_s = 1 - Fr_s^2`, and a start at critical depth has `z_s = 0`. A crown start
//! moves by `dX/dTW = 1 / (S - Sf_full)`, so `z_e = |sigma_e| / (S - Sf_full)`. Then
//! `dHW/dTW = d_e z_e`. Where `a <= 0` over the profile, `|sigma_e / sigma_s| <= 1`.
//!
//! # Other classes
//!
//! | Class | `E` at least | `|dHW/dTW|` at most |
//! | --- | --- | --- |
//! | critical at the inlet | `Q^2 / (g A_c^2)`: `dEs_c/dQ = v_c`, and `A/T` rises with depth | `0` |
//! | crown reached, then full | `(1 + ke) Q^2 / (g A^2)`: `L - X` and `Sf_full - S > 0` rise with `Q` | `(Sf_full - S) (1 - Fr_s^2) / sigma_s` |
//! | full to the inlet | `(1 + ke) Q^2 / (g A^2)` | `1` |
//!
//! Open, tie and unresolved rows, and rows whose divisors are not certified positive, are
//! unknown.

use super::March;
use super::integrand::{ProfileParams, area, coordinate, shape, slopes};
use super::jet::{self, Term};
use super::trig::{fill, negate};
use crate::structures::numerics::culvert::constants::CONVEYANCE_PEAK;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::culvert::{TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Slope bounds of the outlet headwater per row.
pub(crate) struct HeadSlopes<R: Runtime<DType = DType>> {
    /// Lower bound of `Q dHW_o/dQ` in metres on `known` rows, harmless elsewhere.
    pub(crate) gain: Tensor<R>,
    /// Upper bound of `|dHW_o/dTW|` on `known` rows, harmless elsewhere.
    pub(crate) tail: Tensor<R>,
    /// 1 where both bounds hold.
    pub(crate) known: Tensor<R>,
}

/// Inputs of [`head_slopes`]: the march and the discharges and tailwaters it ran at.
pub(crate) struct SlopeOperands<'o, R: Runtime<DType = DType>> {
    pub(crate) barrel: &'o TensorBarrelContext<R>,
    pub(crate) outlet: &'o TensorOutletContext<R>,
    pub(crate) march: &'o March<R>,
    pub(crate) discharge: &'o Tensor<R>,
    pub(crate) tailwater: &'o Tensor<R>,
}

/// Area, `Fr^2` and `Sf` at one depth enclosure.
struct Section<R: Runtime<DType = DType>> {
    area: Interval<R>,
    froude: Term<R>,
    friction: Term<R>,
}

fn section<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    params: &ProfileParams<R>,
    depth: &Interval<R>,
) -> ArithmeticResult<Section<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let rise = &barrel.rise;
    let v = coordinate(a, &barrel.circular, rise, depth)?;
    let jets = shape(a, &params.circular, &params.beta, &v, 1)?;
    let values = slopes(a, params, &jets)?;
    Ok(Section {
        area: area(a, &barrel.circular, &params.beta, rise, &v)?,
        froude: jet::coefficient(a, &values.froude, 0, rise)?,
        friction: jet::coefficient(a, &values.friction, 0, rise)?,
    })
}

/// `x / y` where `y` is certified positive, with the mask of those rows. Others divide by 1.
fn quotient<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    y: &Interval<R>,
) -> ArithmeticResult<(Interval<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = y.lower();
    let positive = c.gt(like, &fill(like, 0.0)?)?;
    let one = fill(like, 1.0)?;
    let safe = a.bounds(
        &c.where_cond(&positive, y.lower(), &one)?,
        &c.where_cond(&positive, y.upper(), &one)?,
    )?;
    Ok((a.div_positive_interval(x, &safe)?, positive))
}

/// Encloses `|x|`. Every step is exact.
fn magnitude<R>(a: &mut Arithmetic<'_, R>, x: &Interval<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = fill(x.lower(), 0.0)?;
    let upper = c.maximum(&c.abs(x.lower())?, &c.abs(x.upper())?)?;
    let above = c.maximum(x.lower(), &zero)?;
    let below = c.maximum(&c.neg(x.upper())?, &zero)?;
    a.bounds(&c.maximum(&above, &below)?, &upper)
}

/// `[lower, upper]` as an interval, for upper bounds of positive factors.
fn upper_point<R>(a: &mut Arithmetic<'_, R>, upper: &Tensor<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    a.bounds(upper, upper)
}

/// Depth enclosures of the free profile: inlet, start and the circle peak bound.
struct Depths<R: Runtime<DType = DType>> {
    end: Interval<R>,
    start: Interval<R>,
    low: Interval<R>,
    /// 1 where `Sf` falls with depth over the whole profile.
    monotone: Tensor<R>,
}

fn depths<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &SlopeOperands<'_, R>,
    free: &Tensor<R>,
) -> ArithmeticResult<Depths<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let SlopeOperands {
        barrel,
        march,
        tailwater,
        ..
    } = *operands;
    let c = a.client();
    let classes = &march.classes;
    let rise = &barrel.rise;
    let half = &barrel.half_rise;
    let [inlet_low, inlet_high] = &march.headwater.inlet;
    let end = a.bounds(
        &c.where_cond(free, inlet_low, half)?,
        &c.where_cond(free, inlet_high, half)?,
    )?;
    let [yc_low, yc_high] = &classes.critical;
    let start = a.bounds(
        &c.where_cond(&classes.falling, rise, &c.maximum(tailwater, yc_low)?)?,
        &c.where_cond(&classes.falling, rise, &c.maximum(tailwater, yc_high)?)?,
    )?;
    // A box has no conveyance peak below the crown.
    let [peak_low, peak_high] = CONVEYANCE_PEAK.map(f64::from_bits);
    let fraction = a.bounds(&fill(rise, peak_low)?, &fill(rise, peak_high)?)?;
    let d = a.point(rise)?;
    let circle_peak = a.mul(&fraction, &d)?;
    let peak = choose(a, &barrel.circular, &circle_peak, &d)?;
    let deepest = c.maximum(start.upper(), end.upper())?;
    let monotone = c.maximum(&barrel.boxed, &c.le(&deepest, peak.lower())?)?;
    let low = a.bounds(
        &c.maximum(end.lower(), peak.lower())?,
        &c.maximum(end.upper(), peak.upper())?,
    )?;
    Ok(Depths {
        end,
        start,
        low,
        monotone,
    })
}

/// Lower bound of `E` and upper bound of `|dHW/dTW|` on free-inlet rows, with their mask.
struct Bound<R: Runtime<DType = DType>> {
    gain: Interval<R>,
    tail: Interval<R>,
    known: Tensor<R>,
}

/// Flow quantities every class shares.
struct Flow<R: Runtime<DType = DType>> {
    /// `Q^2 / g`.
    kinetic: Interval<R>,
    ke: Interval<R>,
    slope: Interval<R>,
    /// `Sf_full - S`.
    rate: Interval<R>,
}

fn free_bound<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &SlopeOperands<'_, R>,
    flow: &Flow<R>,
    free: &Tensor<R>,
) -> ArithmeticResult<Bound<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let SlopeOperands {
        barrel,
        march,
        tailwater,
        ..
    } = *operands;
    let c = a.client();
    let classes = &march.classes;
    let params = &classes.spec.params;
    let zero = &barrel.zero;
    let one = a.point(&barrel.one)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let Depths {
        end: y_e,
        start: y_s,
        low: y_lo,
        monotone,
    } = depths(a, operands, free)?;
    let end = section(a, barrel, params, &y_e)?;
    let start = section(a, barrel, params, &y_s)?;
    let low = section(a, barrel, params, &y_lo)?;
    let rising = c.gt(&classes.spec.sign, zero)?;
    let crown = &classes.falling;

    // `d_e = 1 - ke Fr_e^2 / (1 - Fr_e^2)` and `Q v_e = Q^2 / (g A_e^2)`.
    let subcritical = a.sub(&one, &end.froude)?;
    let scaled = a.mul(&flow.ke, &end.froude)?;
    let (excess, sub_ok) = quotient(a, &scaled, &subcritical)?;
    let d_e = a.sub(&one, &excess)?;
    let end_square = a.mul(&end.area, &end.area)?;
    let (velocity, end_ok) = quotient(a, &flow.kinetic, &end_square)?;

    // Lower bound of `Q w_e`.
    let deep_area = a.bounds(
        &c.minimum(start.area.lower(), low.area.lower())?,
        &c.minimum(start.area.upper(), low.area.upper())?,
    )?;
    let deep_square = a.mul(&deep_area, &deep_area)?;
    let (deep, deep_ok) = quotient(a, &flow.kinetic, &deep_square)?;
    let lean = a.mul(&flow.slope, &low.froude)?;
    let crown_ok = c.ge(low.friction.lower(), lean.upper())?;
    let falling_ok = c.mul(&deep_ok, &c.maximum(&monotone, &crown_ok)?)?;
    let falling_low = c.where_cond(&falling_ok, deep.lower(), zero)?;
    let omega_low = c.where_cond(&rising, velocity.lower(), &falling_low)?;

    // Upper bound of `Q w_e`, rising.
    let climb = a.sub(&y_e, &y_s)?;
    let target = a.bounds(&classes.target[0], &classes.target[1])?;
    let uphill = a.point(&c.maximum(&operands.outlet.slope, zero)?)?;
    let drop = a.mul(&uphill, &target)?;
    let reach = a.add(&climb, &drop)?;
    let twice = a.mul(&two, &reach)?;
    let rise_high = a.add(&velocity, &twice)?;

    // Upper bound of `Q w_e`, falling.
    let sigma_e = a.sub(&end.friction, &flow.slope)?;
    let sigma_s = a.sub(&start.friction, &flow.slope)?;
    let gap_e = magnitude(a, &sigma_e)?;
    let gap_s = magnitude(a, &sigma_s)?;
    let (ratio, ratio_ok) = quotient(a, &gap_e, &gap_s)?;
    let widest = upper_point(a, &c.maximum(ratio.upper(), &barrel.one)?)?;
    let steepest = upper_point(a, &c.maximum(start.friction.upper(), end.friction.upper())?)?;
    let span = a.mul(&two, &target)?;
    let spread = a.mul(&span, &widest)?;
    let friction = a.mul(&spread, &steepest)?;
    // Crown start: `Q |sigma_e| dX/dQ = 2 |sigma_e| (TW - D) Sf_full / (S - Sf_full)^2`.
    let tw = a.point(tailwater)?;
    let d = a.point(&barrel.rise)?;
    let above = a.sub(&tw, &d)?;
    let full = a.add(&flow.rate, &flow.slope)?;
    let lift = a.mul(&above, &full)?;
    let lifted = a.mul(&gap_e, &lift)?;
    let doubled = a.mul(&two, &lifted)?;
    let rate_square = a.mul(&flow.rate, &flow.rate)?;
    let (shift, shift_ok) = quotient(a, &doubled, &rate_square)?;
    let nothing = a.point(zero)?;
    let shift = choose(a, crown, &shift, &nothing)?;
    let shifted = a.add(&velocity, &shift)?;
    let fall_high = a.add(&shifted, &friction)?;
    let fall_ok = c.mul(
        &ratio_ok,
        &c.maximum(&c.sub(&barrel.one, crown)?, &shift_ok)?,
    )?;
    let high_ok = c.where_cond(&rising, &monotone, &fall_ok)?;
    let omega_high = c.where_cond(&rising, rise_high.upper(), fall_high.upper())?;
    let omega_high = c.where_cond(&high_ok, &omega_high, &omega_low)?;
    let omega = a.bounds(&omega_low, &c.maximum(&omega_high, &omega_low)?)?;

    // `E >= d_e Q w_e + ke Q v_e / (1 - Fr_e^2)`.
    let product = a.mul(&d_e, &omega)?;
    let entry = a.mul(&flow.ke, &velocity)?;
    let (entry, _) = quotient(a, &entry, &subcritical)?;
    let gain = a.add(&product, &entry)?;
    let sign_ok = c.maximum(&c.ge(d_e.lower(), zero)?, &high_ok)?;

    // `|dHW/dTW| <= |d_e| |z_e|`.
    let unit = upper_point(a, &barrel.one)?;
    let bounded = c.where_cond(
        &ratio_ok,
        &c.minimum(ratio.upper(), &barrel.one)?,
        &barrel.one,
    )?;
    let share = c.where_cond(&monotone, &bounded, ratio.upper())?;
    let share_ok = c.maximum(&monotone, &ratio_ok)?;
    let share = upper_point(a, &share)?;
    let start_sub = a.sub(&unit, &start.froude)?;
    let start_sub = magnitude(a, &start_sub)?;
    let outlet_z = a.mul(&start_sub, &share)?;
    let decline = negate(a, &flow.rate)?;
    let (crown_z, crown_ok) = quotient(a, &gap_e, &decline)?;
    let z = choose(a, crown, &crown_z, &outlet_z)?;
    let z_ok = c.where_cond(crown, &crown_ok, &share_ok)?;
    let scale = magnitude(a, &d_e)?;
    let tail = a.mul(&scale, &z)?;

    let known = c.mul(
        &c.mul(&c.mul(free, &sub_ok)?, &c.mul(&end_ok, &sign_ok)?)?,
        &z_ok,
    )?;
    Ok(Bound { gain, tail, known })
}

/// Slope bounds of every row of the march.
pub(crate) fn head_slopes<R>(
    a: &mut Arithmetic<'_, R>,
    operands: &SlopeOperands<'_, R>,
) -> ArithmeticResult<HeadSlopes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let SlopeOperands {
        barrel,
        outlet,
        march,
        discharge,
        tailwater,
    } = *operands;
    let c = a.client();
    let classes = &march.classes;
    let reach = &march.reach;
    let zero = &barrel.zero;
    let one = &barrel.one;
    let not = |mask: &Tensor<R>| c.sub(one, mask);

    let wet = c.gt(discharge, zero)?;
    let q = a.point(&c.where_cond(&wet, discharge, one)?)?;
    let square = a.mul(&q, &q)?;
    let flow = Flow {
        kinetic: a.div_positive(&square, &outlet.gravity)?,
        ke: a.point(&outlet.entrance)?,
        slope: a.point(&outlet.slope)?,
        rate: a.bounds(&classes.rate[0], &classes.rate[1])?,
    };
    let unit = a.point(one)?;
    let loss = a.add(&unit, &flow.ke)?;

    // Free inlet rows: a profile that ends inside the barrel, with a resolved inlet depth.
    let [inlet_low, inlet_high] = &march.headwater.inlet;
    let finite = c.mul(&c.eq(inlet_low, inlet_low)?, &c.eq(inlet_high, inlet_high)?)?;
    let free = c.mul(
        &c.mul(&reach.solve, &not(&reach.open)?)?,
        &c.mul(&finite, &not(&classes.tie)?)?,
    )?;
    let free_bound = free_bound(a, operands, &flow, &free)?;

    // Critical depth at the inlet.
    let critical = c.mul(&c.maximum(&classes.critical_now, &reach.critical)?, &wet)?;
    let critical_depth = a.bounds(&classes.critical[0], &classes.critical[1])?;
    let v_c = coordinate(a, &barrel.circular, &barrel.rise, &critical_depth)?;
    let critical_area = area(a, &barrel.circular, &outlet.beta, &barrel.rise, &v_c)?;
    let critical_square = a.mul(&critical_area, &critical_area)?;
    let (critical_gain, critical_ok) = quotient(a, &flow.kinetic, &critical_square)?;

    // Full at the inlet, from the outlet or after the crown.
    let full_area = a.bounds(&barrel.full_area[0], &barrel.full_area[1])?;
    let full_square = a.mul(&full_area, &full_area)?;
    let lossy = a.mul(&loss, &flow.kinetic)?;
    let (full_gain, full_ok) = quotient(a, &lossy, &full_square)?;
    let crown = c.mul(&reach.crown_full, &wet)?;
    let entire = c.mul(&classes.full_inlet, &wet)?;
    // Crown rows: `|dHW/dTW| = (Sf_full - S) (1 - Fr_s^2) / sigma_s` at the start `max(TW, yc)`.
    let [yc_low, yc_high] = &classes.critical;
    let h0 = a.bounds(
        &c.maximum(tailwater, yc_low)?,
        &c.maximum(tailwater, yc_high)?,
    )?;
    let start = section(a, barrel, &classes.spec.params, &h0)?;
    let start_sub = a.sub(&unit, &start.froude)?;
    let start_sub = magnitude(a, &start_sub)?;
    let sigma = a.sub(&start.friction, &flow.slope)?;
    let sigma = magnitude(a, &sigma)?;
    let rate = magnitude(a, &flow.rate)?;
    let product = a.mul(&rate, &start_sub)?;
    let (crown_tail, crown_ok) = quotient(a, &product, &sigma)?;

    let pick = |mask: &Tensor<R>, on: &Tensor<R>, off: &Tensor<R>| c.where_cond(mask, on, off);
    let mut gain = pick(&free, free_bound.gain.lower(), zero)?;
    let mut tail = pick(&free, free_bound.tail.upper(), zero)?;
    let mut known = free_bound.known;
    gain = pick(&entire, full_gain.lower(), &gain)?;
    tail = pick(&entire, one, &tail)?;
    known = pick(&entire, &full_ok, &known)?;
    gain = pick(&crown, full_gain.lower(), &gain)?;
    tail = pick(&crown, crown_tail.upper(), &tail)?;
    known = pick(&crown, &c.mul(&full_ok, &crown_ok)?, &known)?;
    gain = pick(&critical, critical_gain.lower(), &gain)?;
    tail = pick(&critical, zero, &tail)?;
    known = pick(&critical, &critical_ok, &known)?;
    Ok(HeadSlopes {
        gain,
        tail,
        known: c.mul(&known, &wet)?,
    })
}
