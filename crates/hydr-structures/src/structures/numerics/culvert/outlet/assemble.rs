//! Inlet head and outlet-control headwater of every row from its class and its profile.
//!
//! | Class | Inlet head `h_in` | Area at the inlet |
//! | --- | --- | --- |
//! | no flow | `max(TW - S L, 0)` | none |
//! | full | `h_out + (Sf_full - S) L` | full |
//! | profile reaches the crown | `D + (Sf_full - S) (L_s - X)` | full |
//! | profile ends inside the barrel | depth at `v_in` | `A(v_in)` |
//! | critical at the inlet | `yc`, capped at `D` | `A(yc)` |
//! | crown tie | hull of capped `yc` and `h_out + (Sf_full - S) L` | hull of `A(yc)` and full |
//!
//! `HW_o = h_in + (1 + ke) Q^2 / (2 g A^2)`. Unresolved rows hold NaN.

use super::classes::Classes;
use super::depth::DepthEnclosure;
use super::integrand::{area, coordinate, depth};
use super::partition::{LANES, Profile};
use super::trig::fill;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose};
use crate::structures::tensor::codes;
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorOutletContext};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Rows whose free profile reaches its finite limit, with the distance it covers.
pub(crate) struct Reach<R: Runtime<DType = DType>> {
    /// Profile rows that need the depth at the target distance.
    pub(crate) solve: Tensor<R>,
    /// Rows that reach the crown and run full to the inlet.
    pub(crate) crown_full: Tensor<R>,
    /// Rows that reach critical depth before the inlet.
    pub(crate) critical: Tensor<R>,
    /// Finite rows whose length is not certified above or below the target. Their inlet lies
    /// between `near` and the limit, or the profile reaches the limit at the inlet.
    pub(crate) open: Tensor<R>,
    /// Last node certified short of the target, or the far end of the start.
    pub(crate) near: Tensor<R>,
    /// Distance `X` to the finite limit.
    pub(crate) distance: Bounds<R>,
}

/// Compares each finite profile's length with the barrel length it must cover.
pub(crate) fn reach<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    classes: &Classes<R>,
    profile: &Profile<R>,
) -> ArithmeticResult<Reach<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let one = &barrel.one;
    let m = barrel.width();
    let column = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value.narrow(1, LANES, 1)?.contiguous()?.reshape(&[m])
    };
    let last = a.bounds(&column(&profile.reach[0])?, &column(&profile.reach[1])?)?;
    let error = a.bounds(&classes.end_error[0], &classes.end_error[1])?;
    let distance = a.add(&last, &error)?;
    let active = &classes.active;
    let finite = c.mul(active, &c.sub(one, &profile.spec.asymptotic)?)?;
    let reached = c.mul(&finite, &c.lt(distance.upper(), &classes.target[0])?)?;
    let beyond = c.mul(&finite, &c.gt(distance.lower(), &classes.target[1])?)?;
    let open = c.sub(&c.sub(&finite, &reached)?, &beyond)?;
    // A crown row whose full-barrel head would not rise upstream stays unresolved.
    let rising = c.gt(&classes.rate[0], &barrel.zero)?;
    let crown_full = c.mul(&c.mul(&reached, &classes.crown_up)?, &rising)?;
    let critical = c.mul(&reached, &classes.critical_down)?;
    let open = c.mul(
        &open,
        &c.maximum(&classes.critical_down, &c.mul(&classes.crown_up, &rising)?)?,
    )?;

    // The last node whose distance lies certainly short of the target.
    let count = LANES + 1;
    let wide = |value: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        value
            .reshape(&[m, 1])?
            .broadcast_to(&[m, count])?
            .contiguous()
    };
    let short = c.lt(&profile.reach[1], &wide(&classes.target[0])?)?;
    let shorter = c.sum(&short, &[1], false)?;
    let index = c.maximum(&c.sub(&shorter, one)?, &barrel.zero)?;
    let index = c.cast(&index, DType::I64)?.reshape(&[m, 1])?;
    let node = c.gather(&profile.nodes, 1, &index)?.reshape(&[m])?;
    let none = c.eq(&shorter, &barrel.zero)?;
    let near = c.where_cond(&none, &profile.spec.start_far, &node)?;
    Ok(Reach {
        solve: c.maximum(&profile.spec.asymptotic, &beyond)?,
        crown_full,
        critical,
        open,
        near,
        distance: [distance.lower().clone(), distance.upper().clone()],
    })
}

/// Status codes, inlet heads and headwaters per row.
pub(crate) struct Assembled<R: Runtime<DType = DType>> {
    pub(crate) status: Tensor<R>,
    pub(crate) inlet: Bounds<R>,
    pub(crate) headwater: Bounds<R>,
}

/// Inputs of the final assembly.
pub(crate) struct Parts<'p, R: Runtime<DType = DType>> {
    pub(crate) barrel: &'p TensorBarrelContext<R>,
    pub(crate) outlet: &'p TensorOutletContext<R>,
    pub(crate) classes: &'p Classes<R>,
    pub(crate) reach: &'p Reach<R>,
    pub(crate) inlet: &'p DepthEnclosure<R>,
}

/// Inlet head and headwater of every row.
pub(crate) fn assemble<R>(
    a: &mut Arithmetic<'_, R>,
    parts: &Parts<'_, R>,
) -> ArithmeticResult<Assembled<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Parts {
        barrel,
        outlet,
        classes,
        reach,
        inlet,
    } = *parts;
    let c = a.client();
    let one = &barrel.one;
    let rise = &barrel.rise;
    let not = |mask: &Tensor<R>| c.sub(one, mask);

    let free_inlet = c.mul(&reach.solve, &inlet.solved)?;
    let critical = c.maximum(&classes.critical_now, &reach.critical)?;
    let full = c.maximum(&classes.full_inlet, &reach.crown_full)?;

    // Crown then full: D + r (L_s - X).
    let rate = a.bounds(&classes.rate[0], &classes.rate[1])?;
    let target = a.bounds(&classes.target[0], &classes.target[1])?;
    let distance = a.bounds(&reach.distance[0], &reach.distance[1])?;
    let left = a.sub(&target, &distance)?;
    let gain = a.mul(&rate, &left)?;
    let d = a.point(rise)?;
    let crown_head = a.add(&d, &gain)?;
    let head_full = a.bounds(&classes.head_full[0], &classes.head_full[1])?;
    let full_head = choose(a, &classes.full_inlet, &head_full, &crown_head)?;

    // Free inlet depth and area.
    let harmless = fill(rise, 0.5)?;
    let v = a.bounds(
        &c.where_cond(&free_inlet, &inlet.lower, &harmless)?,
        &c.where_cond(&free_inlet, &inlet.upper, &harmless)?,
    )?;
    let free_head = depth(a, &barrel.circular, rise, &v)?;
    let free_area = area(a, &barrel.circular, &outlet.beta, rise, &v)?;

    // Critical depth at the inlet.
    let critical_head = a.bounds(&classes.critical[0], &classes.critical[1])?;
    let v_c = coordinate(a, &barrel.circular, rise, &critical_head)?;
    let critical_area = area(a, &barrel.circular, &outlet.beta, rise, &v_c)?;

    let full_area = a.bounds(&barrel.full_area[0], &barrel.full_area[1])?;
    let head = choose(a, &free_inlet, &free_head, &critical_head)?;
    let head = choose(a, &full, &full_head, &head)?;
    let section = choose(a, &free_inlet, &free_area, &critical_area)?;
    let section = choose(a, &full, &full_area, &section)?;
    let square = a.mul(&section, &section)?;
    let velocity = a.bounds(&classes.velocity[0], &classes.velocity[1])?;
    let kinetic = a.div_positive_interval(&velocity, &square)?;
    let headwater = a.add(&head, &kinetic)?;

    // Open rows: the inlet lies between `near` and the limit, or the limit sits at the inlet.
    let open = &reach.open;
    let near = c.where_cond(open, &reach.near, &harmless)?;
    let end_low = c.where_cond(open, &classes.end[0], &harmless)?;
    let end_high = c.where_cond(open, &classes.end[1], &harmless)?;
    let span = a.bounds(&c.minimum(&near, &end_low)?, &c.maximum(&near, &end_high)?)?;
    let span_head = depth(a, &barrel.circular, rise, &span)?;
    let span_area = area(a, &barrel.circular, &outlet.beta, rise, &span)?;
    let span_square = a.mul(&span_area, &span_area)?;
    let span_kinetic = a.div_positive_interval(&velocity, &span_square)?;
    let span_headwater = a.add(&span_head, &span_kinetic)?;
    // A crown row may also run full over at most L_s - X.lo before the inlet.
    let zero = &barrel.zero;
    let rest = a.bounds(zero, &c.maximum(left.upper(), zero)?)?;
    let rate_open = a.bounds(
        &c.maximum(rate.lower(), zero)?,
        &c.maximum(rate.upper(), zero)?,
    )?;
    let rise_gain = a.mul(&rate_open, &rest)?;
    let crown_open = a.add(&d, &rise_gain)?;
    let full_square = a.mul(&full_area, &full_area)?;
    let full_kinetic = a.div_positive_interval(&velocity, &full_square)?;
    let crown_headwater = a.add(&crown_open, &full_kinetic)?;
    let hull = |a: &mut Arithmetic<'_, R>,
                x: &solvr::common::arithmetic::Interval<R>,
                y: &solvr::common::arithmetic::Interval<R>| {
        a.bounds(
            &c.minimum(x.lower(), y.lower())?,
            &c.maximum(x.upper(), y.upper())?,
        )
    };
    let open_crown = c.mul(open, &classes.crown_up)?;
    let crown_span = hull(a, &span_headwater, &crown_headwater)?;
    let crown_heads = hull(a, &span_head, &crown_open)?;
    let open_headwater = choose(a, &open_crown, &crown_span, &span_headwater)?;
    let open_head = choose(a, &open_crown, &crown_heads, &span_head)?;
    let headwater = choose(a, open, &open_headwater, &headwater)?;
    let head = choose(a, open, &open_head, &head)?;

    // Tie rows: the hull of critical depth at the crown and of a full barrel to the inlet.
    let tie = &classes.tie;
    let tie_head = a.bounds(
        &c.where_cond(tie, &classes.tie_head[0], &harmless)?,
        &c.where_cond(tie, &classes.tie_head[1], &harmless)?,
    )?;
    let tie_area = a.bounds(
        critical_area.lower(),
        &c.maximum(critical_area.upper(), full_area.upper())?,
    )?;
    let tie_square = a.mul(&tie_area, &tie_area)?;
    let tie_kinetic = a.div_positive_interval(&velocity, &tie_square)?;
    let tie_headwater = a.add(&tie_head, &tie_kinetic)?;
    let headwater = choose(a, tie, &tie_headwater, &headwater)?;
    let head = choose(a, tie, &tie_head, &head)?;

    let still = a.bounds(&classes.still[0], &classes.still[1])?;
    let headwater = choose(a, &classes.dry, &still, &headwater)?;
    let head = choose(a, &classes.dry, &still, &head)?;

    let resolved = c.maximum(
        &c.maximum(&classes.dry, &full)?,
        &c.maximum(&c.maximum(&free_inlet, &critical)?, &c.maximum(open, tie)?)?,
    )?;
    let code = |value: f64| fill(rise, value);
    let mut status = code(codes::OUTLET_UNRESOLVED)?;
    let open_critical = c.mul(open, &classes.critical_down)?;
    let critical = c.maximum(&c.maximum(&critical, &open_critical)?, tie)?;
    status = c.where_cond(&critical, &code(codes::OUTLET_INLET_CRITICAL)?, &status)?;
    let partly = c.maximum(&c.maximum(&free_inlet, &reach.crown_full)?, &open_crown)?;
    status = c.where_cond(&partly, &code(codes::OUTLET_PARTLY_FULL)?, &status)?;
    let entirely = c.mul(&classes.full_inlet, &not(&classes.falling)?)?;
    status = c.where_cond(&entirely, &code(codes::OUTLET_FULL)?, &status)?;
    status = c.where_cond(&classes.dry, &code(codes::OUTLET_NO_FLOW)?, &status)?;
    let missing = code(f64::NAN)?;
    let keep = |value: &Tensor<R>| c.where_cond(&resolved, value, &missing);
    Ok(Assembled {
        status,
        inlet: [keep(head.lower())?, keep(head.upper())?],
        headwater: [keep(headwater.lower())?, keep(headwater.upper())?],
    })
}
