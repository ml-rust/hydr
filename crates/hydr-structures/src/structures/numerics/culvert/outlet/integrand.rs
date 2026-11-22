//! Direct-step integrand `G = D (ds/dv) lambda (1 - Fr^2) / (mu (S - Sf))` in a smooth coordinate.
//!
//! A circle uses the half angle `v = phi`, with `s = (1 - cos phi) / 2`. Area, top width and
//! perimeter are then entire functions of `phi`, so the profile stays smooth up to the crown.
//! A box uses `v = s`. An arc barrel uses the normal angle `v = psi`, band by band, as
//! [`bands`](super::bands) gives it. In units of `D`:
//!
//! | Shape | `a = A / D^2` | `t = T / D` | `p = P / D` | `ds/dv` |
//! | --- | --- | --- | --- | --- |
//! | circle | `(phi - sin phi cos phi) / 4` | `sin phi` | `phi` | `sin phi / 2` |
//! | box | `beta s` | `beta` | `beta + 2 s` | `1` |
//! | arc band `k` | `c_k - 2 x_k r_k cos psi + r_k^2 (psi - sin psi cos psi)` | `2 (x_k + r_k sin psi)` | `p_k + 2 r_k (psi - psi_k)` | `r_k sin psi` |
//!
//! `beta = B / D`. With `F = Q^2 / (g D^5)` and `K = n^2 Q^2 / D^(16/3)`, `Fr^2 = F t / a^3` and
//! `Sf = K exp((4 ln p - 10 ln a) / 3)`. The signs `lambda` and `mu` make the numerator and the
//! denominator positive along a profile, so `G > 0` is the distance gained per unit of `v`.

use super::bands::{self, ArcShape, Outline, Trig};
use super::jet::{self, Jet};
use super::trig::{fill, negate};
use super::trig_ends::{sin_cos_ends, sin_cos_near};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};
use std::rc::Rc;

/// Integrand parameters, one row per lane or per barrel.
#[derive(Clone)]
pub(crate) struct ProfileParams<R: Runtime<DType = DType>> {
    /// 1 on circle rows, 0 on box rows.
    pub(crate) circular: Tensor<R>,
    /// `B / D`.
    pub(crate) beta: Bounds<R>,
    /// Rise `D` in metres.
    pub(crate) rise: Tensor<R>,
    /// `F = Q^2 / (g D^5)`.
    pub(crate) froude: Bounds<R>,
    /// `ln K`, `K = n^2 Q^2 / D^(16/3)`.
    pub(crate) log_friction: Bounds<R>,
    /// Signed barrel slope `S`.
    pub(crate) slope: Tensor<R>,
    /// `lambda`: +1 for subcritical profiles, -1 for supercritical ones.
    pub(crate) numerator: Tensor<R>,
    /// `mu`: +1 where `S > Sf` along the profile, -1 where `Sf > S`.
    pub(crate) denominator: Tensor<R>,
    /// Coordinate enclosure of the conveyance peak `s_K` on circle and arc rows.
    pub(crate) peak: Bounds<R>,
    /// Lower bound of the signed denominator `mu (S - Sf)` over the peak enclosure where `Sf`
    /// may take its extreme there, `f64::MAX` elsewhere. See [`super::screen::with_peak`].
    pub(crate) peak_excess: Tensor<R>,
    /// Arc bands in units of the rise, `None` when no row is an arc barrel.
    pub(crate) arcs: Option<ArcShape<R>>,
}

impl<R: Runtime<DType = DType>> ProfileParams<R> {
    fn map(
        &self,
        take: &dyn Fn(&Tensor<R>) -> numr::error::Result<Tensor<R>>,
        arcs: Option<ArcShape<R>>,
    ) -> numr::error::Result<Self> {
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        Ok(Self {
            circular: take(&self.circular)?,
            beta: pair(&self.beta)?,
            rise: take(&self.rise)?,
            froude: pair(&self.froude)?,
            log_friction: pair(&self.log_friction)?,
            slope: take(&self.slope)?,
            numerator: take(&self.numerator)?,
            denominator: take(&self.denominator)?,
            peak: pair(&self.peak)?,
            peak_excess: take(&self.peak_excess)?,
            arcs,
        })
    }

    /// Gathers the rows at the I64 indices `rows`. Indices may repeat.
    pub(crate) fn select(&self, c: &R::Client, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let arcs = self
            .arcs
            .as_ref()
            .map(|arcs| arcs.select(c, rows))
            .transpose()?;
        self.map(&|value| c.index_select(value, 0, rows), arcs)
    }

    /// Rows `start .. start + len`.
    pub(crate) fn narrow(&self, start: usize, len: usize) -> numr::error::Result<Self> {
        let arcs = self
            .arcs
            .as_ref()
            .map(|arcs| arcs.narrow(start, len))
            .transpose()?;
        self.map(&|value| value.narrow(0, start, len), arcs)
    }

    /// The shape view of these rows.
    pub(crate) fn outline(&self) -> Outline<'_, R> {
        Outline {
            circular: &self.circular,
            beta: &self.beta,
            arcs: self.arcs.as_ref(),
        }
    }
}

/// Section jets `[a, t, p, ds/dv]` in units of `D`.
pub(crate) struct ShapeJets<R: Runtime<DType = DType>> {
    pub(crate) area: Jet<R>,
    pub(crate) top: Jet<R>,
    pub(crate) perimeter: Jet<R>,
    pub(crate) step: Jet<R>,
    /// 1 on rows whose jets hold, when some may not: an arc derivative jet across a junction.
    pub(crate) valid: Option<Tensor<R>>,
}

/// `x / divisor` for an exact positive divisor.
pub(crate) fn part<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
    divisor: f64,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    a.div_positive(x, &fill(x.lower(), divisor)?)
}

/// Section jets with `n` terms over the coordinate enclosure `v`.
///
/// Circle and arc rows need `v` inside `[0, pi]` and box rows inside `[0, 1]`, up to one F64
/// outward.
pub(crate) fn shape<R>(
    a: &mut Arithmetic<'_, R>,
    outline: &Outline<'_, R>,
    v: &Interval<R>,
    n: usize,
) -> ArithmeticResult<ShapeJets<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let circular = outline.circular;
    let beta = outline.beta;
    let like = v.lower();
    let one = Rc::new(a.point(&fill(like, 1.0)?)?);
    let take = |a: &mut Arithmetic<'_, R>, mask: &Tensor<R>, harmless: f64| {
        let other = fill(like, harmless)?;
        let lower = c.where_cond(mask, v.lower(), &other)?;
        let upper = c.where_cond(mask, v.upper(), &other)?;
        a.bounds(&lower, &upper)
    };

    // Circle and arc rows share the angle and its sine and cosine.
    let phi = Rc::new(take(a, &outline.angled(c)?, 1.5)?);
    // Value queries take narrow intervals and derivative jets take wide panels.
    let [sine, cosine] = if n > 1 {
        sin_cos_ends(a, &phi)?
    } else {
        sin_cos_near(a, &phi)?
    };
    let (sine, cosine) = (Rc::new(sine), Rc::new(cosine));
    let mut sine_terms = vec![Some(Rc::clone(&sine))];
    let mut cosine_terms = vec![Some(Rc::clone(&cosine))];
    if n > 1 {
        // sin^(k) / k! and cos^(k) / k! cycle through +-sin and +-cos.
        let minus_sine = negate(a, &sine)?;
        let minus_cosine = negate(a, &cosine)?;
        let shared = |value: Interval<R>| Some(Rc::new(value));
        sine_terms.extend([
            Some(Rc::clone(&cosine)),
            shared(part(a, &minus_sine, 2.0)?),
            shared(part(a, &minus_cosine, 6.0)?),
            shared(part(a, &sine, 24.0)?),
        ]);
        cosine_terms.extend([
            shared(minus_sine),
            shared(part(a, &minus_cosine, 2.0)?),
            shared(part(a, &sine, 6.0)?),
            shared(part(a, &cosine, 24.0)?),
        ]);
    }
    sine_terms.truncate(n);
    cosine_terms.truncate(n);
    let sin_jet = Jet::new(sine_terms);
    let cos_jet = Jet::new(cosine_terms);
    let angle = Jet::linear(Rc::clone(&phi), Some(Rc::clone(&one)), n);
    let product = jet::mul(a, &sin_jet, &cos_jet)?;
    let difference = jet::sub(a, &angle, &product)?;
    let circle_area = jet::divide(a, &difference, 4.0)?;
    let circle_step = jet::divide(a, &sin_jet, 2.0)?;

    let boxed = c.sub(&fill(like, 1.0)?, circular)?;
    let s = take(a, &boxed, 0.5)?;
    let b = Rc::new(a.bounds(&beta[0], &beta[1])?);
    let scaled = Rc::new(a.mul(&b, &s)?);
    let box_area = Jet::linear(scaled, Some(Rc::clone(&b)), n);
    let two = Rc::new(a.point(&fill(like, 2.0)?)?);
    let doubled = a.mul(&two, &s)?;
    let wetted = Rc::new(a.add(&b, &doubled)?);
    let box_perimeter = Jet::linear(wetted, Some(two), n);
    let box_top = Jet::constant(b, n);
    let box_step = Jet::constant(one, n);

    let closed = ShapeJets {
        area: jet::choose(a, circular, &circle_area, &box_area)?,
        top: jet::choose(a, circular, &sin_jet, &box_top)?,
        perimeter: jet::choose(a, circular, &angle, &box_perimeter)?,
        step: jet::choose(a, circular, &circle_step, &box_step)?,
        valid: None,
    };
    let Some(arcs) = outline.arcs else {
        return Ok(closed);
    };
    let ends = if n > 1 {
        let mut ends = Vec::with_capacity(2);
        for end in [phi.lower(), phi.upper()] {
            let point = a.point(end)?;
            let [sine, cosine] = sin_cos_near(a, &point)?;
            ends.push([point, sine, cosine]);
        }
        <[[Interval<R>; 3]; 2]>::try_from(ends).ok()
    } else {
        None
    };
    let trig = Trig {
        sine: &sin_jet,
        cosine: &cos_jet,
        segment: &difference,
        ends,
    };
    let arc = bands::shape_jets(a, arcs, &trig, &phi, n)?;
    let arced = &arcs.arced;
    let valid = match arc.valid {
        Some(valid) => c.where_cond(arced, &valid, &fill(like, 1.0)?)?,
        None => fill(like, 1.0)?,
    };
    Ok(ShapeJets {
        area: jet::choose(a, arced, &arc.area, &closed.area)?,
        top: jet::choose(a, arced, &arc.top, &closed.top)?,
        perimeter: jet::choose(a, arced, &arc.perimeter, &closed.perimeter)?,
        step: jet::choose(a, arced, &arc.step, &closed.step)?,
        valid: Some(valid),
    })
}

/// `Fr^2` and `Sf` jets of a section, with a mask of the rows whose area is positive.
pub(crate) struct Slopes<R: Runtime<DType = DType>> {
    pub(crate) froude: Jet<R>,
    pub(crate) friction: Jet<R>,
    pub(crate) valid: Tensor<R>,
}

/// `Fr^2 = F t / a^3` and `Sf = exp(ln K + (4 ln p - 10 ln a) / 3)`.
///
/// Rows whose area may be zero or negative take a unit area and report 0 in `valid`.
pub(crate) fn slopes<R>(
    a: &mut Arithmetic<'_, R>,
    params: &ProfileParams<R>,
    section: &ShapeJets<R>,
) -> ArithmeticResult<Slopes<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = params.rise.clone();
    let n = section.area.len();
    let base = jet::coefficient(a, &section.area, 0, &like)?;
    let valid = c.gt(base.lower(), &fill(&like, 0.0)?)?;
    let valid = match &section.valid {
        Some(held) => c.mul(&valid, held)?,
        None => valid,
    };
    let unit = Jet::constant(Rc::new(a.point(&fill(&like, 1.0)?)?), n);
    let area = jet::choose(a, &valid, &section.area, &unit)?;
    let square = jet::mul(a, &area, &area)?;
    let cube = jet::mul(a, &square, &area)?;
    let ratio = jet::div(a, &section.top, &cube)?;
    let froude_scale = a.bounds(&params.froude[0], &params.froude[1])?;
    let froude = jet::scale(a, &ratio, &froude_scale)?;

    let log_perimeter = jet::ln(a, &section.perimeter)?;
    let log_area = jet::ln(a, &area)?;
    let four = a.point(&fill(&like, 4.0)?)?;
    let ten = a.point(&fill(&like, 10.0)?)?;
    let raised = jet::scale(a, &log_perimeter, &four)?;
    let lowered = jet::scale(a, &log_area, &ten)?;
    let combined = jet::sub(a, &raised, &lowered)?;
    let third = jet::divide(a, &combined, 3.0)?;
    let offset = a.bounds(&params.log_friction[0], &params.log_friction[1])?;
    let exponent = jet::add(a, &Jet::constant(Rc::new(offset), n), &third)?;
    let friction = jet::exp(a, &exponent, &like)?;
    Ok(Slopes {
        froude,
        friction,
        valid,
    })
}

/// Integrand jet and a mask of the rows where every divisor is certified positive.
pub(crate) struct Integrand<R: Runtime<DType = DType>> {
    pub(crate) value: Jet<R>,
    pub(crate) valid: Tensor<R>,
    /// Lower bound of the signed denominator `mu (S - Sf)` over `v`, on rows inside `valid`.
    pub(crate) denominator: Tensor<R>,
}

/// Encloses the integrand jet with `n` terms over the coordinate enclosure `v`.
///
/// `bound`, when given, is a known lower bound of the signed denominator over `v`. It replaces
/// the interval lower bound where it is higher. Rows whose area or signed denominator is not
/// certified positive take harmless operands and report 0 in `valid`.
pub(crate) fn integrand<R>(
    a: &mut Arithmetic<'_, R>,
    params: &ProfileParams<R>,
    v: &Interval<R>,
    n: usize,
    bound: Option<&Tensor<R>>,
) -> ArithmeticResult<Integrand<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = params.rise.clone();
    let section = shape(a, &params.outline(), v, n)?;
    let Slopes {
        froude,
        friction,
        valid,
    } = slopes(a, params, &section)?;
    let unit = Jet::constant(Rc::new(a.point(&fill(&like, 1.0)?)?), n);
    let lambda = a.point(&params.numerator)?;
    let mu = a.point(&params.denominator)?;
    let subcritical = jet::sub(a, &unit, &froude)?;
    let numerator = jet::scale(a, &subcritical, &lambda)?;
    let slope = Jet::constant(Rc::new(a.point(&params.slope)?), n);
    let excess = jet::sub(a, &slope, &friction)?;
    let signed = jet::scale(a, &excess, &mu)?;
    let base = jet::coefficient(a, &signed, 0, &like)?;
    let (signed, lower, floor) = match bound {
        // A tighter c_0 still encloses the denominator over v, so the jet stays an enclosure.
        // A jet divides by c_0 four times. A c_0 that is positive only at rounding level would
        // overflow, so jets need it to clear the rounding level of its upper end.
        Some(known) => {
            let lower = c.maximum(base.lower(), known)?;
            let tight = a.bounds(&lower, &c.maximum(base.upper(), &lower)?)?;
            let floor = c.mul_scalar(base.upper(), f64::EPSILON)?;
            (signed.rebased(Rc::new(tight)), lower, floor)
        }
        None => (signed, base.lower().clone(), fill(&like, 0.0)?),
    };
    let positive = c.gt(&lower, &floor)?;
    let denominator = jet::choose(a, &positive, &signed, &unit)?;
    let ratio = jet::div(a, &numerator, &denominator)?;
    let stepped = jet::mul(a, &section.step, &ratio)?;
    let rise = a.point(&params.rise)?;
    Ok(Integrand {
        value: jet::scale(a, &stepped, &rise)?,
        valid: c.mul(&valid, &positive)?,
        denominator: lower,
    })
}

/// Lower bound of `G` over every `v` in `[lower, upper]`, or a nonpositive value when none holds.
///
/// `ds/dv` is concave on the circle and box domains, so its minimum sits at an end. On an arc
/// row `ds/dpsi = r_k sin psi`, with `sin` concave on `[0, pi]`, so it is at least the least
/// band radius times the smaller end sine, across junctions too. `Fr^2` falls with depth on
/// every convex section, because `A T' <= T^2` below the widest depth and `T' <= 0` above it.
/// So the numerator is smallest at the end nearest critical depth. `a` and `p` rise with `v`,
/// so `|S - Sf| <= |S| + K p(upper)^(4/3) / a(lower)^(10/3)`.
pub(crate) fn floor<R>(
    a: &mut Arithmetic<'_, R>,
    params: &ProfileParams<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = params.rise.clone();
    let zero = fill(&like, 0.0)?;
    let one = a.point(&fill(&like, 1.0)?)?;
    let low = a.point(lower)?;
    let high = a.point(upper)?;
    let outline = params.outline();
    let bottom = shape(a, &outline, &low, 1)?;
    let top = shape(a, &outline, &high, 1)?;
    let at_bottom = slopes(a, params, &bottom)?;
    let at_top = slopes(a, params, &top)?;
    let term = |a: &mut Arithmetic<'_, R>, jet: &Jet<R>| jet::coefficient(a, jet, 0, &like);

    let step_low = term(a, &bottom.step)?;
    let step_high = term(a, &top.step)?;
    let mut step = c.minimum(step_low.lower(), step_high.lower())?;
    if let Some(arcs) = &params.arcs {
        let [sine_low, _] = sin_cos_near(a, &low)?;
        let [sine_high, _] = sin_cos_near(a, &high)?;
        let sine = c.minimum(sine_low.lower(), sine_high.lower())?;
        let sine = a.bounds(&c.maximum(&sine, &zero)?, &c.maximum(&sine, &zero)?)?;
        let least = a.bounds(&arcs.least_radius, &arcs.least_radius)?;
        let arc = a.mul(&least, &sine)?;
        step = c.where_cond(&arcs.arced, arc.lower(), &step)?;
    }

    let froude_low = term(a, &at_bottom.froude)?;
    let froude_high = term(a, &at_top.froude)?;
    let sub = a.sub(&one, &froude_low)?;
    let sup = a.sub(&froude_high, &one)?;
    let rising = c.gt(&params.numerator, &zero)?;
    let numerator = c.where_cond(&rising, sub.lower(), sup.lower())?;

    let log_perimeter = term(a, &top.perimeter)?;
    let log_perimeter = a.log_positive(&log_perimeter)?;
    let area = term(a, &bottom.area)?;
    let positive = c.gt(area.lower(), &zero)?;
    let area = a.bounds(
        &c.where_cond(&positive, area.lower(), one.lower())?,
        &c.where_cond(&positive, area.upper(), one.upper())?,
    )?;
    let log_area = a.log_positive(&area)?;
    let four = a.point(&fill(&like, 4.0)?)?;
    let ten = a.point(&fill(&like, 10.0)?)?;
    let raised = a.mul(&four, &log_perimeter)?;
    let lowered = a.mul(&ten, &log_area)?;
    let combined = a.sub(&raised, &lowered)?;
    let third = a.div_positive(&combined, &fill(&like, 3.0)?)?;
    let offset = a.bounds(&params.log_friction[0], &params.log_friction[1])?;
    let exponent = a.add(&offset, &third)?;
    let friction = jet::exp_interval(a, &exponent)?;
    let magnitude = a.point(&c.abs(&params.slope)?)?;
    let excess = a.add(&magnitude, &friction)?;
    let largest = a.point(excess.upper())?;

    let usable = c.mul(
        &c.mul(&c.gt(&numerator, &zero)?, &c.gt(&step, &zero)?)?,
        &c.mul(&positive, &c.mul(&at_bottom.valid, &at_top.valid)?)?,
    )?;
    let numerator = a.point(&c.where_cond(&usable, &numerator, one.lower())?)?;
    let step = a.point(&c.where_cond(&usable, &step, one.lower())?)?;
    let rise = a.point(&params.rise)?;
    let product = a.mul(&step, &numerator)?;
    let scaled = a.mul(&rise, &product)?;
    let bound = a.div_positive_interval(&scaled, &largest)?;
    Ok(c.where_cond(&usable, bound.lower(), &zero)?)
}
