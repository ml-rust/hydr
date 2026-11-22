//! Section jets of arc rows in the normal angle, from the shared sine and cosine jets.

use super::{ArcShape, MarchBand, hull};
use crate::structures::numerics::culvert::outlet::integrand::ShapeJets;
use crate::structures::numerics::culvert::outlet::jet::{self, Jet, Term};
use crate::structures::numerics::culvert::outlet::trig::{fill, negate};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::Bounds;
use numr::dtype::DType;
use numr::ops::{BinaryOps, NextRepresentableOps};
use numr::runtime::Runtime;
use solvr::common::arithmetic::{ArithmeticResult, Interval};
use std::rc::Rc;

/// Jets of `sin psi`, `cos psi` and `psi - sin psi cos psi` over the coordinate enclosure.
pub(crate) struct Trig<'t, R: Runtime<DType = DType>> {
    pub(crate) sine: &'t Jet<R>,
    pub(crate) cosine: &'t Jet<R>,
    pub(crate) segment: &'t Jet<R>,
    /// `[psi, sin psi, cos psi]` at each end of a derivative query's coordinate enclosure.
    pub(crate) ends: Option<[[Interval<R>; 3]; 2]>,
}

fn enter<R>(a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    a.bounds(lower, upper)
}

/// Jets of one band over `psi` with `n` terms.
fn band_jets<R>(
    a: &mut Arithmetic<'_, R>,
    band: &MarchBand<R>,
    trig: &Trig<'_, R>,
    psi: &Interval<R>,
    n: usize,
) -> ArithmeticResult<ShapeJets<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = psi.lower();
    let x = enter(a, &band.centre_x)?;
    let r = Rc::new(enter(a, &band.radius)?);
    let two = a.point(&fill(like, 2.0)?)?;
    let twice_x = Rc::new(a.mul(&two, &x)?);
    let twice_r = Rc::new(a.mul(&two, &r)?);
    let reach = jet::scale(a, trig.sine, &twice_r)?;
    let top = jet::add(a, &Jet::constant(twice_x, n), &reach)?;
    let step = jet::scale(a, trig.sine, &r)?;

    let lever = a.mul(&twice_r, &x)?;
    let lever = negate(a, &lever)?;
    let square = a.mul(&r, &r)?;
    let offset = Rc::new(enter(a, &band.area_offset)?);
    let area = match &trig.ends {
        None => {
            let tilt = jet::scale(a, trig.cosine, &lever)?;
            let cap = jet::scale(a, trig.segment, &square)?;
            let walls = jet::add(a, &tilt, &cap)?;
            jet::add(a, &Jet::constant(offset, n), &walls)?
        }
        Some(ends) => {
            // Over a wide panel the wall terms of the closed form cancel. So the area rises
            // from its end values, and its higher terms integrate `da/dpsi = t ds/dpsi`.
            let mut values = Vec::with_capacity(2);
            for [point, sine, cosine] in ends {
                let tilt = a.mul(&lever, cosine)?;
                let product = a.mul(sine, cosine)?;
                let segment = a.sub(point, &product)?;
                let cap = a.mul(&square, &segment)?;
                let walls = a.add(&tilt, &cap)?;
                values.push(a.add(&offset, &walls)?);
            }
            let (Some(low), Some(high)) = (values.first(), values.last()) else {
                return Err(numr::error::Error::InvalidArgument {
                    arg: "culvert outlet arc jets",
                    reason: "no panel ends: report this runtime".to_string(),
                }
                .into());
            };
            let base = a.bounds(low.lower(), &c.maximum(high.upper(), low.lower())?)?;
            let rate = jet::mul(a, &top, &step)?;
            let mut terms = vec![Some(Rc::new(base))];
            for k in 1..n {
                terms.push(match rate.term(k - 1) {
                    None => None,
                    Some(term) => Some(Rc::new(a.div_positive(term, &fill(like, k as f64)?)?)),
                });
            }
            Jet::new(terms)
        }
    };

    // The angle from the band start stays at or above zero on the band.
    let zero = fill(like, 0.0)?;
    let lower = c.maximum(&c.sub(psi.lower(), &band.start_angle[1])?, &zero)?;
    let upper = c.maximum(&c.sub(psi.upper(), &band.start_angle[0])?, &zero)?;
    // The differences round to nearest, so step each one F64 outward.
    let turned = a.bounds(&lower, &upper)?;
    let turned = widen(a, &turned)?;
    let start = enter(a, &band.start_perimeter)?;
    let arc = a.mul(&twice_r, &turned)?;
    let wetted = Rc::new(a.add(&start, &arc)?);
    let perimeter = Jet::linear(wetted, Some(twice_r), n);
    Ok(ShapeJets {
        area,
        top,
        perimeter,
        step,
        valid: None,
    })
}

/// `x` widened by one F64 at each end, with the lower end kept at or above zero.
fn widen<R>(a: &mut Arithmetic<'_, R>, x: &Interval<R>) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = fill(x.lower(), 0.0)?;
    let lower = c.maximum(&c.next_down(x.lower())?, &zero)?;
    let upper = c.next_up(x.upper())?;
    a.bounds(&lower, &upper)
}

/// Value-only jet of the hull of three bands' `c_0` in `members`.
fn hull_jet<R>(
    a: &mut Arithmetic<'_, R>,
    members: &[numr::tensor::Tensor<R>; 3],
    jets: [&Jet<R>; 3],
    like: &numr::tensor::Tensor<R>,
) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let [j0, j1, j2] = jets;
    let v0: Term<R> = jet::coefficient(a, j0, 0, like)?;
    let v1 = jet::coefficient(a, j1, 0, like)?;
    let v2 = jet::coefficient(a, j2, 0, like)?;
    let value = hull(a, members, [&v0, &v1, &v2])?;
    Ok(Jet::new(vec![Some(Rc::new(value))]))
}

/// Section jets of arc rows over `psi` with `n` terms.
///
/// One band per row gives its jets. With three bands, a value query (`n = 1`) takes the hull
/// of every band that may hold `psi`, and a derivative query the one band that holds all of
/// `psi`. A derivative query across a junction reports 0 in `valid`.
pub(crate) fn shape_jets<R>(
    a: &mut Arithmetic<'_, R>,
    arcs: &ArcShape<R>,
    trig: &Trig<'_, R>,
    psi: &Interval<R>,
    n: usize,
) -> ArithmeticResult<ShapeJets<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let like = psi.lower();
    let mut per_band = Vec::with_capacity(arcs.bands.len());
    for band in &arcs.bands {
        per_band.push(band_jets(a, band, trig, psi, n)?);
    }
    let one = fill(like, 1.0)?;
    let [first, second, third] = match <[ShapeJets<R>; 3]>::try_from(per_band) {
        Ok(three) => three,
        Err(mut single) => {
            return match single.pop() {
                Some(jets) => Ok(ShapeJets {
                    valid: Some(one),
                    ..jets
                }),
                None => Err(numr::error::Error::InvalidArgument {
                    arg: "culvert outlet arc bands",
                    reason: "no band to evaluate: report this runtime".to_string(),
                }
                .into()),
            };
        }
    };
    if n == 1 {
        let members = arcs.members(c, psi.lower(), psi.upper())?;
        let pick = |a: &mut Arithmetic<'_, R>, f: fn(&ShapeJets<R>) -> &Jet<R>| {
            hull_jet(a, &members, [f(&first), f(&second), f(&third)], like)
        };
        return Ok(ShapeJets {
            area: pick(a, |s| &s.area)?,
            top: pick(a, |s| &s.top)?,
            perimeter: pick(a, |s| &s.perimeter)?,
            step: pick(a, |s| &s.step)?,
            valid: Some(one),
        });
    }
    let [low, middle, high] = arcs.singles(c, psi.lower(), psi.upper())?;
    let valid = c.maximum(&c.maximum(&low, &middle)?, &high)?;
    let pick = |a: &mut Arithmetic<'_, R>, f: fn(&ShapeJets<R>) -> &Jet<R>| {
        let upper = jet::choose(a, &middle, f(&second), f(&third))?;
        jet::choose(a, &low, f(&first), &upper)
    };
    Ok(ShapeJets {
        area: pick(a, |s| &s.area)?,
        top: pick(a, |s| &s.top)?,
        perimeter: pick(a, |s| &s.perimeter)?,
        step: pick(a, |s| &s.step)?,
        valid: Some(valid),
    })
}
