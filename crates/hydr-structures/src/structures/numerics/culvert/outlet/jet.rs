//! Truncated Taylor series with interval coefficients.
//!
//! A jet holds `c_k` for `k < n` with `f(xi + h) = sum c_k h^k + O(h^n)`. Evaluated with an
//! interval base, every `c_k` encloses `f^(k)(xi) / k!` for all `xi` in that interval, so
//! `24 c_4` bounds the fourth derivative over the interval. A `None` term is an exact zero.

use super::trig::{fill, negate};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps, CorrectlyRoundedOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};
use std::rc::Rc;

/// One shared Taylor coefficient.
pub(crate) type Term<R> = Rc<Interval<R>>;

/// Taylor coefficients `c_0 .. c_{n-1}`. `None` is an exact zero.
pub(crate) struct Jet<R: Runtime<DType = DType>> {
    terms: Vec<Option<Term<R>>>,
}

impl<R: Runtime<DType = DType>> Jet<R> {
    /// Jet of the given terms.
    pub(crate) fn new(terms: Vec<Option<Term<R>>>) -> Self {
        Self { terms }
    }

    /// `first + second h`, with `n` terms.
    pub(crate) fn linear(first: Term<R>, second: Option<Term<R>>, n: usize) -> Self {
        let mut terms: Vec<Option<Term<R>>> = (0..n).map(|_| None).collect();
        if let Some(slot) = terms.get_mut(0) {
            *slot = Some(first);
        }
        if let Some(slot) = terms.get_mut(1) {
            *slot = second;
        }
        Self { terms }
    }

    /// Constant `value`, with `n` terms.
    pub(crate) fn constant(value: Term<R>, n: usize) -> Self {
        Self::linear(value, None, n)
    }

    /// Term count `n`.
    pub(crate) fn len(&self) -> usize {
        self.terms.len()
    }

    /// Coefficient `c_k`, or `None` for an exact zero.
    pub(crate) fn term(&self, k: usize) -> Option<&Term<R>> {
        self.terms.get(k).and_then(Option::as_ref)
    }

    /// The jet with `c_0` replaced by `base`.
    pub(crate) fn rebased(&self, base: Term<R>) -> Self {
        let mut terms = self.terms.clone();
        if let Some(slot) = terms.get_mut(0) {
            *slot = Some(base);
        }
        Self { terms }
    }
}

/// Coefficient `c_k` with an exact zero materialized in the shape of `like`.
pub(crate) fn coefficient<R>(
    a: &mut Arithmetic<'_, R>,
    jet: &Jet<R>,
    k: usize,
    like: &Tensor<R>,
) -> ArithmeticResult<Term<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    match jet.term(k) {
        Some(value) => Ok(Rc::clone(value)),
        None => Ok(Rc::new(a.point(&fill(like, 0.0)?)?)),
    }
}

fn plus<R>(
    a: &mut Arithmetic<'_, R>,
    left: Option<Term<R>>,
    right: Option<&Term<R>>,
) -> ArithmeticResult<Option<Term<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok(match (left, right) {
        (None, None) => None,
        (Some(x), None) => Some(x),
        (None, Some(y)) => Some(Rc::clone(y)),
        (Some(x), Some(y)) => Some(Rc::new(a.add(&x, y)?)),
    })
}

/// `x - y` with exact zeros as `None`.
fn minus<R>(
    a: &mut Arithmetic<'_, R>,
    x: Option<&Term<R>>,
    y: Option<&Term<R>>,
) -> ArithmeticResult<Option<Term<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    Ok(match (x, y) {
        (None, None) => None,
        (Some(x), None) => Some(Rc::clone(x)),
        (None, Some(y)) => Some(Rc::new(negate(a, y)?)),
        (Some(x), Some(y)) => Some(Rc::new(a.sub(x, y)?)),
    })
}

/// `u + v`.
pub(crate) fn add<R>(a: &mut Arithmetic<'_, R>, u: &Jet<R>, v: &Jet<R>) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut terms = Vec::with_capacity(u.len());
    for k in 0..u.len() {
        terms.push(plus(a, u.term(k).cloned(), v.term(k))?);
    }
    Ok(Jet::new(terms))
}

/// `u - v`.
pub(crate) fn sub<R>(a: &mut Arithmetic<'_, R>, u: &Jet<R>, v: &Jet<R>) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut terms = Vec::with_capacity(u.len());
    for k in 0..u.len() {
        terms.push(minus(a, u.term(k), v.term(k))?);
    }
    Ok(Jet::new(terms))
}

/// `factor u` for an interval factor.
pub(crate) fn scale<R>(
    a: &mut Arithmetic<'_, R>,
    u: &Jet<R>,
    factor: &Interval<R>,
) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut terms = Vec::with_capacity(u.len());
    for k in 0..u.len() {
        terms.push(match u.term(k) {
            None => None,
            Some(x) => Some(Rc::new(a.mul(factor, x)?)),
        });
    }
    Ok(Jet::new(terms))
}

/// `u / divisor` for a represented positive divisor.
pub(crate) fn divide<R>(
    a: &mut Arithmetic<'_, R>,
    u: &Jet<R>,
    divisor: f64,
) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut terms = Vec::with_capacity(u.len());
    for k in 0..u.len() {
        terms.push(match u.term(k) {
            None => None,
            Some(x) => Some(Rc::new(a.div_positive(x, &fill(x.lower(), divisor)?)?)),
        });
    }
    Ok(Jet::new(terms))
}

/// Weighted Cauchy terms `sum_{j = first}^{k} weight(j) u_j v_{k-j}`, skipping exact zeros.
fn convolve<R>(
    a: &mut Arithmetic<'_, R>,
    u: &Jet<R>,
    v: &Jet<R>,
    k: usize,
    first: usize,
    weighted: bool,
) -> ArithmeticResult<Option<Term<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut sum = None;
    for j in first..=k {
        if let (Some(x), Some(y)) = (u.term(j), v.term(k - j)) {
            let mut product = a.mul(x, y)?;
            if weighted && j != 1 {
                let factor = a.point(&fill(product.lower(), j as f64)?)?;
                product = a.mul(&factor, &product)?;
            }
            sum = plus(a, sum, Some(&Rc::new(product)))?;
        }
    }
    Ok(sum)
}

/// `u v`.
pub(crate) fn mul<R>(a: &mut Arithmetic<'_, R>, u: &Jet<R>, v: &Jet<R>) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let mut terms = Vec::with_capacity(u.len());
    for k in 0..u.len() {
        terms.push(convolve(a, u, v, k, 0, false)?);
    }
    Ok(Jet::new(terms))
}

/// `u / v`. Every lower end of `v_0` must be positive, or the enclosing call rejects.
pub(crate) fn div<R>(a: &mut Arithmetic<'_, R>, u: &Jet<R>, v: &Jet<R>) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(base) = v.term(0).cloned() else {
        return Ok(Jet::new((0..u.len()).map(|_| None).collect()));
    };
    let mut quotient = Jet::new(Vec::with_capacity(u.len()));
    for k in 0..u.len() {
        // u_k = sum_{j=0}^{k} v_j w_{k-j}.
        let mut sum = None;
        for j in 1..=k {
            if let (Some(x), Some(y)) = (v.term(j), quotient.term(k - j)) {
                let product = Rc::new(a.mul(x, y)?);
                sum = plus(a, sum, Some(&product))?;
            }
        }
        let numerator = minus(a, u.term(k), sum.as_ref())?;
        let term = match numerator {
            None => None,
            Some(x) => Some(Rc::new(a.div_positive_interval(&x, &base)?)),
        };
        quotient.terms.push(term);
    }
    Ok(quotient)
}

/// `ln u`. Every lower end of `u_0` must be positive, or the enclosing call rejects.
pub(crate) fn ln<R>(a: &mut Arithmetic<'_, R>, u: &Jet<R>) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let Some(base) = u.term(0).cloned() else {
        return Ok(Jet::new((0..u.len()).map(|_| None).collect()));
    };
    let mut result = Jet::new(Vec::with_capacity(u.len()));
    result.terms.push(Some(Rc::new(a.log_positive(&base)?)));
    for k in 1..u.len() {
        // k w_k u_0 = k u_k - sum_{j=1}^{k-1} j w_j u_{k-j}.
        let sum = convolve(a, &result, u, k, 1, true)?;
        let scaled = match u.term(k) {
            None => None,
            Some(x) => {
                let factor = a.point(&fill(x.lower(), k as f64)?)?;
                Some(Rc::new(a.mul(&factor, x)?))
            }
        };
        let numerator = minus(a, scaled.as_ref(), sum.as_ref())?;
        let term = match numerator {
            None => None,
            Some(x) => {
                let ratio = a.div_positive_interval(&x, &base)?;
                Some(Rc::new(
                    a.div_positive(&ratio, &fill(ratio.lower(), k as f64)?)?,
                ))
            }
        };
        result.terms.push(term);
    }
    Ok(result)
}

/// Encloses `exp x` from correctly rounded endpoint values stepped one F64 outward.
pub(crate) fn exp_interval<R>(
    a: &mut Arithmetic<'_, R>,
    x: &Interval<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = fill(x.lower(), 0.0)?;
    let lower = c.maximum(&c.next_down(&c.cr_exp(x.lower())?)?, &zero)?;
    let upper = c.next_up(&c.cr_exp(x.upper())?)?;
    a.bounds(&lower, &upper)
}

/// `exp u`.
pub(crate) fn exp<R>(
    a: &mut Arithmetic<'_, R>,
    u: &Jet<R>,
    like: &Tensor<R>,
) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let base = coefficient(a, u, 0, like)?;
    let mut result = Jet::new(Vec::with_capacity(u.len()));
    result.terms.push(Some(Rc::new(exp_interval(a, &base)?)));
    for k in 1..u.len() {
        // k w_k = sum_{j=1}^{k} j u_j w_{k-j}.
        let sum = convolve(a, u, &result, k, 1, true)?;
        let term = match sum {
            None => None,
            Some(x) => Some(Rc::new(a.div_positive(&x, &fill(x.lower(), k as f64)?)?)),
        };
        result.terms.push(term);
    }
    Ok(result)
}

/// Takes `on` where `mask` is 1 and `off` where it is 0, term by term.
pub(crate) fn choose<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    on: &Jet<R>,
    off: &Jet<R>,
) -> ArithmeticResult<Jet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let zero = fill(mask, 0.0)?;
    let ends = |value: Option<&Term<R>>| match value {
        Some(i) => (i.lower().clone(), i.upper().clone()),
        None => (zero.clone(), zero.clone()),
    };
    let mut terms = Vec::with_capacity(on.len());
    for k in 0..on.len() {
        terms.push(match (on.term(k), off.term(k)) {
            (None, None) => None,
            (x, y) => {
                let (on_lower, on_upper) = ends(x);
                let (off_lower, off_upper) = ends(y);
                let lower = c.where_cond(mask, &on_lower, &off_lower)?;
                let upper = c.where_cond(mask, &on_upper, &off_upper)?;
                Some(Rc::new(a.bounds(&lower, &upper)?))
            }
        });
    }
    Ok(Jet::new(terms))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureError;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    type Terms = Vec<[f64; 2]>;

    fn read(jet: &Jet<CpuRuntime>) -> numr::error::Result<Terms> {
        (0..jet.len())
            .map(|k| match jet.term(k) {
                None => Ok([0.0, 0.0]),
                Some(x) => Ok([
                    x.lower().try_to_vec::<f64>()?[0],
                    x.upper().try_to_vec::<f64>()?[0],
                ]),
            })
            .collect()
    }

    #[test]
    fn jets_enclose_taylor_coefficients_of_closed_forms() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let at = Tensor::<CpuRuntime>::from_slice(&[2.0], &[1], &device)?;
        let [quotient, logarithm, exponential] =
            with_interval_arithmetic(&client, DType::F64, "jets", |a| {
                let x = Rc::new(a.point(&at)?);
                let one = Rc::new(a.point(&fill(&at, 1.0)?)?);
                let variable = Jet::linear(x, Some(Rc::clone(&one)), 5);
                let unit = Jet::constant(one, 5);
                let reciprocal = div(a, &unit, &variable)?;
                let logarithm = ln(a, &variable)?;
                let exponential = exp(a, &variable, &at)?;
                Ok([read(&reciprocal)?, read(&logarithm)?, read(&exponential)?])
            })?;
        // 1/x at 2: (-1)^k / 2^(k+1). ln x: ln 2, then (-1)^(k+1) / (k 2^k). exp x: e^2 / k!.
        let e2 = 2.0_f64.exp();
        let mut factorial = 1.0;
        let mut power = 1.0;
        let terms = quotient.iter().zip(&exponential).zip(&logarithm);
        for (k, ((reciprocal, growth), log)) in terms.enumerate() {
            if k > 0 {
                factorial *= k as f64;
                power *= 2.0;
            }
            let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
            for ([low, high], value) in [
                (*reciprocal, sign / (2.0 * power)),
                (*growth, e2 / factorial),
            ] {
                let slack = value.abs() * 1e-14;
                assert!(low <= value + slack && value - slack <= high, "k={k}");
            }
            if k > 0 {
                let [low, high] = *log;
                let value = -sign / (k as f64 * power);
                assert!(low <= value + 1e-15 && value - 1e-15 <= high, "k={k}");
            }
        }
        assert_eq!(quotient.len(), 5);
        Ok(())
    }
}
