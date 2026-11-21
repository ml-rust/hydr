//! Circle closed forms for `D/16 <= y < D` through interval `acos`.
//!
//! With `u = (D - 2y)/D`, `phi = acos(u)` and `sigma = sin(phi) = T/D`:
//! `A = (D^2/4)(phi - sigma u)`, `P = D phi`, `R = A/P` and
//! `M = (D^3/24)(3 sigma - sigma^3 - 3 phi u)`.

use super::{Quantities, top_width};
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Encloses the free-surface geometry at depths `0 <= depth <= D` of circle rows.
pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let rise = &barrel.rise;
    let y = a.point(depth)?;
    let d = a.point(rise)?;
    let top_width = top_width(a, barrel, &y, &d)?;
    let sigma = a.div_positive(&top_width, rise)?;

    let two = a.point(&barrel.constant(2.0)?)?;
    let twice = a.mul(&two, &y)?;
    let numerator = a.sub(&d, &twice)?;
    let cosine = a.div_positive(&numerator, rise)?;
    // The exact u lies in [-1, 1]. Clamping removes only outward padding past the domain.
    let lower = c.maximum(cosine.lower(), &barrel.constant(-1.0)?)?;
    let upper = c.minimum(cosine.upper(), &barrel.one)?;
    let cosine = a.bounds(&lower, &upper)?;
    let angle = a.acos(&cosine)?;

    let square = a.mul(&d, &d)?;
    let product = a.mul(&sigma, &cosine)?;
    let segment = a.sub(&angle, &product)?;
    let scaled = a.mul(&square, &segment)?;
    let quarter = a.point(&barrel.constant(0.25)?)?;
    let area = a.mul(&scaled, &quarter)?;
    let perimeter = a.mul(&d, &angle)?;
    let radius = a.div_positive_interval(&area, &perimeter)?;

    let three = a.point(&barrel.constant(3.0)?)?;
    let triple = a.mul(&three, &sigma)?;
    let sigma_square = a.mul(&sigma, &sigma)?;
    let cube = a.mul(&sigma_square, &sigma)?;
    let angle_cosine = a.mul(&angle, &cosine)?;
    let triple_angle_cosine = a.mul(&three, &angle_cosine)?;
    let partial = a.sub(&triple, &cube)?;
    let bracket = a.sub(&partial, &triple_angle_cosine)?;
    let cubed = a.mul(&square, &d)?;
    let volume = a.mul(&cubed, &bracket)?;
    let moment = a.div_positive(&volume, &barrel.constant(24.0)?)?;

    Ok(Quantities {
        area,
        top_width,
        perimeter,
        radius,
        moment,
    })
}
