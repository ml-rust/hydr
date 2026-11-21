//! Box closed forms for `0 <= y < D`: `A = B y`, `T = B`, `P = B + 2y`, `M = B y^2 / 2`.

use super::Quantities;
use crate::structures::numerics::culvert::{Arithmetic, BarrelClient};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Encloses the free-surface geometry at nonnegative depths of box rows.
pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    depth: &Tensor<R>,
) -> ArithmeticResult<Quantities<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let y = a.point(depth)?;
    let span = a.point(&barrel.span)?;
    let area = a.mul(&span, &y)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let twice = a.mul(&two, &y)?;
    let perimeter = a.add(&span, &twice)?;
    let radius = a.div_positive_interval(&area, &perimeter)?;
    let half = a.point(&barrel.constant(0.5)?)?;
    let first = a.mul(&area, &y)?;
    let moment = a.mul(&first, &half)?;
    Ok(Quantities {
        area,
        top_width: span,
        perimeter,
        radius,
        moment,
    })
}
