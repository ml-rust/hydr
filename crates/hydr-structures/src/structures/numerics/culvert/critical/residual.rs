//! Critical-depth residual `r_c = 3 ln A - ln T - (2 ln Q - ln g)` in `s = y/D`.
//!
//! - A circle reads `A = D^2 a(s)` and `T = D t(s)` from unit-diameter geometry, so
//!   `r_c = 3 ln a(s) - ln t(s) - (2 ln Q - ln g - 5 ln D)`.
//! - A box has `A^3 / T = B^2 D^3 s^3`, so `r_c = 3 ln s - (2 ln Q - ln g - 2 ln B - 3 ln D)`.
//!
//! Both rise in `s`. The bracketed term is the row target, fixed per query.
//! Unit geometry keeps the area of a tiny rise away from underflow.

use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose, section};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the row target `2 ln Q - ln g - 5 ln D` (circle) or `- 2 ln B - 3 ln D` (box).
///
/// `discharge` and `gravity` are `[m]` tensors. Rows with zero discharge take a unit discharge,
/// so the caller must classify them before reading the target.
pub(crate) fn target<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    gravity: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let wet = c.gt(discharge, &barrel.zero)?;
    let q = a.point(&c.where_cond(&wet, discharge, &barrel.one)?)?;
    let ln_q = a.log_positive(&q)?;
    let g = a.point(gravity)?;
    let ln_g = a.log_positive(&g)?;
    let d = a.point(&barrel.rise)?;
    let ln_d = a.log_positive(&d)?;
    let b = a.point(&barrel.span)?;
    let ln_b = a.log_positive(&b)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let three = a.point(&barrel.constant(3.0)?)?;
    let five = a.point(&barrel.constant(5.0)?)?;

    let twice_q = a.mul(&two, &ln_q)?;
    let flow = a.sub(&twice_q, &ln_g)?;
    let circle = a.mul(&five, &ln_d)?;
    let twice_b = a.mul(&two, &ln_b)?;
    let thrice_d = a.mul(&three, &ln_d)?;
    let rectangle = a.add(&twice_b, &thrice_d)?;
    let scale = choose(a, &barrel.circular, &circle, &rectangle)?;
    a.sub(&flow, &scale)
}

/// Encloses `r_c` at the depth fractions `fraction` (`[m]`) against the stored row `target`.
///
/// `unit` holds unit-diameter circles of the same width. Every fraction must lie in
/// `[2^-30, 1 - 2^-53]` on circle rows and in `[2^-30, 1]` on box rows. Other fractions
/// reject the enclosing call at its completion check.
pub(crate) fn residual<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    target: &Interval<R>,
    fraction: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let half = barrel.constant(0.5)?;
    let three = a.point(&barrel.constant(3.0)?)?;

    // Each shape runs on every row. The other shape's rows take the harmless fraction 1/2.
    let circle_fraction = c.where_cond(&barrel.circular, fraction, &half)?;
    let geometry = section::at_depth(a, unit, &circle_fraction)?.geometry;
    let ln_area = a.log_positive(&geometry.area)?;
    let ln_top = a.log_positive(&geometry.top_width)?;
    let thrice_area = a.mul(&three, &ln_area)?;
    let circle = a.sub(&thrice_area, &ln_top)?;

    let box_fraction = c.where_cond(&barrel.boxed, fraction, &half)?;
    let s = a.point(&box_fraction)?;
    let ln_s = a.log_positive(&s)?;
    let rectangle = a.mul(&three, &ln_s)?;

    let shape = choose(a, &barrel.circular, &circle, &rectangle)?;
    a.sub(&shape, target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::numerics::culvert::constants::CRITICAL_FLOOR;
    use crate::structures::{CulvertBarrel, StructureError};
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    use solvr::common::arithmetic::with_interval_arithmetic;

    /// Residual `[lower, upper]` per row at the floor and at the domain top.
    fn ends(
        barrels: Vec<CulvertBarrel>,
        discharges: &[f64],
    ) -> Result<[[Vec<f64>; 2]; 2], StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrel = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        let unit = barrel.unit_circles()?;
        let discharge = Tensor::<CpuRuntime>::from_slice(discharges, &[discharges.len()], &device)?;
        let gravity = barrel.constant(9.81)?;
        let floor = barrel.constant(CRITICAL_FLOOR)?;
        let [bottom, top] = with_interval_arithmetic(&client, DType::F64, "ends", |a| {
            let row_target = target(a, &barrel, &discharge, &gravity)?;
            let bottom = residual(a, &barrel, &unit, &row_target, &floor)?;
            let top = residual(a, &barrel, &unit, &row_target, &barrel.critical_top)?;
            Ok([bottom, top])
        })?;
        let pair = |i: &Interval<CpuRuntime>| -> Result<[Vec<f64>; 2], StructureError> {
            Ok([
                i.lower().try_to_vec::<f64>()?,
                i.upper().try_to_vec::<f64>()?,
            ])
        };
        Ok([pair(&bottom)?, pair(&top)?])
    }

    #[test]
    fn residual_signs_bracket_interior_roots_and_flag_the_ends() -> Result<(), StructureError> {
        let circle = CulvertBarrel::circular(1.0)?;
        let rectangle = CulvertBarrel::rectangular_box(2.5, 0.8)?;
        // Interior roots near s = 1/2, then a tiny discharge, then a box past its capacity.
        let barrels = vec![circle, rectangle, circle, rectangle, rectangle];
        let discharges = [0.77, 2.8, 1e-30, 1e-30, 6.0];
        let [[bottom_lower, bottom_upper], [top_lower, top_upper]] = ends(barrels, &discharges)?;
        for row in 0..2 {
            assert!(bottom_upper[row] < 0.0 && 0.0 < top_lower[row], "row={row}");
        }
        for (row, lower) in bottom_lower.iter().enumerate().take(4).skip(2) {
            assert!(*lower > 0.0, "row={row}");
        }
        assert!(top_upper[4] < 0.0);
        for row in 0..discharges.len() {
            assert!(bottom_lower[row] <= bottom_upper[row] && top_lower[row] <= top_upper[row]);
            assert!(
                bottom_upper[row] < top_lower[row],
                "row={row}: the residual must rise"
            );
        }
        Ok(())
    }
}
