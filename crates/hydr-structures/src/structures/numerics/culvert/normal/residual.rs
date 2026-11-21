//! Normal-depth residual `r_n = 5 ln A - 2 ln P - (3 ln Q + 3 ln n - 1.5 ln S)` in `s = y/D`.
//!
//! `r_n = 3 ln K - 3 ln(Q n / sqrt(S))` with the conveyance `K = A^(5/3) / P^(2/3)`.
//!
//! - A circle reads `A = D^2 a(s)` and `P = D p(s)` from unit-diameter geometry, so
//!   `r_n = 5 ln a(s) - 2 ln p(s) - (3 ln Q + 3 ln n - 1.5 ln S - 8 ln D)`.
//! - A box has `A = B D s` and `P = B + 2 D s`, so
//!   `r_n = 5 ln s - 2 ln(B + 2 D s) - (3 ln Q + 3 ln n - 1.5 ln S - 5 ln B - 5 ln D)`.
//!
//! Both rise in `s` on their domains. The bracketed term is the row target, fixed per query.
//! Unit geometry keeps the area of a tiny rise away from underflow.

use crate::structures::numerics::culvert::{Arithmetic, BarrelClient, choose, section};
use crate::structures::tensor::culvert::TensorBarrelContext;
use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval};

/// Encloses the row target `3 ln Q + 3 ln n - 1.5 ln S` minus `8 ln D` (circle) or
/// `5 ln B + 5 ln D` (box).
///
/// `discharge`, `roughness` and `slope` are `[m]` tensors. Rows with zero discharge take a unit
/// discharge and rows with a nonpositive slope a unit slope, so the caller must classify them
/// before reading the target. Every roughness must be finite and positive.
pub(crate) fn target<R>(
    a: &mut Arithmetic<'_, R>,
    barrel: &TensorBarrelContext<R>,
    discharge: &Tensor<R>,
    roughness: &Tensor<R>,
    slope: &Tensor<R>,
) -> ArithmeticResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R>,
{
    let c = a.client();
    let wet = c.gt(discharge, &barrel.zero)?;
    let sloped = c.gt(slope, &barrel.zero)?;
    let q = a.point(&c.where_cond(&wet, discharge, &barrel.one)?)?;
    let ln_q = a.log_positive(&q)?;
    let n = a.point(roughness)?;
    let ln_n = a.log_positive(&n)?;
    let s = a.point(&c.where_cond(&sloped, slope, &barrel.one)?)?;
    let ln_s = a.log_positive(&s)?;
    let d = a.point(&barrel.rise)?;
    let ln_d = a.log_positive(&d)?;
    let b = a.point(&barrel.span)?;
    let ln_b = a.log_positive(&b)?;
    let three = a.point(&barrel.constant(3.0)?)?;
    let five = a.point(&barrel.constant(5.0)?)?;
    let eight = a.point(&barrel.constant(8.0)?)?;
    let three_halves = a.point(&barrel.constant(1.5)?)?;

    let thrice_q = a.mul(&three, &ln_q)?;
    let thrice_n = a.mul(&three, &ln_n)?;
    let resistance = a.add(&thrice_q, &thrice_n)?;
    let drive = a.mul(&three_halves, &ln_s)?;
    let flow = a.sub(&resistance, &drive)?;
    let circle = a.mul(&eight, &ln_d)?;
    let both = a.add(&ln_b, &ln_d)?;
    let rectangle = a.mul(&five, &both)?;
    let scale = choose(a, &barrel.circular, &circle, &rectangle)?;
    a.sub(&flow, &scale)
}

/// Encloses `r_n` at the depth fractions `fraction` (`[m]`) against the stored row `target`.
///
/// `unit` holds unit-diameter circles of the same width. Every fraction must lie in
/// `[2^-30, 0.93818]` on circle rows and in `[2^-30, 1]` on box rows. Other fractions
/// reject the enclosing call at its completion check, or leave the rising limb.
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
    let two = a.point(&barrel.constant(2.0)?)?;
    let five = a.point(&barrel.constant(5.0)?)?;

    // Each shape runs on every row. The other shape's rows take the harmless fraction 1/2.
    let circle_fraction = c.where_cond(&barrel.circular, fraction, &half)?;
    let geometry = section::at_depth(a, unit, &circle_fraction)?.geometry;
    let ln_area = a.log_positive(&geometry.area)?;
    let ln_perimeter = a.log_positive(&geometry.perimeter)?;
    let area_term = a.mul(&five, &ln_area)?;
    let perimeter_term = a.mul(&two, &ln_perimeter)?;
    let circle = a.sub(&area_term, &perimeter_term)?;

    let box_fraction = c.where_cond(&barrel.boxed, fraction, &half)?;
    let s = a.point(&box_fraction)?;
    let ln_s = a.log_positive(&s)?;
    let d = a.point(&barrel.rise)?;
    let b = a.point(&barrel.span)?;
    let depth = a.mul(&d, &s)?;
    let walls = a.mul(&two, &depth)?;
    let perimeter = a.add(&b, &walls)?;
    let ln_perimeter = a.log_positive(&perimeter)?;
    let area_term = a.mul(&five, &ln_s)?;
    let perimeter_term = a.mul(&two, &ln_perimeter)?;
    let rectangle = a.sub(&area_term, &perimeter_term)?;

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

    /// Residual `[lower, upper]` per row at the floor and at the normal domain top.
    fn ends(
        barrels: Vec<CulvertBarrel>,
        discharges: &[f64],
        slope: f64,
    ) -> Result<[[Vec<f64>; 2]; 2], StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrel = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        let unit = barrel.unit_circles()?;
        let discharge = Tensor::<CpuRuntime>::from_slice(discharges, &[discharges.len()], &device)?;
        let roughness = barrel.constant(0.013)?;
        let slope = barrel.constant(slope)?;
        let floor = barrel.constant(CRITICAL_FLOOR)?;
        let [bottom, top] = with_interval_arithmetic(&client, DType::F64, "ends", |a| {
            let row_target = target(a, &barrel, &discharge, &roughness, &slope)?;
            let bottom = residual(a, &barrel, &unit, &row_target, &floor)?;
            let top = residual(a, &barrel, &unit, &row_target, &barrel.normal_top)?;
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
        // Interior roots, then a tiny discharge, then a box and a circle past capacity.
        let barrels = vec![circle, rectangle, circle, rectangle, rectangle, circle];
        let discharges = [0.5, 1.5, 1e-30, 1e-30, 20.0, 20.0];
        let [[bottom_lower, bottom_upper], [top_lower, top_upper]] =
            ends(barrels, &discharges, 0.01)?;
        let bottom = bottom_lower.iter().zip(&bottom_upper);
        let top = top_lower.iter().zip(&top_upper);
        for (row, ((&bl, &bu), (&tl, &tu))) in bottom.zip(top).enumerate() {
            assert!(bl <= bu && tl <= tu, "row={row}");
            assert!(bu < tl, "row={row}: the residual must rise");
            match row {
                0 | 1 => assert!(bu < 0.0 && 0.0 < tl, "row={row}: interior root"),
                2 | 3 => assert!(bl > 0.0, "row={row}: root below the floor"),
                _ => assert!(tu < 0.0, "row={row}: target above the top"),
            }
        }
        Ok(())
    }
}
