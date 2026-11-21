//! Closed-form sensitivity of inlet control and of a drowned full barrel.

use super::super::Selected;
use super::super::inlet::InletDischarge;
use super::super::lanes::Lanes;
use super::reciprocal;
use crate::structures::CulvertClient;
use crate::structures::numerics::culvert::{Arithmetic, choose};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Inlet bound `1 / E` on `wanted` lanes, a harmless value elsewhere.
pub(super) fn inlet_sensitivity<R>(
    a: &mut Arithmetic<'_, R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
    wanted: &Tensor<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let coefficients = &rows.inlet;
    let c = a.client();
    let one = &barrel.one;
    let lower = a.point(&inlet.bounds[0])?;
    let scale = a.bounds(&coefficients.scale[0], &coefficients.scale[1])?;
    let x = a.div_positive_interval(&lower, &scale)?;
    let x_low = x.lower().clone();
    let wet = c.gt(&x_low, &barrel.zero)?;
    let x_point = a.point(&c.where_cond(&wet, &x_low, one)?)?;
    let d = a.point(&barrel.rise)?;
    let m = a.point(&coefficients.m)?;
    let k = a.point(&coefficients.k)?;

    // Form 2: `E = M HW`.
    let h = a.point(&lanes.headwater)?;
    let form_two = a.mul(&m, &h)?;
    // Form 1: `E >= M D K X^M`.
    let ln_x = a.log_positive(&x_point)?;
    let ln_power = a.mul(&m, &ln_x)?;
    let power = super::super::exp(a, &ln_power)?;
    let gain = a.mul(&k, &power)?;
    let scaled = a.mul(&d, &gain)?;
    let form_one = a.mul(&m, &scaled)?;
    let free = choose(a, &coefficients.form_one, &form_one, &form_two)?;
    // Transition: `E = 2 (HW_4 - HW_3.5) X`.
    let low = a.bounds(&coefficients.low[0], &coefficients.low[1])?;
    let high = a.bounds(&coefficients.high[0], &coefficients.high[1])?;
    let rise = a.sub(&high, &low)?;
    let two = a.point(&barrel.constant(2.0)?)?;
    let doubled = a.mul(&two, &rise)?;
    let middle = a.mul(&doubled, &x_point)?;
    // Submerged: `E = 2 D c X^2`.
    let square = a.mul(&x_point, &x_point)?;
    let coefficient = a.point(&coefficients.c)?;
    let product = a.mul(&coefficient, &square)?;
    let depth = a.mul(&d, &product)?;
    let full = a.mul(&two, &depth)?;

    let mut e = barrel.constant(f64::INFINITY)?;
    for (mask, value) in [
        (&inlet.bands.unsubmerged, &free),
        (&inlet.bands.transition, &middle),
        (&inlet.bands.submerged, &full),
    ] {
        e = c.where_cond(mask, &c.minimum(&e, value.lower())?, &e)?;
    }
    // A zero discharge bound leaves only `E = M HW` of a form-2 lane, and no bound on form 1.
    let bare = c.where_cond(&coefficients.form_one, &barrel.zero, form_two.lower())?;
    let e = c.where_cond(&wet, &e, &bare)?;
    let e = c.where_cond(wanted, &e, one)?;
    reciprocal(a, &e)
}

/// Drowned full-barrel bound `1 / (2 dH)` on `wanted` lanes, a harmless value elsewhere.
pub(super) fn full_sensitivity<R>(
    a: &mut Arithmetic<'_, R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    wanted: &Tensor<R>,
) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = a.client();
    let gap = c.where_cond(wanted, &lanes.gap, &barrel.one)?;
    let doubled = a.point(&c.mul_scalar(&gap, 2.0)?)?;
    reciprocal(a, doubled.lower())
}

#[cfg(test)]
mod tests {
    use super::super::fixture::{check, culvert, level};
    use crate::structures::numerics::sensitivity::fixture::{Lane, TestResult, grid, lanes};
    use crate::structures::{CulvertBarrel, CulvertInlet, CulvertInletForm, DirectionPolicy};

    #[test]
    fn inlet_sensitivity_bounds_unsubmerged_transition_and_submerged_heads() -> TestResult {
        // The steep explicit inlet governs at every head. Its anchors sit at
        // `HW = D K 3.5^M` and `D (16 c + Y)` above the invert at zero.
        let inlet = CulvertInlet::explicit(CulvertInletForm::Two, [2.0, 0.6, 0.5, 0.0], false)?;
        let law = culvert(
            level(CulvertBarrel::rectangular_box(2.0, 1.0)?, (10.0, 0.012)),
            inlet,
            false,
        )?;
        let low = 2.0 * 3.5_f64.powf(0.6);
        // The receiver is dry, so one lane per donor stage suffices.
        let cases: Vec<Lane> = lanes(&grid(0.05, 10.0, 40), &[0.0, low, 8.0], [true, false])
            .into_iter()
            .filter(|lane| lane.stage[1] == lane.stage[0])
            .collect();
        let flowing = check(&law, DirectionPolicy::AtoBOnly, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }

    #[test]
    fn drowned_full_barrel_sensitivity_is_half_over_the_head_difference() -> TestResult {
        let law = culvert(
            level(CulvertBarrel::circular(1.0)?, (200.0, 0.03)),
            CulvertInlet::hds5(1)?,
            true,
        )?;
        let cases = lanes(&grid(1.3, 4.0, 6), &[], [true, true]);
        let flowing = check(&law, DirectionPolicy::Bidirectional, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }
}
