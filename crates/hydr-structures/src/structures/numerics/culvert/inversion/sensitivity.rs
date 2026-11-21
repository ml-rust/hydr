//! Culvert head sensitivity: a bound on `|dQ/dH| / |Q|` per metre of either head.
//!
//! `|dQ/dH| / Q = 1 / E` for the governing control, with `E = Q dHW/dQ` at the root.
//!
//! | Control | `E` or bound |
//! | --- | --- |
//! | inlet, form 2 unsubmerged | `E = M HW`, since `HW = D K X^M` |
//! | inlet, form 1 unsubmerged | `E >= M D K X^M`, since the critical head rises with `Q` |
//! | inlet, transition | `E = 2 (HW_4 - HW_3.5) X` |
//! | inlet, submerged | `E = 2 D c X^2` |
//! | outlet, drowned full barrel | `Q = A sqrt(2 g dH / k)`, so the bound is `1 / (2 dH)` |
//! | outlet, other | twice a secant estimate, see below |
//!
//! Inlet bounds take the lower end of the discharge enclosure for `X`, and the smallest `E`
//! over the bands a lane may lie in. The inlet ignores the tailwater.
//!
//! No closed form bounds the partly full profile. Its lanes take twice the centred secants of
//! the outlet march over `Q (1 +- 2^-7)` and over `TW +- 2^-7 D`:
//! `2 max(1, |dHW/dTW|) / (Q dHW/dQ)`. This is an estimate, not a proof. A tie takes the larger
//! of both controls. Lanes without flow carry zero.
//!
//! Every bound is capped at `2^256` per metre, and `E` is floored at `2^-256` m, so products
//! with volumes and heads stay finite. A secant through an unresolved march point takes the cap.

use super::Selected;
use super::inlet::{InletDischarge, any};
use super::lanes::Lanes;
use super::outlet::OutletDischarge;
use crate::structures::numerics::culvert::{Arithmetic, RowNames, outlet};
use crate::structures::tensor::culvert::TensorCulvertLaw;
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, with_interval_arithmetic};

/// Relative discharge step and tailwater step per unit rise of the outlet secants, `2^-7`.
const STEP: f64 = 0.0078125;

/// Margin on the outlet secant estimate.
const SECANT_MARGIN: f64 = 2.0;

/// Smallest `E`, `2^-256` m, so every bound stays at or below [`CAP`].
const E_FLOOR: f64 = f64::from_bits(0x2ff0_0000_0000_0000);

/// Largest bound, `2^256` per metre. Products with volumes and heads stay finite.
const CAP: f64 = f64::from_bits(0x4ff0_0000_0000_0000);

/// 0 or 1 masks of the governing control of each flowing lane.
pub(crate) struct Governs<R: Runtime<DType = DType>> {
    pub(crate) inlet: Tensor<R>,
    pub(crate) outlet: Tensor<R>,
    pub(crate) tie: Tensor<R>,
}

/// Bounds the head sensitivity of every lane. `point` is the per-barrel discharge.
pub(crate) fn bound<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
    outlet_discharge: &OutletDischarge<R>,
    governs: &Governs<R>,
    point: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let one = &barrel.one;
    let inlet_side = c.maximum(&governs.inlet, &governs.tie)?;
    let outlet_side = c.maximum(&governs.outlet, &governs.tie)?;
    let full = c.mul(&outlet_side, &outlet_discharge.full)?;
    let secant = c.mul(&outlet_side, &c.sub(one, &outlet_discharge.full)?)?;
    let [inlet_bound, full_bound] =
        with_interval_arithmetic(c, DType::F64, "culvert inlet sensitivity", |a| {
            Ok([
                inlet_sensitivity(a, rows, lanes, inlet, &inlet_side)?,
                full_sensitivity(a, rows, lanes, &full)?,
            ])
        })?;
    let profile = if any(c, &secant)? {
        secant_sensitivity(law, rows, lanes, &secant, point)?
    } else {
        barrel.zero.clone()
    };
    let outlet_bound = c.where_cond(&full, &full_bound, &profile)?;
    let combined = c.where_cond(
        &governs.inlet,
        &inlet_bound,
        &c.where_cond(
            &governs.outlet,
            &outlet_bound,
            &c.maximum(&inlet_bound, &outlet_bound)?,
        )?,
    )?;
    let flowing = c.maximum(&inlet_side, &governs.outlet)?;
    Ok(c.where_cond(&flowing, &combined, &barrel.zero)?)
}

/// `1 / E` upper bounds, with `E` floored at [`E_FLOOR`].
fn reciprocal<R>(a: &mut Arithmetic<'_, R>, e: &Tensor<R>) -> ArithmeticResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let c = a.client();
    let floor = Tensor::<R>::full_scalar(e.shape(), DType::F64, E_FLOOR, c.device())?;
    let safe = c.maximum(e, &floor)?;
    let ones = Tensor::<R>::ones(e.shape(), DType::F64, c.device())?;
    let unit = a.point(&ones)?;
    Ok(a.div_positive(&unit, &safe)?.upper().clone())
}

/// Inlet bound `1 / E` on `wanted` lanes, a harmless value elsewhere.
fn inlet_sensitivity<R>(
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
    let power = super::exp(a, &ln_power)?;
    let gain = a.mul(&k, &power)?;
    let scaled = a.mul(&d, &gain)?;
    let form_one = a.mul(&m, &scaled)?;
    let free = super::super::choose(a, &coefficients.form_one, &form_one, &form_two)?;
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
fn full_sensitivity<R>(
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

/// Twice the secant estimate on the `wanted` lanes, zero elsewhere.
fn secant_sensitivity<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    wanted: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let m = barrel.width();
    let q = c.where_cond(wanted, point, &barrel.rise)?;
    let tw = &lanes.tailwater;
    let step = c.mul_scalar(&barrel.rise, STEP)?;
    let q_low = c.mul_scalar(&q, 1.0 - STEP)?;
    let q_high = c.mul_scalar(&q, 1.0 + STEP)?;
    let tw_high = c.add(tw, &step)?;
    let tw_low = c.maximum(&c.sub(tw, &step)?, &barrel.zero)?;
    let discharge = c.cat(&[&q_low, &q_high, &q, &q], 0)?;
    let tailwater = c.cat(&[tw, tw, &tw_high, &tw_low], 0)?;
    let lane: Vec<i64> = (0..4 * m)
        .map(|index| i64::try_from(index % m).unwrap_or(i64::MAX))
        .collect();
    let owner = Tensor::<R>::from_slice(&lane, &[4 * m], &barrel.device)?;
    let picked = rows.select(&owner)?;
    let oriented = c.index_select(&lanes.rows, 0, &owner)?;
    let march = outlet::march(
        RowNames::mapped(&law.barrels, &oriented),
        &picked.barrel,
        &picked.unit,
        &picked.outlet,
        &discharge,
        &tailwater,
    )?;
    let [low, high] = &march.headwater.headwater;
    let middle = c.mul_scalar(&c.add(low, high)?, 0.5)?;
    let part = |index: usize| middle.narrow(0, index * m, m);
    let [hw_q_low, hw_q_high, hw_tw_high, hw_tw_low]: Bounds4<R> =
        [part(0)?, part(1)?, part(2)?, part(3)?];
    let e = c.div(
        &c.mul(&q, &c.sub(&hw_q_high, &hw_q_low)?)?,
        &c.sub(&q_high, &q_low)?,
    )?;
    let g = c.div(&c.sub(&hw_tw_high, &hw_tw_low)?, &c.sub(&tw_high, &tw_low)?)?;
    let floor = barrel.constant(E_FLOOR)?;
    let numerator = c.mul_scalar(&c.maximum(&c.abs(&g)?, &barrel.one)?, SECANT_MARGIN)?;
    let estimate = c.minimum(
        &c.div(&numerator, &c.maximum(&e, &floor)?)?,
        &barrel.constant(CAP)?,
    )?;
    // A secant through an unresolved march point is NaN. The cap replaces it.
    let unknown = c.isnan(&estimate)?;
    let estimate = c.where_cond(&unknown, &barrel.constant(CAP)?, &estimate)?;
    Ok(c.where_cond(wanted, &estimate, &barrel.zero)?)
}

/// Four `[m]` tensors.
type Bounds4<R> = [Tensor<R>; 4];

#[cfg(test)]
mod tests {
    use super::{CAP, E_FLOOR};
    use crate::structures::numerics::sensitivity::fixture::{
        Lane, MARGIN, STEP as DIFFERENCE, TestResult, check_bound, culvert_structures, grid, lanes,
    };
    use crate::structures::{
        CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds,
        CulvertInlet, CulvertInletForm, CulvertParameters, DirectionPolicy, StructureLaw,
        StructureOperatingPoint,
    };

    fn culvert(
        barrel: CulvertBarrel,
        (length_m, manning_n): (f64, f64),
        inlet: CulvertInlet,
        both: bool,
    ) -> TestResult<StructureLaw> {
        let end = CulvertEnd::new(inlet, 0.5)?;
        Ok(StructureLaw::Culvert(CulvertParameters::new(
            barrel,
            CulvertAlignment {
                length_m,
                invert_a_m: 0.0,
                invert_b_m: 0.0,
            },
            manning_n,
            1,
            CulvertEnds {
                a: Some(end),
                b: both.then_some(end),
            },
            9.81,
            CulvertApplicability {
                coefficient_provenance: "Analytical coefficients".into(),
                barrel_condition_assertion: "Clean analytical barrel".into(),
            },
        )?))
    }

    fn check(law: &StructureLaw, direction: DirectionPolicy, cases: &[Lane]) -> TestResult<usize> {
        let structures = culvert_structures(
            |_| (law.clone(), direction, StructureOperatingPoint::Passive),
            cases.len(),
        )?;
        check_bound(&structures, cases)
    }

    #[test]
    fn floor_and_cap_are_the_stated_powers_of_two() {
        assert_eq!(E_FLOOR, 2.0_f64.powi(-256));
        assert_eq!(CAP, 2.0_f64.powi(256));
    }

    #[test]
    fn inlet_sensitivity_bounds_unsubmerged_transition_and_submerged_heads() -> TestResult {
        // The steep explicit inlet governs at every head. Its anchors sit at
        // `HW = D K 3.5^M` and `D (16 c + Y)` above the invert at zero.
        let inlet = CulvertInlet::explicit(CulvertInletForm::Two, [2.0, 0.6, 0.5, 0.0], false)?;
        let law = culvert(
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            (10.0, 0.012),
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
            CulvertBarrel::circular(1.0)?,
            (200.0, 0.03),
            CulvertInlet::hds5(1)?,
            true,
        )?;
        let cases = lanes(&grid(1.3, 4.0, 6), &[], [true, true]);
        let flowing = check(&law, DirectionPolicy::Bidirectional, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }

    #[test]
    fn partly_full_outlet_estimate_bounds_a_centred_difference() -> TestResult {
        let law = culvert(
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            (100.0, 0.03),
            CulvertInlet::hds5(9)?,
            false,
        )?;
        let cases: Vec<Lane> = [0.9, 1.15]
            .iter()
            .map(|&a| Lane {
                stage: [a, 0.5],
                wet: [true, true],
                step: DIFFERENCE * MARGIN,
            })
            .collect();
        let flowing = check(&law, DirectionPolicy::AtoBOnly, &cases)?;
        assert_eq!(flowing, cases.len());
        Ok(())
    }
}
