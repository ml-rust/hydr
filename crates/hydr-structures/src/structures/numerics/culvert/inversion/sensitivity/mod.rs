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
//! | outlet, other | `max(1, |dHW/dTW|) / E` from one march, see below |
//!
//! Inlet bounds take the lower end of the discharge enclosure for `X`, and the smallest `E`
//! over the bands a lane may lie in. The inlet ignores the tailwater.
//!
//! The other outlet lanes run one coarse march at the reported discharge and tailwater.
//! [`head_slopes`](outlet::head_slopes) bounds `E` below and `|dHW/dTW|` above in closed form
//! from its class and depths. `dQ/dHW = 1 / (dHW/dQ)` and `dQ/dTW = -(dHW/dTW) / (dHW/dQ)` at
//! fixed other head, so `max(1, |dHW/dTW|) / E` bounds both heads. A tie takes the larger of
//! both controls. Lanes without flow carry zero.
//!
//! Every bound is capped at `2^256` per metre, and `E` is floored at `2^-256` m, so products
//! with volumes and heads stay finite. A lane the march leaves without a slope bound takes the
//! cap: an open, tie or unresolved row, or one whose divisors are not certified positive.

mod closed;
#[cfg(test)]
mod fixture;

use super::Selected;
use super::inlet::{InletDischarge, any};
use super::lanes::Lanes;
use super::outlet::OutletDischarge;
use crate::structures::numerics::culvert::outlet::{self, Accuracy};
use crate::structures::numerics::culvert::{Arithmetic, RowNames};
use crate::structures::tensor::culvert::TensorCulvertLaw;
use crate::structures::{CulvertClient, StructureError};
use closed::{full_sensitivity, inlet_sensitivity};
use numr::dtype::DType;
use numr::ops::{BinaryOps, ConditionalOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, Interval, with_interval_arithmetic};

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
///
/// Also returns the outlet profile bounds of each head alone, zero off outlet profile lanes.
pub(crate) fn bound<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    inlet: &InletDischarge<R>,
    outlet_discharge: &OutletDischarge<R>,
    governs: &Governs<R>,
    point: &Tensor<R>,
) -> Result<(Tensor<R>, Profile<R>), StructureError>
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
    let profile_side = c.mul(&outlet_side, &c.sub(one, &outlet_discharge.full)?)?;
    let [inlet_bound, full_bound] =
        with_interval_arithmetic(c, DType::F64, "culvert inlet sensitivity", |a| {
            Ok([
                inlet_sensitivity(a, rows, lanes, inlet, &inlet_side)?,
                full_sensitivity(a, rows, lanes, &full)?,
            ])
        })?;
    let profile = if any(c, &profile_side)? {
        profile_sensitivity(law, rows, lanes, &profile_side, point)?
    } else {
        Profile {
            bound: barrel.zero.clone(),
            head: barrel.zero.clone(),
            tail: barrel.zero.clone(),
        }
    };
    let outlet_bound = c.where_cond(&full, &full_bound, &profile.bound)?;
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
    Ok((c.where_cond(&flowing, &combined, &barrel.zero)?, profile))
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

/// Profile bounds on the `wanted` lanes from one march, zero elsewhere.
fn profile_sensitivity<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    wanted: &Tensor<R>,
    point: &Tensor<R>,
) -> Result<Profile<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let q = c.where_cond(wanted, point, &barrel.rise)?;
    let tailwater = &lanes.tailwater;
    // The bounds read the class and depth enclosures, which a coarse march certifies too.
    let march = outlet::march(
        RowNames::mapped(&law.barrels, &lanes.rows),
        barrel,
        &rows.unit,
        &rows.outlet,
        &q,
        tailwater,
        Accuracy::Coarse,
    )?;
    let operands = outlet::SlopeOperands {
        barrel,
        outlet: &rows.outlet,
        march: &march,
        discharge: &q,
        tailwater,
    };
    let [bound, head, tail] =
        with_interval_arithmetic(c, DType::F64, "culvert outlet sensitivity", |a| {
            let slopes = outlet::head_slopes(a, &operands)?;
            let c = a.client();
            let one = &barrel.one;
            let cap = barrel.constant(CAP)?;
            let gain = c.where_cond(&slopes.known, &slopes.gain, one)?;
            let gain = c.maximum(&gain, &barrel.constant(E_FLOOR)?)?;
            let tail = c.where_cond(&slopes.known, &slopes.tail, one)?;
            let wide = a.point(&c.maximum(&tail, one)?)?;
            let unit = a.point(one)?;
            let alone = a.point(&tail)?;
            let ratios = [
                a.div_positive(&wide, &gain)?,
                a.div_positive(&unit, &gain)?,
                a.div_positive(&alone, &gain)?,
            ];
            let c = a.client();
            let capped = |ratio: &Interval<R>| -> ArithmeticResult<Tensor<R>> {
                Ok(c.where_cond(&slopes.known, &c.minimum(ratio.upper(), &cap)?, &cap)?)
            };
            Ok([
                capped(&ratios[0])?,
                capped(&ratios[1])?,
                capped(&ratios[2])?,
            ])
        })?;
    let keep = |value: &Tensor<R>| c.where_cond(wanted, value, &barrel.zero);
    Ok(Profile {
        bound: keep(&bound)?,
        head: keep(&head)?,
        tail: keep(&tail)?,
    })
}

/// Outlet profile sensitivities on the wanted lanes, each per metre and zero elsewhere.
pub(crate) struct Profile<R: Runtime<DType = DType>> {
    /// `max(1, |dHW/dTW|) / E`, a bound for either head.
    pub(crate) bound: Tensor<R>,
    /// `1 / E`, the bound for the headwater alone.
    pub(crate) head: Tensor<R>,
    /// `|dHW/dTW| / E`, the bound for the tailwater alone.
    pub(crate) tail: Tensor<R>,
}

#[cfg(test)]
mod tests {
    use super::fixture::{Spec, certified, level, pairs, weak};
    use super::{CAP, E_FLOOR};
    use crate::structures::numerics::sensitivity::fixture::{TestResult, grid};
    use crate::structures::{CulvertBarrel, CulvertInlet};

    #[test]
    fn floor_and_cap_are_the_stated_powers_of_two() {
        assert_eq!(E_FLOOR, 2.0_f64.powi(-256));
        assert_eq!(CAP, 2.0_f64.powi(256));
    }

    #[test]
    fn mild_partly_full_circle_bound_holds_over_a_head_grid() -> TestResult {
        // Fall 0.12 m over 60 m. Both heads stay below the 1.2 m crown. Two heads per side
        // hold both outlet classes: profiles falling and rising toward normal depth.
        let spec = Spec {
            barrel: CulvertBarrel::circular(1.2)?,
            length_m: 60.0,
            manning_n: 0.013,
            inverts: [0.12, 0.0],
        };
        let cases = pairs(
            &grid(0.3, 1.3, 2),
            &grid(0.02, 1.15, 2),
            &[0.0, 0.12, 1.2, 1.32],
        );
        assert!(certified(spec, CulvertInlet::hds5(1)?, &cases)? > cases.len() / 4);
        Ok(())
    }

    #[test]
    fn adverse_partly_full_box_bound_holds_over_a_head_grid() -> TestResult {
        // The barrel rises 0.3 m toward its outlet. Every outlet profile rises to the crown. Two
        // heads per side hold it, a headwater on each side of the inlet crown, and both flow
        // directions.
        let spec = Spec {
            barrel: CulvertBarrel::rectangular_box(2.0, 1.0)?,
            length_m: 30.0,
            manning_n: 0.015,
            inverts: [0.0, 0.3],
        };
        let cases = pairs(
            &grid(0.4, 1.6, 2),
            &grid(0.32, 1.25, 2),
            &[0.0, 0.3, 1.0, 1.3],
        );
        assert!(certified(spec, CulvertInlet::hds5(9)?, &cases)? > cases.len() / 4);
        Ok(())
    }

    #[test]
    fn zero_slope_partly_full_box_bound_holds_over_a_head_grid() -> TestResult {
        // Every outlet profile rises to the crown. Three heads per side hold it, a headwater on
        // each side of the crown, and both flow directions. Two would keep no headwater above
        // the crown.
        let spec = level(CulvertBarrel::rectangular_box(2.0, 1.0)?, (100.0, 0.03));
        let cases = pairs(&grid(0.1, 1.4, 3), &grid(0.02, 0.95, 3), &[0.0, 1.0]);
        assert!(certified(spec, CulvertInlet::hds5(9)?, &cases)? > cases.len() / 4);
        Ok(())
    }

    #[test]
    fn steep_drowned_barrels_bound_holds_over_a_head_grid() -> TestResult {
        // Slope 0.02 with the outlet above the crown: full, then free from the crown.
        for (barrel, rise, code) in [
            (CulvertBarrel::rectangular_box(2.0, 1.5)?, 1.5, 9),
            (CulvertBarrel::circular(1.2)?, 1.2, 1),
        ] {
            let spec = Spec {
                barrel,
                length_m: 40.0,
                manning_n: 0.013,
                inverts: [0.8, 0.0],
            };
            let cases = pairs(
                &grid(rise + 1.0, rise + 3.0, 8),
                &grid(rise + 0.02, rise + 0.8, 8),
                &[0.0, 0.8, rise, rise + 0.8],
            );
            assert!(certified(spec, CulvertInlet::hds5(code)?, &cases)? > cases.len() / 4);
        }
        Ok(())
    }

    #[test]
    fn mild_arc_barrels_bound_holds_over_a_head_grid() -> TestResult {
        // Fall 0.12 m over 60 m. The heads span free and full barrels, both flow directions and
        // both controls on a horizontal ellipse, a pipe-arch and a flat-floor arch.
        for (barrel, code) in [
            (
                CulvertBarrel::horizontal_ellipse(1.524, 0.9652, 0.419_475_92)?,
                30,
            ),
            (
                CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
                36,
            ),
            (CulvertBarrel::arch(3.048, 1.6002)?, 45),
        ] {
            let rise = barrel.rise_m();
            let spec = Spec {
                barrel,
                length_m: 60.0,
                manning_n: 0.024,
                inverts: [0.12, 0.0],
            };
            let cases = pairs(
                &grid(0.3, rise + 0.6, 2),
                &grid(0.02, rise + 0.4, 2),
                &[0.0, 0.12, rise, rise + 0.12],
            );
            assert!(certified(spec, CulvertInlet::hds5(code)?, &cases)? > cases.len() / 4);
        }
        Ok(())
    }

    #[test]
    fn steep_box_bound_holds_where_the_barrel_primes() -> TestResult {
        // A free outfall below the 1.5 m crown. The donor heads span critical depth at the crown
        // and the priming discharge, where the free-surface friction at the crown passes the
        // slope and the barrel runs full from the crown.
        let spec = Spec {
            barrel: CulvertBarrel::rectangular_box(2.0, 1.5)?,
            length_m: 40.0,
            manning_n: 0.013,
            inverts: [0.8, 0.0],
        };
        let cases = pairs(
            &grid(4.5, 9.5, 10),
            &grid(0.02, 1.4, 6),
            &[0.0, 0.8, 1.5, 2.3],
        );
        assert!(certified(spec, weak()?, &cases)? > cases.len() / 2);
        Ok(())
    }
}
