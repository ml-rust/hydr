//! Inlet-control discharge `Q_i(HW)`, the inverse of the HDS-5 inlet curve.
//!
//! | Band | `X = Ku Q / (A D^0.5)` from the headwater `HW` |
//! | --- | --- |
//! | unsubmerged, form 2 | `(HW / (D K))^(1/M)` |
//! | unsubmerged, form 1 | certified root in the critical-depth fraction, see `form_one` |
//! | transition | `3.5 + (HW - HW_3.5) / (2 (HW_4 - HW_3.5))` |
//! | submerged | `sqrt((HW / D - Y - Ks S) / c)` |
//!
//! The anchors `HW_3.5` and `HW_4` are enclosures, so a headwater next to one may lie in two
//! bands. The discharge then takes the hull of both band inverses. The exact inverse equals one
//! of them, because the curve rises through both bands.
//!
//! A form-1 curve starts at `HW_i(0) = D Ks S`, which is positive on an adverse or mitered
//! barrel. A headwater at or below it gives exact zero discharge.

use super::lanes::Lanes;
use super::{Selected, form_one};
use crate::structures::numerics::culvert::Arithmetic;
use crate::structures::numerics::culvert::constants::UNSUBMERGED_LIMIT;
use crate::structures::tensor::culvert::{Bounds, TensorCulvertLaw};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticResult, with_interval_arithmetic};

/// 0 or 1 masks of the bands each active lane may lie in.
pub(crate) struct Bands<R: Runtime<DType = DType>> {
    pub(crate) unsubmerged: Tensor<R>,
    pub(crate) transition: Tensor<R>,
    pub(crate) submerged: Tensor<R>,
}

/// Inlet-control discharge per barrel on every lane.
pub(crate) struct InletDischarge<R: Runtime<DType = DType>> {
    /// Enclosure of `Q_i`. Inactive and zero lanes hold `[0, 0]`.
    pub(crate) bounds: Bounds<R>,
    /// 1 on active lanes whose headwater lies at or below `HW_i(0)`, so `Q_i = 0` exactly.
    pub(crate) zero: Tensor<R>,
    /// Bands of each active lane.
    pub(crate) bands: Bands<R>,
}

/// Inverts the inlet curve of every active lane at its headwater.
pub(crate) fn invert<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
) -> Result<InletDischarge<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let inlet = &rows.inlet;
    let c = &barrel.client;
    let h = &lanes.headwater;
    let active = &lanes.active;
    let [low_lower, low_upper] = &inlet.low;
    let [high_lower, high_upper] = &inlet.high;
    let bands = Bands {
        unsubmerged: c.mul(active, &c.le(h, low_upper)?)?,
        transition: c.mul(active, &c.mul(&c.gt(h, low_lower)?, &c.lt(h, high_upper)?)?)?,
        submerged: c.mul(active, &c.ge(h, high_lower)?)?,
    };
    // Form 1 starts at `HW_i(0) = D Ks S`. Form 2 starts at zero, below every active headwater.
    let start = with_interval_arithmetic(c, DType::F64, "culvert inlet start", |a| {
        let d = a.point(&barrel.rise)?;
        let slope_term = a.bounds(&inlet.slope_term[0], &inlet.slope_term[1])?;
        Ok(a.mul(&d, &slope_term)?.lower().clone())
    })?;
    let form_one_free = c.mul(&inlet.form_one, &bands.unsubmerged)?;
    let zero = c.mul(&form_one_free, &c.le(h, &start)?)?;
    let solve = c.mul(&form_one_free, &c.sub(&barrel.one, &zero)?)?;
    let solved = if any(c, &solve)? {
        form_one::invert(rows, h, &solve)?
    } else {
        [
            barrel.constant(f64::INFINITY)?,
            barrel.constant(f64::NEG_INFINITY)?,
        ]
    };
    let closed = Closed {
        rows,
        headwater: h,
        bands: &bands,
        form_one: &solved,
    };
    let [lower, upper] =
        with_interval_arithmetic(c, DType::F64, "culvert inlet inversion", |a| closed.hull(a))?;
    let flowing = c.mul(active, &c.sub(&barrel.one, &zero)?)?;
    let empty = c.mul(&flowing, &c.gt(&lower, &upper)?)?;
    if let Some(lane) = super::first(&empty)? {
        return Err(StructureError::numerical(
            "culvert inlet inversion found no band for the headwater. Report this lane",
        )
        .at(law.elements[lane]));
    }
    Ok(InletDischarge {
        bounds: [
            c.where_cond(&flowing, &lower, &barrel.zero)?,
            c.where_cond(&flowing, &upper, &barrel.zero)?,
        ],
        zero,
        bands,
    })
}

/// Operands of the closed-form band inverses.
struct Closed<'o, R: Runtime<DType = DType>> {
    rows: &'o Selected<R>,
    headwater: &'o Tensor<R>,
    bands: &'o Bands<R>,
    /// Form-1 unsubmerged discharge enclosure, empty where not solved.
    form_one: &'o Bounds<R>,
}

impl<R> Closed<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    /// Hull of the discharge over the bands each lane may lie in. Lanes in no band get
    /// `(+inf, -inf)`.
    fn hull(&self, a: &mut Arithmetic<'_, R>) -> ArithmeticResult<Bounds<R>> {
        let barrel = &self.rows.barrel;
        let inlet = &self.rows.inlet;
        let c = a.client();
        let enter = |a: &mut Arithmetic<'_, R>, [lower, upper]: &Bounds<R>| a.bounds(lower, upper);
        let d = a.point(&barrel.rise)?;
        let scale = enter(a, &inlet.scale)?;
        let slope_term = enter(a, &inlet.slope_term)?;
        let harmless =
            |mask: &Tensor<R>, value: &Tensor<R>| c.where_cond(mask, self.headwater, value);

        // Form 2: `X = exp((ln HW - ln (D K)) / M)`. Other lanes evaluate at `HW = D K`.
        let k = a.point(&inlet.k)?;
        let dk = a.mul(&d, &k)?;
        let form_two = c.mul(
            &self.bands.unsubmerged,
            &c.sub(&barrel.one, &inlet.form_one)?,
        )?;
        let h_two = a.point(&harmless(&form_two, dk.upper())?)?;
        let ln_h = a.log_positive(&h_two)?;
        let ln_dk = a.log_positive(&dk)?;
        let ratio = a.sub(&ln_h, &ln_dk)?;
        let ln_x = a.div_positive(&ratio, &inlet.m)?;
        let x_two = super::exp(a, &ln_x)?;
        let q_two = a.mul(&x_two, &scale)?;

        // Transition: linear in `X` between the anchors.
        let low = enter(a, &inlet.low)?;
        let high = enter(a, &inlet.high)?;
        let h_mid = a.point(self.headwater)?;
        let past = a.sub(&h_mid, &low)?;
        let rise = a.sub(&high, &low)?;
        let two = a.point(&barrel.constant(2.0)?)?;
        let span = a.mul(&two, &rise)?;
        let fraction = a.div_positive_interval(&past, &span)?;
        let limit = a.point(&barrel.constant(UNSUBMERGED_LIMIT)?)?;
        let x_mid = a.add(&limit, &fraction)?;
        let x_mid = a.bounds(
            &c.maximum(x_mid.lower(), &barrel.zero)?,
            &c.maximum(x_mid.upper(), &barrel.zero)?,
        )?;
        let q_mid = a.mul(&x_mid, &scale)?;

        // Submerged: `X = sqrt((HW / D - Y - Ks S) / c)`.
        let h_full = a.point(self.headwater)?;
        let relative = a.div_positive(&h_full, &barrel.rise)?;
        let y = a.point(&inlet.y)?;
        let offset = a.add(&y, &slope_term)?;
        let excess = a.sub(&relative, &offset)?;
        let square = a.div_positive(&excess, &inlet.c)?;
        let square = a.bounds(
            &c.maximum(square.lower(), &barrel.zero)?,
            &c.maximum(square.upper(), &barrel.zero)?,
        )?;
        let x_full = a.sqrt_nonnegative(&square)?;
        let q_full = a.mul(&x_full, &scale)?;

        // The form-1 pair may be empty, so it stays outside the interval context.
        let free = [
            c.where_cond(&inlet.form_one, &self.form_one[0], q_two.lower())?,
            c.where_cond(&inlet.form_one, &self.form_one[1], q_two.upper())?,
        ];
        let mut lower = barrel.constant(f64::INFINITY)?;
        let mut upper = barrel.constant(f64::NEG_INFINITY)?;
        for (mask, [low, high]) in [
            (&self.bands.unsubmerged, free),
            (
                &self.bands.transition,
                [q_mid.lower().clone(), q_mid.upper().clone()],
            ),
            (
                &self.bands.submerged,
                [q_full.lower().clone(), q_full.upper().clone()],
            ),
        ] {
            lower = c.where_cond(mask, &c.minimum(&lower, &low)?, &lower)?;
            upper = c.where_cond(mask, &c.maximum(&upper, &high)?, &upper)?;
        }
        Ok([lower, upper])
    }
}

/// Whether any row of the 0 or 1 `mask` is 1. Reads one value to the host.
pub(crate) fn any<R>(c: &R::Client, mask: &Tensor<R>) -> Result<bool, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    Ok(c.sum(mask, &[0], false)?.item::<f64>()? != 0.0)
}
