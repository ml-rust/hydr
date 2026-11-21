//! Inlet context construction: band limits, transition anchors and admission of rising sets.
//!
//! Both anchors sit at exact `X`: the unsubmerged headwater at `X = 3.5` and the submerged
//! headwater at `X = 4`. A form-1 anchor needs the critical depth at the exact limit discharge
//! `3.5 scale`. Critical depth rises with discharge, so solving at both ends of the enclosure of
//! that discharge encloses it.
//!
//! Each band rises in `X` because `K`, `M` and `c` are positive and `Hc` rises with `dc`.
//! The transition rises when the upper anchor lies above the lower one. Admission certifies
//! that on every row, so the whole headwater curve strictly rises in `Q`.

use super::head::critical_head;
use super::power::power;
use crate::structures::numerics::culvert::constants::{
    INLET_KU, MITERED_SLOPE_FACTOR, SLOPE_FACTOR, SUBMERGED_LIMIT, UNSUBMERGED_LIMIT,
};
use crate::structures::numerics::culvert::{
    BarrelClient, BarrelDepths, RowNames, choose, critical,
};
use crate::structures::tensor::culvert::{Bounds, TensorBarrelContext, TensorInletContext};
use crate::structures::{CulvertBarrel, CulvertInletForm, CulvertInletRow, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{Interval, with_interval_arithmetic};
use solvr::optimize::CertifiedBisectionAlgorithms;

/// Builds the inlet rows of `rows` on the barrel context and admits them.
///
/// `barrels` lists the barrel of each row. `unit` is the unit-circle context of `barrel`.
/// Rejects a row whose critical depth at the unsubmerged limit is not below the crown, and a row
/// whose transition does not certifiably rise, naming the row and its inlet.
pub(crate) fn prepare<R>(
    barrels: &[CulvertBarrel],
    barrel: &TensorBarrelContext<R>,
    unit: &TensorBarrelContext<R>,
    rows: &[CulvertInletRow],
) -> Result<TensorInletContext<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: BarrelClient<R> + CertifiedBisectionAlgorithms<R>,
{
    let device = &barrel.device;
    let m = rows.len();
    let column = |f: &dyn Fn(&CulvertInletRow) -> f64| -> numr::error::Result<Tensor<R>> {
        let values: Vec<f64> = rows.iter().map(f).collect();
        Tensor::<R>::from_slice(&values, &[m], device)
    };
    let form_one = column(&|r| f64::from(u8::from(r.inlet.form() == CulvertInletForm::One)))?;
    let [k, m_exponent, c, y] = [0, 1, 2, 3].map(|i| column(&|r| r.inlet.coefficients()[i]));
    let (k, m_exponent, c, y) = (k?, m_exponent?, c?, y?);
    let gravity = column(&|r| r.gravity_m_per_s2)?;
    let slope = column(&|r| r.slope)?;
    let factor = column(&|r| {
        if r.inlet.mitered() {
            MITERED_SLOPE_FACTOR
        } else {
            SLOPE_FACTOR
        }
    })?;

    let client = &barrel.client;
    let [
        slope_term,
        scale,
        unsubmerged_discharge,
        submerged_discharge,
        high,
        partial,
        form_two,
    ] = with_interval_arithmetic(client, DType::F64, "culvert inlet limits", |a| {
        let ends = |i: &Interval<R>| -> Bounds<R> { [i.lower().clone(), i.upper().clone()] };
        let ku = a.bounds(
            &barrel.constant(f64::from_bits(INLET_KU[0]))?,
            &barrel.constant(f64::from_bits(INLET_KU[1]))?,
        )?;
        let area = a.bounds(&barrel.full_area[0], &barrel.full_area[1])?;
        let d = a.point(&barrel.rise)?;
        let root = a.sqrt_nonnegative(&d)?;
        let product = a.mul(&area, &root)?;
        let scale = a.div_positive_interval(&product, &ku)?;
        let low_x = a.point(&barrel.constant(UNSUBMERGED_LIMIT)?)?;
        let high_x = a.point(&barrel.constant(SUBMERGED_LIMIT)?)?;
        let low_q = a.mul(&low_x, &scale)?;
        let high_q = a.mul(&high_x, &scale)?;
        let ks = a.point(&factor)?;
        let s = a.point(&slope)?;
        let slope_term = a.mul(&ks, &s)?;

        let square = a.mul(&high_x, &high_x)?;
        let coefficient = a.point(&c)?;
        let quadratic = a.mul(&coefficient, &square)?;
        let offset = a.point(&y)?;
        let shifted = a.add(&quadratic, &offset)?;
        let total = a.add(&shifted, &slope_term)?;
        let high = a.mul(&d, &total)?;

        let raised = power(a, barrel, &low_x, &m_exponent)?;
        let gain = a.point(&k)?;
        let scaled = a.mul(&gain, &raised)?;
        let form_two = a.mul(&d, &scaled)?;
        let sloped = a.add(&scaled, &slope_term)?;
        let partial = a.mul(&d, &sloped)?;
        Ok([
            ends(&slope_term),
            ends(&scale),
            ends(&low_q),
            ends(&high_q),
            ends(&high),
            ends(&partial),
            ends(&form_two),
        ])
    })?;

    // Critical depth at both ends of the limit-discharge enclosure, on form-1 rows only.
    let at = |end: &Tensor<R>| -> Result<BarrelDepths<R>, StructureError> {
        let flow = client.where_cond(&form_one, end, &barrel.zero)?;
        critical::solve(RowNames::new(barrels), barrel, unit, &flow, &gravity)
    };
    let below = at(&unsubmerged_discharge[0])?;
    let above = at(&unsubmerged_discharge[1])?;
    let status = client.maximum(&below.status, &above.status)?;
    let enclosure = BarrelDepths {
        status,
        lower: below.lower,
        upper: above.upper,
    };
    let (low, valid) =
        with_interval_arithmetic(client, DType::F64, "culvert inlet anchors", |a| {
            let head = critical_head(a, barrel, unit, &enclosure)?;
            let partial = a.bounds(&partial[0], &partial[1])?;
            let one = a.add(&head.head, &partial)?;
            let two = a.bounds(&form_two[0], &form_two[1])?;
            let low = choose(a, &form_one, &one, &two)?;
            Ok(([low.lower().clone(), low.upper().clone()], head.valid))
        })?;

    let rising = client.gt(&high[0], &low[1])?;
    let reached = client.mul(&form_one, &client.sub(&barrel.one, &valid)?)?;
    let good = client.mul(&rising, &client.sub(&barrel.one, &reached)?)?;
    let bad = client.sub(&barrel.one, &good)?;
    if client.sum(&bad, &[0], false)?.item::<f64>()? != 0.0 {
        return Err(reject(rows, &reached, [&low, &high])?);
    }
    Ok(TensorInletContext {
        form_one,
        k,
        m: m_exponent,
        c,
        y,
        gravity,
        slope_term,
        scale,
        unsubmerged_discharge,
        submerged_discharge,
        low,
        high,
    })
}

/// Names the first rejected row. Reads the masks and anchors to the host.
fn reject<R>(
    rows: &[CulvertInletRow],
    reached: &Tensor<R>,
    [low, high]: [&Bounds<R>; 2],
) -> Result<StructureError, StructureError>
where
    R: Runtime<DType = DType>,
{
    let reached = reached.try_to_vec::<f64>()?;
    let read = |[lower, upper]: &Bounds<R>| -> Result<Vec<(f64, f64)>, StructureError> {
        Ok(lower
            .try_to_vec::<f64>()?
            .into_iter()
            .zip(upper.try_to_vec::<f64>()?)
            .collect())
    };
    let low = read(low)?;
    let high = read(high)?;
    for (index, row) in rows.iter().enumerate() {
        let (Some(&crown), Some(&(low_lower, low_upper)), Some(&(high_lower, high_upper))) =
            (reached.get(index), low.get(index), high.get(index))
        else {
            break;
        };
        let fields = format!(
            "{},slope={:?},gravity_m_per_s2={:?}",
            row.inlet.describe(),
            row.slope,
            row.gravity_m_per_s2
        );
        if crown != 0.0 {
            return Ok(StructureError::definition(format!(
                "{fields}: the critical depth at X = 3.5 is not below the crown, so the form-1 critical head is undefined. Supply a larger gravity or a form-2 inlet"
            ))
            .at(index));
        }
        if high_lower <= low_upper {
            return Ok(StructureError::definition(format!(
                "{fields}: the headwater falls across the transition, from [{low_lower:?}, {low_upper:?}] m at X = 3.5 to [{high_lower:?}, {high_upper:?}] m at X = 4. Supply coefficients whose submerged headwater at X = 4 exceeds the unsubmerged headwater at X = 3.5"
            ))
            .at(index));
        }
    }
    Ok(StructureError::numerical(
        "the device check found a rejected inlet row that the host copy does not show: report this runtime",
    ))
}
