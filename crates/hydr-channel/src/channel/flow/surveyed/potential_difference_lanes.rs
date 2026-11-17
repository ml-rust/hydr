use super::PrismaticRiemannRejection;
use super::equation_ops::{choose, scalar, scale};
use super::error::{Arithmetic, HydraulicError, HydraulicResult};
use crate::channel::section::{GatheredBands, PRISMATIC_BAND_WORK_LIMIT};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, TypeConversionOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) struct DifferenceLanes<R: Runtime<DType = DType>> {
    pub active: Tensor<R>,
    pub equal: Tensor<R>,
    pub original: Interval<R>,
    pub delta: Interval<R>,
    pub hull: Interval<R>,
    pub start_area: Interval<R>,
    pub width: Interval<R>,
    pub slope: Interval<R>,
    pub gravity: Interval<R>,
    pub face_count: usize,
    pub band_count: usize,
}

impl<R> DifferenceLanes<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub fn build(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        original: &Tensor<R>,
        target: &Interval<R>,
        gravity: f64,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let count = original.numel();
        let total = count
            .checked_mul(bands.band_count)
            .filter(|n| *n <= PRISMATIC_BAND_WORK_LIMIT && *n <= u32::MAX as usize)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()).map(|_| n))
            .ok_or_else(|| HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                query: None,
                section_owner: None,
                context:
                    "Surveyed difference lane allocation exceeds its limit. Split the input batch"
                        .into(),
            })?;
        let zero = scalar(a, 0.0)?;
        let one = scalar(a, 1.0)?;
        let original = a.point(&original.unsqueeze(1)?)?;
        let target = a.bounds(&target.lower().unsqueeze(1)?, &target.upper().unsqueeze(1)?)?;
        let equal = c.mul(
            &c.eq(original.lower(), target.lower())?,
            &c.eq(original.upper(), target.upper())?,
        )?;
        let hull = a.bounds(
            &c.minimum(original.lower(), target.lower())?,
            &c.maximum(original.upper(), target.upper())?,
        )?;
        // Half the upper endpoint avoids overflowing twice the lower endpoint.
        let half = scale(a, &hull, 0.5)?;
        let conditioned = c.mul(
            &c.gt(hull.lower(), zero.upper())?,
            &c.le(half.upper(), hull.lower())?,
        )?;
        let inside = c.mul(
            &c.ge(hull.lower(), bands.start_area.upper())?,
            &c.le(hull.upper(), bands.end_area.lower())?,
        )?;
        let active = c.mul(&bands.valid, &c.mul(&conditioned, &inside)?)?;
        let active = c.mul(&active, &c.eq(&equal, zero.lower())?)?;
        let safe_hull = choose(a, &active, &hull, &one)?;
        let start = choose(a, &active, &bands.start_area, &zero)?;
        let width = choose(a, &active, &bands.start_width, &one)?;
        let slope = choose(a, &active, &bands.slope, &zero)?;
        let local = a.sub(&safe_hull, &start)?;
        let local = a.bounds(&c.maximum(local.lower(), zero.lower())?, local.upper())?;
        let width_squared = a.mul(&width, &width)?;
        let variable = a.mul(&slope, &local)?;
        let variable = scale(a, &variable, 2.0)?;
        let squared_width = a.add(&width_squared, &variable)?;
        let active = c.mul(&active, &c.gt(squared_width.lower(), zero.upper())?)?;
        let safe_original = choose(a, &active, &original, &one)?;
        let safe_target = choose(a, &active, &target, &one)?;
        let delta = a.sub(&safe_target, &safe_original)?;
        let safe_hull = choose(a, &active, &hull, &one)?;
        let start = choose(a, &active, &start, &zero)?;
        let width = choose(a, &active, &width, &one)?;
        let slope = choose(a, &active, &slope, &zero)?;
        let flatten =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                let captured = a.bounds(
                    &value.lower().broadcast_to(&[count, bands.band_count])?,
                    &value.upper().broadcast_to(&[count, bands.band_count])?,
                )?;
                Ok(a.bounds(
                    &captured.lower().reshape(&[total])?,
                    &captured.upper().reshape(&[total])?,
                )?)
            };
        Ok(Self {
            active: c.cast(&active.flatten()?, DType::F64)?,
            equal: c.cast(&equal.reshape(&[count])?, DType::F64)?,
            original: flatten(a, &safe_original)?,
            delta: flatten(a, &delta)?,
            hull: flatten(a, &safe_hull)?,
            start_area: flatten(a, &start)?,
            width: flatten(a, &width)?,
            slope: flatten(a, &slope)?,
            gravity: scalar(a, gravity)?,
            face_count: count,
            band_count: bands.band_count,
        })
    }
}
