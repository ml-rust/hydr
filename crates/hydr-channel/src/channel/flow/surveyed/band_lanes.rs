use super::PrismaticRiemannRejection;
use super::analytic_bottom::kinds;
use super::equation_ops::{choose, scalar};
use super::error::{Arithmetic, HydraulicError, HydraulicResult};
use crate::channel::section::{Constitutive, GatheredBands, PRISMATIC_BAND_WORK_LIMIT};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    TypeConversionOps, UtilityOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) struct IntegrationLanes<R: Runtime<DType = DType>> {
    pub face_owner: Tensor<R>,
    pub band_index: Tensor<R>,
    pub bottom: Tensor<R>,
    pub active: Tensor<R>,
    pub analytic: Tensor<R>,
    pub extent: Interval<R>,
    pub width: Interval<R>,
    pub slope: Interval<R>,
    pub area: Interval<R>,
    pub gravity: Interval<R>,
    pub face_count: usize,
    pub band_count: usize,
}

impl<R> IntegrationLanes<R>
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
        state: &Constitutive<R>,
        gravity: f64,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let count = state.depth.lower().numel();
        let total = count
            .checked_mul(bands.band_count)
            .filter(|n| *n <= PRISMATIC_BAND_WORK_LIMIT && *n <= u32::MAX as usize)
            .and_then(|n| n.checked_mul(std::mem::size_of::<f64>()).map(|_| n))
            .ok_or_else(|| HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                query: None,
                section_owner: None,
                context:
                    "Surveyed integration lane allocation exceeds its limit. Split the input batch"
                        .into(),
            })?;
        let zero = scalar(a, 0.0)?;
        let one = scalar(a, 1.0)?;
        let shape = [count, bands.band_count];
        let flatten =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                Ok(a.bounds(&value.lower().flatten()?, &value.upper().flatten()?)?)
            };
        let depth = a.bounds(
            &state.depth.lower().reshape(&[count, 1])?,
            &state.depth.upper().reshape(&[count, 1])?,
        )?;
        let local = a.sub(&depth, &bands.start_depth)?;
        // Clamp each physical contribution before evaluating its independent band polynomial.
        let extent = a.bounds(
            &c.maximum(
                &c.minimum(local.lower(), bands.thickness.lower())?,
                zero.lower(),
            )?,
            &c.maximum(
                &c.minimum(local.upper(), bands.thickness.upper())?,
                zero.upper(),
            )?,
        )?;
        let extent = choose(a, &bands.valid, &extent, &zero)?;
        let extent = flatten(a, &extent)?;
        let face = c
            .arange(0.0, count as f64, 1.0, DType::I64)?
            .reshape(&[count, 1])?;
        let band = c
            .arange(0.0, bands.band_count as f64, 1.0, DType::I64)?
            .reshape(&[1, bands.band_count])?;
        let face_owner = face.broadcast_to(&shape)?.contiguous()?.reshape(&[total])?;
        let band_index = band.broadcast_to(&shape)?.contiguous()?.reshape(&[total])?;
        let bottom = c.cast(
            &c.eq(&band_index, &Tensor::from_slice(&[0_i64], &[], c.device())?)?,
            DType::F64,
        )?;
        let width = flatten(a, &bands.start_width)?;
        let slope = flatten(a, &bands.slope)?;
        let area = flatten(a, &bands.start_area)?;
        let (rectangular, pointed) = kinds(a, &bottom, &bands.pointed.flatten()?, &width, &slope)?;
        let zero_extent = c.eq(extent.upper(), zero.upper())?;
        let analytic = c.maximum(&zero_extent, &c.maximum(&pointed, &rectangular)?)?;
        let active = c.mul(&bands.valid.flatten()?, &c.eq(&analytic, zero.lower())?)?;
        let safe_width = choose(a, &active, &width, &one)?;
        let safe_area = choose(a, &active, &area, &one)?;
        let safe_slope = choose(a, &active, &slope, &zero)?;
        Ok(Self {
            face_owner,
            band_index,
            bottom,
            active,
            analytic,
            extent,
            width: safe_width,
            slope: safe_slope,
            area: safe_area,
            gravity: scalar(a, gravity)?,
            face_count: count,
            band_count: bands.band_count,
        })
    }
}
