use super::chart::{AreaAnchor, AreaQuery};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::GatheredBands,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps,
        TensorOps, TwoSumOps, TypeConversionOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(super) fn column<R: Runtime<DType = DType>>(
    value: &Tensor<R>,
    index: usize,
) -> numr::error::Result<Tensor<R>> {
    Ok(value.narrow(1, index, 1)?.squeeze(Some(1)))
}

pub(super) fn interval_column<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    index: usize,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.bounds(
        &column(value.lower(), index)?,
        &column(value.upper(), index)?,
    )?)
}

pub(super) fn counts<R>(
    a: &Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.client()
        .cast(&a.client().sum(&bands.valid, &[1], false)?, DType::I64)?)
}

fn merge<R>(
    a: &mut Arithmetic<'_, R>,
    carried: &Interval<R>,
    present: &Tensor<R>,
    reset: &Interval<R>,
    reset_mask: &Tensor<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let both = c.mul(present, reset_mask)?;
    let low = c.where_cond(
        &both,
        &c.minimum(carried.lower(), reset.lower())?,
        &c.where_cond(reset_mask, reset.lower(), carried.lower())?,
    )?;
    let high = c.where_cond(
        &both,
        &c.maximum(carried.upper(), reset.upper())?,
        &c.where_cond(reset_mask, reset.upper(), carried.upper())?,
    )?;
    Ok((a.bounds(&low, &high)?, c.maximum(present, reset_mask)?))
}

/// Transport local increments without subtracting complete prefix certificates.
pub(super) fn relative<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    query: &AreaQuery<R>,
) -> HydraulicResult<Vec<Interval<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let n = bands.section_owner.numel();
    let zeros = Tensor::zeros(&[n], DType::F64, c.device())?;
    let zero = a.point(&zeros)?;
    let boundary = match &query.anchor {
        AreaAnchor::Zero => Some(Tensor::zeros(&[n], DType::I64, c.device())?),
        AreaAnchor::Cap => Some(counts(a, bands)?),
        AreaAnchor::BandStart(index) => Some(index.clone()),
        AreaAnchor::BandEnd(index) => {
            Some(c.add(index, &Tensor::from_slice(&[1_i64], &[], c.device())?)?)
        }
        AreaAnchor::Source { .. } => None,
    };
    let boundary = boundary
        .map(|value| -> HydraulicResult<_> {
            Ok((
                value,
                c.arange(0.0, (bands.band_count + 1) as f64, 1.0, DType::I64)?,
            ))
        })
        .transpose()?;
    let mut forward = Vec::with_capacity(bands.band_count + 1);
    let mut masks = Vec::with_capacity(bands.band_count + 1);
    let mut value = a.point(&zeros)?;
    let mut present = zeros.clone();
    for index in 0..=bands.band_count {
        if index != 0 {
            let increment = interval_column(a, &bands.area_increment, index - 1)?;
            let transported = a.sub(&value, &increment)?;
            value = a.bounds(
                &c.where_cond(&present, transported.lower(), &zeros)?,
                &c.where_cond(&present, transported.upper(), &zeros)?,
            )?;
        }
        let (reset, reset_mask) = if let Some((boundary, indices)) = &boundary {
            let i = indices.narrow(0, index, 1)?.squeeze(Some(0));
            (a.point(&zeros)?, c.cast(&c.eq(boundary, &i)?, DType::F64)?)
        } else if let AreaAnchor::Source { coordinates, .. } = &query.anchor {
            if index == 0 {
                (a.point(&zeros)?, zeros.clone())
            } else {
                let upper = interval_column(a, coordinates.area_below_end_m2(), index - 1)?;
                (
                    a.sub(&zero, &upper)?,
                    column(coordinates.candidate_mask(), index - 1)?,
                )
            }
        } else {
            (a.point(&zeros)?, zeros.clone())
        };
        (value, present) = merge(a, &value, &present, &reset, &reset_mask)?;
        forward.push(a.bounds(value.lower(), value.upper())?);
        masks.push(present.clone());
    }
    let mut reverse = Vec::with_capacity(bands.band_count + 1);
    value = a.point(&zeros)?;
    present = zeros.clone();
    for index in (0..=bands.band_count).rev() {
        if index != bands.band_count {
            let increment = interval_column(a, &bands.area_increment, index)?;
            let transported = a.add(&value, &increment)?;
            value = a.bounds(
                &c.where_cond(&present, transported.lower(), &zeros)?,
                &c.where_cond(&present, transported.upper(), &zeros)?,
            )?;
        }
        let (reset, reset_mask) = if let Some((boundary, indices)) = &boundary {
            let i = indices.narrow(0, index, 1)?.squeeze(Some(0));
            (a.point(&zeros)?, c.cast(&c.eq(boundary, &i)?, DType::F64)?)
        } else if let AreaAnchor::Source { coordinates, .. } = &query.anchor {
            if index == bands.band_count {
                (a.point(&zeros)?, zeros.clone())
            } else {
                (
                    interval_column(a, coordinates.area_above_start_m2(), index)?,
                    column(coordinates.candidate_mask(), index)?,
                )
            }
        } else {
            (a.point(&zeros)?, zeros.clone())
        };
        (value, present) = merge(a, &value, &present, &reset, &reset_mask)?;
        reverse.push((a.bounds(value.lower(), value.upper())?, present.clone()));
    }
    reverse.reverse();
    forward
        .into_iter()
        .zip(masks)
        .zip(reverse)
        .map(|((forward, forward_mask), (reverse, reverse_mask))| {
            Ok(merge(a, &forward, &forward_mask, &reverse, &reverse_mask)?.0)
        })
        .collect()
}
