use super::super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::super::{AreaAnchor, SupportedAreaQuery, area_expression::resolve_anchor};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct TargetCoordinates<R: Runtime<DType = DType>> {
    pub above: Interval<R>,
    pub below: Interval<R>,
    pub at_start: Tensor<R>,
    pub at_end: Tensor<R>,
}

pub(super) fn matrix<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    shape: &[usize; 2],
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(a.expansion_bounds(
        &value.high().unsqueeze(1)?.broadcast_to(shape)?,
        &value.low().unsqueeze(1)?.broadcast_to(shape)?,
        &value.abs_error().unsqueeze(1)?.broadcast_to(shape)?,
    )?)
}

pub(super) fn target<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    endpoint: &SupportedAreaQuery<R>,
    boundaries: &BoundaryAreas<R>,
    indices: &Tensor<R>,
) -> HydraulicResult<TargetCoordinates<R>>
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
    let b = bands.band_count;
    let shape = [n, b];
    let zero = Tensor::zeros(&shape, DType::F64, c.device())?;
    let anchor = resolve_anchor(a, &endpoint.query, boundaries)?;
    let anchor = matrix(a, &anchor, &shape)?;
    let boundary = |a: &mut Arithmetic<'_, R>, index| -> HydraulicResult<CertifiedExpansion<R>> {
        let values = boundaries.areas();
        Ok(a.expansion_bounds(
            &values.high().narrow(1, index, b)?,
            &values.low().narrow(1, index, b)?,
            &values.abs_error().narrow(1, index, b)?,
        )?)
    };
    let start = boundary(a, 0)?;
    let end = boundary(a, 1)?;
    let identity = match &endpoint.query.anchor {
        AreaAnchor::Zero => Some(Tensor::zeros(&[n], DType::I64, c.device())?),
        #[cfg(test)]
        AreaAnchor::BandStart(index) => Some(index.clone()),
        #[cfg(test)]
        AreaAnchor::BandEnd(index) => Some(c.add_scalar(index, 1.0)?),
        AreaAnchor::Cap => Some(boundaries.cap_indices().clone()),
        AreaAnchor::Source { .. } => None,
    };
    let (at_start, at_end) = if let Some(index) = identity {
        let index = index.reshape(&[n, 1])?;
        (
            c.cast(&c.eq(&index, indices)?, DType::F64)?,
            c.cast(&c.eq(&index, &c.add_scalar(indices, 1.0)?)?, DType::F64)?,
        )
    } else {
        (zero.clone(), zero.clone())
    };
    let cancel = |a: &mut Arithmetic<'_, R>,
                  value: &CertifiedExpansion<R>,
                  shared: &Tensor<R>|
     -> HydraulicResult<CertifiedExpansion<R>> {
        Ok(a.expansion_bounds(
            &c.where_cond(shared, &zero, value.high())?,
            &c.where_cond(shared, &zero, value.low())?,
            &c.where_cond(shared, &zero, value.abs_error())?,
        )?)
    };
    // Mask shared anchors before subtraction. Absolute anchor uncertainty remains in the separate endpoint expression.
    let above_anchor = cancel(a, &anchor, &at_start)?;
    let above_boundary = cancel(a, &start, &at_start)?;
    let below_anchor = cancel(a, &anchor, &at_end)?;
    let below_boundary = cancel(a, &end, &at_end)?;
    let above = a.expansion_sub(&above_anchor, &above_boundary)?;
    let below = a.expansion_sub(&below_boundary, &below_anchor)?;
    let offset = matrix(a, &endpoint.query.offsets.lower, &shape)?;
    let offset_zero = c.mul(
        &c.eq(offset.high(), &zero)?,
        &c.mul(
            &c.eq(offset.low(), &zero)?,
            &c.eq(offset.abs_error(), &zero)?,
        )?,
    )?;
    let above = a.expansion_add(&above, &offset)?;
    let below = a.expansion_sub(&below, &offset)?;
    Ok(TargetCoordinates {
        above: a.expansion_interval(&above)?,
        below: a.expansion_interval(&below)?,
        at_start: c.mul(&at_start, &offset_zero)?,
        at_end: c.mul(&at_end, &offset_zero)?,
    })
}
