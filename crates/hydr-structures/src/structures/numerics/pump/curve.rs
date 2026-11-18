//! Closed transformed-domain and declared-zero-region admission before Solvr Linear evaluation.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::ops::IndexingOps;
use numr::ops::ReduceOps;
use numr::ops::ShapeOps;
use numr::ops::TypeConversionOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::tensor::{
    TensorStructureContext,
    pump::{TensorPumpContext, TensorPumpGroup, TensorPumpInput},
};
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::tensor::Tensor;
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};

/// Reference curve at every lane's normalized head, each `[n]`.
pub(super) struct Curve<R: Runtime<DType = DType>> {
    /// Reference discharge.
    pub reference: Tensor<R>,
    /// One where a running lane lies in the declared zero-flow region.
    pub zero_region: Tensor<R>,
    /// `|dQ_ref / dh|` of the segment holding the query, where the reference flow is positive.
    pub slope: Tensor<R>,
}

pub(super) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    pump: &TensorPumpContext<R>,
    operation: &TensorPumpInput<R>,
    normalized: &super::normalization::Normalized<R>,
) -> Result<Curve<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::interpolate::Interp1dAlgorithms<R>,
{
    let c = &context.client;
    let mut flows = Vec::with_capacity(pump.groups.len());
    let mut zeros = Vec::with_capacity(pump.groups.len());
    let mut slopes = Vec::with_capacity(pump.groups.len());
    for group in &pump.groups {
        let running = c.index_select(&operation.running, 0, &group.indices)?;
        let lower = c.index_select(&normalized.lower, 0, &group.indices)?;
        let upper = c.index_select(&normalized.upper, 0, &group.indices)?;
        let outside = c.mul(
            &running,
            &c.maximum(
                &c.lt(&upper, &group.minimum)?,
                &c.gt(&lower, &group.maximum)?,
            )?,
        )?;
        reject(
            context,
            group,
            &outside,
            Some(StructureInputErrorKind::UnsupportedRegime),
            "complete pump normalized-head interval lies outside closed curve domain",
        )?;
        let overlap = c.mul(
            &running,
            &c.maximum(
                &c.lt(&lower, &group.minimum)?,
                &c.gt(&upper, &group.maximum)?,
            )?,
        )?;
        reject(
            context,
            group,
            &overlap,
            None,
            "pump normalized-head interval overlaps curve endpoint",
        )?;
        let spans = c.mul(
            &running,
            &c.mul(
                &group.has_zero,
                &c.mul(
                    &c.lt(&lower, &group.zero_start)?,
                    &c.ge(&upper, &group.zero_start)?,
                )?,
            )?,
        )?;
        reject(
            context,
            group,
            &spans,
            None,
            "pump normalization interval spans positive and zero-flow regions",
        )?;
        let zero_region = c.mul(
            &running,
            &c.mul(&group.has_zero, &c.ge(&lower, &group.zero_start)?)?,
        )?;
        let query = c.where_cond(
            &running,
            &c.index_select(&normalized.query, 0, &group.indices)?,
            &group.minimum,
        )?;
        let reference = c.interp1d(&group.heads, &group.flows, &query, InterpMethod::Linear)?;
        let invalid = c.maximum(
            &c.cast(&c.isnan(&reference)?, reference.dtype())?,
            &c.cast(&c.isinf(&reference)?, reference.dtype())?,
        )?;
        reject(
            context,
            group,
            &invalid,
            None,
            "pump reference discharge is nonfinite",
        )?;
        reject(
            context,
            group,
            &c.lt(&reference, &group.zero)?,
            None,
            "pump reference discharge is negative",
        )?;
        let positive = c.sub(&running, &zero_region)?;
        reject(
            context,
            group,
            &c.mul(&positive, &c.le(&reference, &group.zero)?)?,
            None,
            "certified positive pump curve support returns zero discharge",
        )?;
        // A query on a knot lies in both of its segments and takes the steeper slope.
        let column = query.reshape(&[query.numel(), 1])?;
        let inside = c.mul(
            &c.ge(&column, &group.segment_heads[0])?,
            &c.le(&column, &group.segment_heads[1])?,
        )?;
        let slope = c.max(&c.mul(&inside, &group.segment_slope)?, &[1], false)?;
        slopes.push(c.where_cond(&positive, &slope, &group.zero)?);
        flows.push(c.where_cond(&running, &reference, &group.zero)?);
        zeros.push(zero_region);
    }
    let reference = c.index_put(
        &context.zero,
        0,
        &pump.all_indices,
        &c.cat(&flows.iter().collect::<Vec<_>>(), 0)?,
    )?;
    let zero_region = c.index_put(
        &context.zero,
        0,
        &pump.all_indices,
        &c.cat(&zeros.iter().collect::<Vec<_>>(), 0)?,
    )?;
    let slope = c.index_put(
        &context.zero,
        0,
        &pump.all_indices,
        &c.cat(&slopes.iter().collect::<Vec<_>>(), 0)?,
    )?;
    Ok(Curve {
        reference,
        zero_region,
        slope,
    })
}

fn reject<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    group: &TensorPumpGroup<R>,
    mask: &Tensor<R>,
    kind: Option<StructureInputErrorKind>,
    quantity: &str,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::interpolate::Interp1dAlgorithms<R>,
{
    if !context.any(mask)? {
        return Ok(());
    }
    let rejected = context.client.argmax(mask, 0, false)?.item::<i64>()?;
    let lane = usize::try_from(rejected)
        .ok()
        .filter(|&lane| lane < mask.numel())
        .ok_or_else(|| {
            StructureError::numerical("pump group rejection mask contradicts reduction")
        })?;
    let index = *group.slots.get(lane).ok_or_else(|| {
        StructureError::numerical("pump group index does not cover rejected curve lane")
    })?;
    let element = context.elements.get(index).ok_or_else(|| {
        StructureError::numerical("pump group index does not cover rejected curve lane")
    })?;
    let message = format!(
        "structure={:?},{quantity}: supply supported heads/speed or resolve the normalized-head uncertainty",
        element.id
    );
    Err(match kind {
        Some(kind) => StructureError::input(kind, message).at(index),
        None => StructureError::numerical(message).at(index),
    })
}
