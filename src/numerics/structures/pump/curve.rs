//! Closed transformed-domain and declared-zero-region admission before Solvr Linear evaluation.

use crate::cpu::{
    CpuTensor,
    structures::{
        CpuStructureContext,
        pump::{CpuPumpContext, CpuPumpGroup, CpuPumpInput},
    },
};
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ShapeOps, TypeConversionOps, UnaryOps,
};
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};

pub(super) fn evaluate(
    context: &CpuStructureContext,
    pump: &CpuPumpContext,
    operation: &CpuPumpInput,
    normalized: &super::normalization::Normalized,
) -> Result<(CpuTensor, CpuTensor), StructureError> {
    let c = &context.client;
    let mut flows = Vec::with_capacity(pump.groups.len());
    let mut zeros = Vec::with_capacity(pump.groups.len());
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
    Ok((reference, zero_region))
}

fn reject(
    context: &CpuStructureContext,
    group: &CpuPumpGroup,
    mask: &CpuTensor,
    kind: Option<StructureInputErrorKind>,
    quantity: &str,
) -> Result<(), StructureError> {
    if !context.any(mask)? {
        return Ok(());
    }
    let flags = mask.try_to_vec::<f64>()?;
    let lane = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
        StructureError::numerical("pump group rejection mask contradicts reduction")
    })?;
    let index = *group.slots.get(lane).ok_or_else(|| {
        StructureError::numerical("pump group index does not cover rejected curve lane")
    })?;
    let message = format!(
        "structure={:?},{quantity}: supply supported heads/speed or resolve the normalized-head uncertainty",
        context.elements[index].id
    );
    Err(match kind {
        Some(kind) => StructureError::input(kind, message).at(index),
        None => StructureError::numerical(message).at(index),
    })
}
