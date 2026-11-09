//! Right-sided accepted-state secant area through existing Numr lookup primitives.

use crate::cpu::{CpuTensor, level_pool::CpuPoolContext};
use crate::storage::LevelPoolError;
use numr::ops::{BinaryOps, IndexingOps, ShapeOps, SortingOps};

pub(super) fn select(
    context: &CpuPoolContext,
    volume: &CpuTensor,
) -> Result<(CpuTensor, CpuTensor), LevelPoolError> {
    let c = &context.client;
    let mut areas = Vec::with_capacity(context.groups.len());
    let mut segments = Vec::with_capacity(context.groups.len());
    for group in &context.groups {
        let query = c.index_select(volume, 0, &group.indices)?;
        let insertion = c.searchsorted(&group.volumes, &query, true)?;
        let zero = c.sub(&group.index_one, &group.index_one)?;
        let segment = c.maximum(
            &zero,
            &c.minimum(&c.sub(&insertion, &group.index_one)?, &group.index_last)?,
        )?;
        areas.push(c.index_select(&group.areas, 0, &segment)?);
        segments.push(segment);
    }
    let area = c.index_select(
        &c.cat(&areas.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    let segment = c.index_select(
        &c.cat(&segments.iter().collect::<Vec<_>>(), 0)?,
        0,
        &context.inverse_permutation,
    )?;
    Ok((area, segment))
}
