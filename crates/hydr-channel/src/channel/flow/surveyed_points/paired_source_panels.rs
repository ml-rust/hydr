use super::{
    paired_source_cache::{self, SourcePotentialCache},
    paired_sources::PairedSources,
    potential_range::PotentialEvaluation,
    source_binding::SourcePanelLevel,
    wave_curve_types::{CurveValue, unavailable},
};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::sync::Arc;

/// Owned checked potentials and their authenticated source/panel policy.
pub(super) struct SourcePanelValues<R: Runtime<DType = DType>> {
    pub(super) values: [CurveValue<R>; 2],
    pub(super) level: SourcePanelLevel,
    pub(super) parent_rows: [Tensor<R>; 2],
    side: [usize; 2],
}

impl<R> PairedSources<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    pub(super) fn panel_level(&self, index: usize) -> HydraulicResult<SourcePanelLevel> {
        self.root_witness()?.panel_level(index)
    }

    pub(super) fn source_values_at_level(
        &mut self,
        level: &SourcePanelLevel,
        rows: [&Tensor<R>; 2],
    ) -> HydraulicResult<SourcePanelValues<R>> {
        self.check_root_witness(level.witness())?;
        admit_rows(self, rows)?;
        let c = self.geometry.client();
        if rows.iter().all(|rows| rows.numel() == 0) {
            let values = checked(c, "empty source panel request", |a| {
                Ok([unavailable(a, 0)?, unavailable(a, 0)?])
            })?;
            return owned_values(c, level, rows, values);
        }
        if level.index() == 0 {
            let values = self.source_values(rows)?;
            return owned_values(c, level, rows, values);
        }
        let position = self
            .panel_cache
            .iter()
            .position(|(index, _)| *index == level.index());
        let position = if let Some(position) = position {
            position
        } else {
            let count = self.original_rows.numel();
            checked::<R, _>(c, "source panel cache storage admission", |a| {
                let mut levels = 1usize;
                let mut panels = self.evaluation.panels;
                for _ in 0..self.evaluation.config.max_refinements {
                    let Some(next) = panels.checked_mul(2) else {
                        break;
                    };
                    if next > self.evaluation.config.max_simpson_panels {
                        break;
                    }
                    panels = next;
                    levels = levels
                        .checked_add(1)
                        .ok_or_else(|| storage_error(count, a.dtype(), levels))?;
                }
                storage::<R>(count, a.dtype(), levels)?;
                Ok(())
            })?;
            let caches = checked::<R, _>(c, "source panel cache allocation", |a| {
                Ok([
                    SourcePotentialCache::with_query(a, Arc::clone(&self.cache[0].query), count)?,
                    SourcePotentialCache::with_query(a, Arc::clone(&self.cache[1].query), count)?,
                ])
            })?;
            self.panel_cache.try_reserve_exact(1).map_err(|_| {
                storage_error(
                    count,
                    self.cache[0].ready.dtype(),
                    self.panel_cache.len() + 1,
                )
            })?;
            self.panel_cache.push((level.index(), caches));
            self.panel_cache.len() - 1
        };
        let evaluation = PotentialEvaluation {
            config: self.evaluation.config,
            panels: level.panels(),
            work: self.evaluation.work,
        };
        let [left, right] = &mut self.panel_cache[position].1;
        let values = [
            paired_source_cache::ensure(
                left,
                self.geometry,
                self.bands,
                self.sources[0],
                self.boundaries,
                &evaluation,
                rows[0],
            )?,
            paired_source_cache::ensure(
                right,
                self.geometry,
                self.bands,
                self.sources[1],
                self.boundaries,
                &evaluation,
                rows[1],
            )?,
        ];
        owned_values(c, level, rows, values)
    }
}

impl<R: Runtime<DType = DType>> SourcePanelValues<R> {
    pub(super) fn checked_side(
        &self,
        sources: &PairedSources<'_, R>,
        side: usize,
        parents: &Tensor<R>,
    ) -> HydraulicResult<&CurveValue<R>>
    where
        R::Client: TensorOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        sources.check_root_witness(self.level.witness())?;
        let Some(rows) = self.parent_rows.get(side) else {
            return Err(HydraulicError::Numerical {
                context: format!("Source panel side={side} is absent. Supply side 0 or 1"),
            });
        };
        if self.side.get(side) != Some(&side)
            || parents.shape() != rows.shape()
            || parents.dtype() != DType::I64
            || !parents.device().is_same(rows.device())
        {
            return Err(HydraulicError::Numerical { context: "Source panel parent/side provenance differs. Preserve the checked source request".into() });
        }
        if sources
            .geometry
            .client()
            .nonzero(&sources.geometry.client().ne(parents, rows)?)?
            .numel()
            != 0
        {
            return Err(HydraulicError::Numerical {
                context:
                    "Source panel parent identities differ. Preserve the checked source request"
                        .into(),
            });
        }
        self.values
            .get(side)
            .ok_or_else(|| HydraulicError::Numerical {
                context: "Source panel value side is absent. Preserve the checked source pair"
                    .into(),
            })
    }
}

fn owned_values<R>(
    c: &R::Client,
    level: &SourcePanelLevel,
    rows: [&Tensor<R>; 2],
    values: [CurveValue<R>; 2],
) -> HydraulicResult<SourcePanelValues<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + RuntimeClient<R>,
{
    let copy = |rows: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let owned = Tensor::empty(rows.shape(), DType::I64, c.device())?;
        c.copy_into(&owned, rows)?;
        Ok(owned)
    };
    Ok(SourcePanelValues {
        values,
        level: level.clone(),
        parent_rows: [copy(rows[0])?, copy(rows[1])?],
        side: [0, 1],
    })
}

pub(super) fn admit_rows<R>(
    sources: &PairedSources<'_, R>,
    rows: [&Tensor<R>; 2],
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = sources.geometry.client();
    let count = sources.original_rows.numel();
    if count as u128 > (1_u128 << f64::MANTISSA_DIGITS) {
        return Err(storage_error(count, sources.cache[0].ready.dtype(), 1));
    }
    for (side, value) in ["left", "right"].into_iter().zip(rows) {
        if value.shape() != [value.numel()]
            || value.dtype() != DType::I64
            || !value.device().is_same(c.device())
        {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Source panel side={side}, shape={:?}, rank={}, dtype={:?}, device={}: supply rank-one I64 parent rows on device={}",
                    value.shape(),
                    value.shape().len(),
                    value.dtype(),
                    value.device().id(),
                    c.device().id(),
                ),
            });
        }
    }
    checked::<R, _>(c, "source panel parent admission", |_a| {
        let zero = Tensor::zeros(&[], DType::I64, c.device())?;
        let limit = Tensor::full_scalar(&[], DType::I64, count as f64, c.device())?;
        for (side, value) in ["left", "right"].into_iter().zip(rows) {
            if value.numel() != 0 {
                let outside = c.maximum(&c.lt(value, &zero)?, &c.ge(value, &limit)?)?;
                let outside = c.nonzero(&outside)?.flatten()?;
                if outside.numel() != 0 {
                    let first = outside.narrow(0, 0, 1)?;
                    let parent = c.index_select(value, 0, &first)?.item::<i64>()?;
                    return Err(HydraulicError::Numerical {
                        context: format!(
                            "Source panel side={side}, parent={parent}, captured_count={count}: supply parent rows in [0, {count})"
                        ),
                    });
                }
            }
        }
        Ok(())
    })
}

fn storage<R: Runtime<DType = DType>>(
    count: usize,
    dtype: DType,
    levels: usize,
) -> HydraulicResult<()> {
    let error = || storage_error(count, dtype, levels);
    let elements = count
        .checked_mul(12)
        .and_then(|n| n.checked_mul(levels))
        .ok_or_else(error)?;
    let bytes = elements
        .checked_mul(dtype.size_in_bytes())
        .and_then(|bytes| {
            levels
                .checked_mul(std::mem::size_of::<(usize, [SourcePotentialCache<R>; 2])>())
                .and_then(|metadata| bytes.checked_add(metadata))
        })
        .ok_or_else(error)?;
    if bytes > isize::MAX as usize
        || count as u128 > (1_u128 << f64::MANTISSA_DIGITS)
        || count > i64::MAX as usize
    {
        return Err(error());
    }
    Ok(())
}
fn storage_error(count: usize, dtype: DType, levels: usize) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!(
            "Source panel cache count={count}, dtype={dtype:?}, levels={levels}, numeric_fields_per_parent=12 exceeds storage/index bounds: bytes_limit={}, exact_row_limit={}, index_limit={}. Split the source graph or reduce panel levels",
            isize::MAX,
            1_u128 << f64::MANTISSA_DIGITS,
            i64::MAX,
        ),
    }
}

#[cfg(test)]
#[path = "paired_source_panels_tests.rs"]
mod tests;
