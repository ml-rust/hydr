use super::paired_sources::{PairedSources, SelectedRows};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{AreaDirection, AreaQuery, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct Candidate<R: Runtime<DType = DType>> {
    pub(super) owner: Tensor<R>,
    pub(super) query: Option<SupportedAreaQuery<R>>,
}

pub(super) fn singleton<R>(
    parent: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    cap: bool,
) -> HydraulicResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(
        parent.geometry.client(),
        "wet root singleton support",
        |a| {
            let c = a.client();
            let n = selected.rows.numel();
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            let parameter = a.point(&zero)?;
            let scale = a.expansion_point(&Tensor::ones(&[n], a.dtype(), c.device())?)?;
            let query = if cap {
                AreaQuery::cap_deficit(a, &selected.bands, &scale, &parameter)?
            } else {
                AreaQuery::source_affine(
                    a,
                    &selected.bands,
                    &selected.sources[0],
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
            };
            c.copy_into(&query.original_rows, &selected.original_rows)?;
            let support = query.select_supported(a, &selected.bands)?;
            Ok(Candidate {
                owner: support.owner,
                query: support.query,
            })
        },
    )
}

pub(super) fn local<R>(
    parent: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    side: usize,
) -> HydraulicResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(
        parent.geometry.client(),
        "wet root source-local support",
        |a| {
            let c = a.client();
            let n = selected.rows.numel();
            let source = selected.sources.get(side).ok_or_else(|| {
                super::chart_admission::unresolved(
                    "Wet root source side is outside the pair. Select side zero or one",
                )
            })?;
            let area = source.requested_areas_m2();
            let capacity = selected
                .boundaries
                .boundary(a, selected.boundaries.cap_indices())?;
            let deficit = a.expansion_sub(&capacity, area)?;
            let deficit = a.expansion_interval(&deficit)?;
            let area = a.expansion_interval(area)?;
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            // An unproved positive extent gives zero trial displacement, never a positive floor.
            let below = c.mul_scalar(
                &c.where_cond(&c.gt(area.lower(), &zero)?, area.lower(), &zero)?,
                -0.5,
            )?;
            let above = c.mul_scalar(
                &c.where_cond(&c.gt(deficit.lower(), &zero)?, deficit.lower(), &zero)?,
                0.5,
            )?;
            let parameter = a.bounds(&below, &above)?;
            let scale = a.expansion_point(&Tensor::ones(&[n], a.dtype(), c.device())?)?;
            let query = AreaQuery::source_affine(
                a,
                &selected.bands,
                source,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?;
            c.copy_into(&query.original_rows, &selected.original_rows)?;
            let support = query.select_supported(a, &selected.bands)?;
            Ok(Candidate {
                owner: support.owner,
                query: support.query,
            })
        },
    )
}
