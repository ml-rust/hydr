use super::potential_absolute::endpoint_potential;
use super::potential_range::{
    PotentialContext, PotentialEvaluation, PotentialValue, scatter, select_expansion,
};
use super::wave_curve_types::{CurveValue, unavailable};
use crate::channel::{
    flow::surveyed::{
        equation_ops::select_interval,
        error::{Arithmetic, HydraulicResult, checked},
    },
    section::{
        BoundaryAreas, GatheredBands, HydraulicGeometry, PhysicalEndpoint,
        PrismaticAreaPointRecovery, SupportedAreaQuery,
    },
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        ScatterReduceOp, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::sync::Arc;

pub(super) struct SourcePotentialCache<R: Runtime<DType = DType>> {
    pub(super) query: Arc<SupportedAreaQuery<R>>,
    pub(super) ready: Tensor<R>,
    pub(super) value: PotentialValue<R>,
}

impl<R> SourcePotentialCache<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(
        a: &mut Arithmetic<'_, R>,
        query: SupportedAreaQuery<R>,
        count: usize,
    ) -> HydraulicResult<Self> {
        Self::with_query(a, Arc::new(query), count)
    }

    pub(super) fn with_query(
        a: &mut Arithmetic<'_, R>,
        query: Arc<SupportedAreaQuery<R>>,
        count: usize,
    ) -> HydraulicResult<Self> {
        let zero = Tensor::zeros(&[count], a.dtype(), a.client().device())?;
        Ok(Self {
            query,
            value: PotentialValue {
                retained: a.expansion_point(&zero)?,
                physical: a.point(&zero)?,
                analytic: None,
            },
            ready: zero,
        })
    }
}

/// Rows are admitted fixed-parent positions. Duplicate requests fill each missing parent row once.
pub(super) fn ensure<R>(
    cache: &mut SourcePotentialCache<R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    source: &PrismaticAreaPointRecovery<R>,
    boundaries: &BoundaryAreas<R>,
    evaluation: &PotentialEvaluation<'_>,
    rows: &Tensor<R>,
) -> HydraulicResult<CurveValue<R>>
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
    let c = geometry.client();
    if rows.numel() == 0 {
        return checked(c, "paired empty source request", |a| unavailable(a, 0));
    }
    let missing = checked::<R, _>(c, "paired source potential requests", |a| {
        let zero = Tensor::zeros(cache.ready.shape(), a.dtype(), c.device())?;
        let requested = c.scatter_reduce(
            &zero,
            0,
            rows,
            &Tensor::ones(rows.shape(), a.dtype(), c.device())?,
            ScatterReduceOp::Max,
            true,
        )?;
        let requested = c.minimum(
            &c.maximum(&requested, &zero)?,
            &Tensor::ones(zero.shape(), a.dtype(), c.device())?,
        )?;
        let missing = c.mul(&requested, &c.eq(&cache.ready, &zero)?)?;
        Ok(c.nonzero(&missing)?.flatten()?)
    })?;
    if missing.numel() != 0 {
        let (bands, source, query, boundaries) =
            checked(c, "paired source potential selection", |a| {
                Ok((
                    bands.select(a, &missing)?,
                    source.select(a, &missing)?,
                    cache.query.select(a, &missing)?,
                    boundaries.select(a, &missing)?,
                ))
            })?;
        let context = PotentialContext::new(
            geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: evaluation.config,
                panels: evaluation.panels,
                work: evaluation.work,
            },
        )?;
        let value = endpoint_potential(&context, PhysicalEndpoint::Lower)?;
        let (committed, ready) = checked(c, "paired source potential cache completion", |a| {
            let previous = PotentialValue {
                retained: a.expansion_bounds(
                    cache.value.retained.high(),
                    cache.value.retained.low(),
                    cache.value.retained.abs_error(),
                )?,
                physical: a.bounds(cache.value.physical.lower(), cache.value.physical.upper())?,
                analytic: None,
            };
            let committed = scatter(a, previous, &missing, &value)?;
            let ready = c.index_put(
                &cache.ready,
                0,
                &missing,
                &Tensor::ones(missing.shape(), a.dtype(), c.device())?,
            )?;
            Ok((committed, ready))
        })?;
        cache.value = committed;
        cache.ready = ready;
    }
    checked(c, "paired cached source potential", |a| {
        Ok(CurveValue {
            retained: select_expansion(a, &cache.value.retained, rows)?,
            physical: select_interval(a, &cache.value.physical, rows)?,
            available: c.index_select(&cache.ready, 0, rows)?,
        })
    })
}
