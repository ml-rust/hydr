use super::{paired_sources::PairedSources, wet_depth_types::WetDepthStatus};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::Interval,
    optimize::{CertifiedRootResult, CertifiedRootStatus, CertifiedRootTolerance},
};

pub(super) struct Cached<R: Runtime<DType = DType>> {
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) depth: Interval<R>,
    pub(super) normalized: Interval<R>,
    pub(super) status: Tensor<R>,
}

pub(super) fn update<R>(
    sources: &PairedSources<'_, R>,
    parent: &Tensor<R>,
    cached: Cached<R>,
    result: &CertifiedRootResult<R>,
    rows: &Tensor<R>,
    tolerance: &CertifiedRootTolerance,
) -> HydraulicResult<Cached<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = sources.geometry.client();
    let selected = sources.select_rows(parent, &cached.query)?;
    let (query, selected_parent, selected_query, mut status) = checked::<R, _>(
        c,
        "source sonic selected native domain reconstruction",
        |a| {
            let lower = c.index_put(
                cached.query.query.parameter.lower(),
                0,
                result.root_owner(),
                result.root_lower(),
            )?;
            let upper = c.index_put(
                cached.query.query.parameter.upper(),
                0,
                result.root_owner(),
                result.root_upper(),
            )?;
            let parameter = a.bounds(
                &c.index_select(&lower, 0, rows)?,
                &c.index_select(&upper, 0, rows)?,
            )?;
            let query = cached.query.with_selected_parameter_bounds(
                a,
                &selected.bands,
                rows,
                &parameter,
            )?;
            let selected_parent = c.index_select(parent, 0, rows)?;
            let selected_query = query.select(a, rows)?;
            let native_status = c.index_select(result.status(), 0, rows)?;
            let mut status = Tensor::full_scalar(
                rows.shape(),
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            for (native, physical) in [
                (
                    CertifiedRootStatus::Converged,
                    WetDepthStatus::DepthConverged,
                ),
                (
                    CertifiedRootStatus::AccuracyFloor,
                    WetDepthStatus::AccuracyFloor,
                ),
                (CertifiedRootStatus::WorkLimit, WetDepthStatus::WorkLimit),
            ] {
                let code = Tensor::full_scalar(&[], DType::I64, native as i64 as f64, c.device())?;
                let value = Tensor::full_scalar(
                    rows.shape(),
                    DType::I64,
                    physical as i64 as f64,
                    c.device(),
                )?;
                status = c.where_cond(&c.eq(&native_status, &code)?, &value, &status)?;
            }
            Ok((query, selected_parent, selected_query, status))
        },
    )?;
    let (depth, normalized) =
        super::point_root_depth::certificates(sources, &selected_parent, &selected_query)?;
    checked::<R, _>(
        c,
        "source sonic selected cached physical certificates",
        |a| {
            let accepted = super::point_root_depth::satisfied(a, &normalized, tolerance)?;
            let converged = Tensor::full_scalar(
                rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let unresolved = Tensor::full_scalar(
                rows.shape(),
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            let other = c.where_cond(&c.eq(&status, &converged)?, &unresolved, &status)?;
            status = c.where_cond(&accepted, &converged, &other)?;
            let depth = a.bounds(
                &c.index_put(cached.depth.lower(), 0, rows, depth.lower())?,
                &c.index_put(cached.depth.upper(), 0, rows, depth.upper())?,
            )?;
            let normalized = a.bounds(
                &c.index_put(cached.normalized.lower(), 0, rows, normalized.lower())?,
                &c.index_put(cached.normalized.upper(), 0, rows, normalized.upper())?,
            )?;
            let status = c.index_put(&cached.status, 0, rows, &status)?;
            Ok(Cached {
                query,
                depth,
                normalized,
                status,
            })
        },
    )
}
