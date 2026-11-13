use super::{analytic_bottom, paired_sources::PairedSources, source_sonic_types::SonicSide};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked},
    section::{AreaAnchor, AreaDirection, AreaQuery, PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{ArithmeticError, CertifiedExpansion};

pub(super) fn chart<R>(
    sources: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    side: SonicSide,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
) -> HydraulicResult<SupportedAreaQuery<R>>
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
        sources.geometry.client(),
        "source sonic exact source-area chart",
        |a| {
            let selected = sources.select_sources(a, rows)?;
            let scale = selected.sources[side.index()].requested_areas_m2();
            let parameter = a.bounds(lower, upper)?;
            let query = AreaQuery::band_affine(
                a,
                &selected.bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                scale,
                &parameter,
            )?;
            a.client()
                .copy_into(&query.original_rows, &selected.original_rows)?;
            query.prove_support(a, &selected.bands)
        },
    )
}

pub(super) fn seed<R>(
    sources: &PairedSources<'_, R>,
    rows: &Tensor<R>,
    side: SonicSide,
    epsilon: &CertifiedExpansion<R>,
    chart: &SupportedAreaQuery<R>,
) -> HydraulicResult<Tensor<R>>
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
    let (mut seed, epsilon, area, width, slope, rectangular, pointed) =
        checked::<R, _>(c, "source sonic normalized seed operands", |a| {
            let selected = sources.select_sources(a, rows)?;
            let source = &selected.sources[side.index()];
            let source_area = a.expansion_interval(source.requested_areas_m2())?;
            let analytic = analytic_bottom::source(
                a,
                &selected.bands,
                source.requested_areas_m2(),
                source.depth_m(),
                sources.evaluation.config.gravity_m_per_s2,
            )?;
            let epsilon = a.expansion_interval(epsilon)?;
            let column = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
            };
            Ok((
                chart.query.parameter.upper().clone(),
                epsilon.lower().clone(),
                source_area.upper().clone(),
                column(selected.bands.start_width.lower())?,
                column(selected.bands.slope.lower())?,
                analytic.rectangular,
                analytic.pointed,
            ))
        })?;
    for (mask, factor, pointed) in [(&rectangular, &width, false), (&pointed, &slope, true)] {
        let selected = c.nonzero(mask)?.flatten()?;
        if selected.numel() == 0 {
            continue;
        }
        let factor = c.index_select(factor, 0, &selected)?;
        let epsilon = c.index_select(&epsilon, 0, &selected)?;
        let area = c.index_select(&area, 0, &selected)?;
        let candidate = checked::<R, _>(c, "source sonic factored normalized seed", |a| {
            let gravity = Tensor::full_scalar(
                &[],
                a.dtype(),
                sources.evaluation.config.gravity_m_per_s2,
                c.device(),
            )?;
            let coefficient = Tensor::full_scalar(
                &[],
                a.dtype(),
                if pointed { 625.0 } else { 9.0 },
                c.device(),
            )?;
            if pointed {
                let two = Tensor::full_scalar(&[], a.dtype(), 2.0, c.device())?;
                Ok(a.positive_product_ratio_lower(
                    &[&two, &factor, &epsilon, &epsilon, &epsilon, &epsilon],
                    &[&coefficient, &gravity, &gravity, &area],
                )?)
            } else {
                Ok(a.positive_product_ratio_lower(
                    &[&factor, &epsilon, &epsilon],
                    &[&coefficient, &gravity, &area],
                )?)
            }
        });
        match candidate {
            Ok(candidate) => {
                // This represented coordinate requests a sign check. It never certifies a root by itself.
                let upper = c.index_select(chart.query.parameter.upper(), 0, &selected)?;
                let candidate = c.minimum(&c.mul_scalar(&candidate, 2.0)?, &upper)?;
                seed = c.index_put(&seed, 0, &selected, &candidate)?;
            }
            Err(HydraulicError::Arithmetic(ArithmeticError::Range { .. })) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(seed)
}

pub(super) fn derivative<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &crate::channel::section::HydraulicGeometry<R>,
    bands: &crate::channel::section::GatheredBands<R>,
    chart: &SupportedAreaQuery<R>,
    gravity: f64,
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
    // Phi' contributes one sound/area term. Convex pressure proves c' nonnegative across every admitted band join.
    let scale = a.expansion_interval(&chart.query.scale)?;
    let gravity = a.point(&Tensor::full_scalar(
        &[],
        a.dtype(),
        gravity,
        a.client().device(),
    )?)?;
    let gravity = a.sqrt_nonnegative(&gravity)?;
    let c = a.client();
    let upper_endpoint = chart.endpoint(a, bands, PhysicalEndpoint::Upper)?;
    let state = geometry.recover_query(a, bands, &upper_endpoint)?.state;
    let zero = Tensor::zeros(upper_endpoint.area.upper().shape(), a.dtype(), c.device())?;
    let local = c.mul(
        &c.gt(upper_endpoint.area.upper(), &zero)?,
        &c.gt(state.top_width.upper(), &zero)?,
    )?;
    // Complete current-bracket maxima preserve the scale of tiny roots. Cap maxima remain a proved fallback.
    let area_upper = c.where_cond(&local, upper_endpoint.area.upper(), bands.cap_area.upper())?;
    let width_upper = c.where_cond(&local, state.top_width.upper(), bands.cap_width.upper())?;
    let area = a.point(&area_upper)?;
    let width = a.point(&width_upper)?;
    let area = a.sqrt_nonnegative(&area)?;
    let width = a.sqrt_nonnegative(&width)?;
    Ok(a.positive_product_ratio_lower(
        &[scale.lower(), gravity.lower()],
        &[area.upper(), width.upper()],
    )?)
}
