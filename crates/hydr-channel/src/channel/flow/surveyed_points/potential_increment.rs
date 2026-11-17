use super::potential_absolute::admitted_analytic;
use super::potential_absolute::prefix_potential;
use super::potential_analytic;
use super::potential_numerical::wet_contributions;
use super::potential_range::{
    PotentialContext, PotentialValue, scatter, select_expansion, zero_value,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::select_interval,
        error::{HydraulicResult, checked},
    },
    section::{ConditionalBandSegments, PhysicalEndpoint, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

/// Integrate source-to-endpoint segments directly. Absolute leaves remain unevaluated.
pub(super) fn endpoint_increment<R>(
    context: &PotentialContext<'_, R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<PotentialValue<R>>
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
    let c = context.geometry.client();
    let source_query = context.source_query()?;
    let (target_query, mut result, dry_forward, dry_reverse, wet) = checked(
        c,
        "point signed potential dispatch",
        |a| {
            let target_query = context.query.endpoint(a, context.bands, endpoint)?;
            let target_area =
                target_query.retained_endpoint_area(a, context.bands, context.boundaries)?;
            let delta = source_difference(
                a,
                context.bands,
                &target_query,
                context.source,
                context.boundaries,
            )?
            .lower;
            let n = context.bands.section_owner.numel();
            let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
            let exact_zero = |value: &solvr::common::arithmetic::CertifiedExpansion<R>| -> HydraulicResult<Tensor<R>> {
                Ok(c.mul(&c.eq(value.high(), &zero)?,
                    &c.mul(&c.eq(value.low(), &zero)?, &c.eq(value.abs_error(), &zero)?)?)?)
            };
            let equal = exact_zero(&delta)?;
            let nonzero_rows = c.nonzero(&c.eq(&equal, &zero)?)?.flatten()?;
            let mut result = zero_value(a, n)?;
            let mut analytic_mask = zero.clone();
            if nonzero_rows.numel() != 0 {
                let bands = context.bands.select(a, &nonzero_rows)?;
                let source_query = source_query.select(a, &nonzero_rows)?;
                let target_query = target_query.select(a, &nonzero_rows)?;
                let source_area =
                    select_expansion(a, context.source.requested_areas_m2(), &nonzero_rows)?;
                let target_area = select_expansion(a, &target_area, &nonzero_rows)?;
                let delta = select_expansion(a, &delta, &nonzero_rows)?;
                let source_state = context
                    .geometry
                    .recover_query(a, &bands, &source_query)?
                    .state;
                let target_state = context
                    .geometry
                    .recover_query(a, &bands, &target_query)?
                    .state;
                let source = admitted_analytic(
                    a,
                    &bands,
                    &source_area,
                    &source_state.depth,
                    context.evaluation.config.gravity_m_per_s2,
                )?;
                let target = admitted_analytic(
                    a,
                    &bands,
                    &target_area,
                    &target_state.depth,
                    context.evaluation.config.gravity_m_per_s2,
                )?;
                let analytic = potential_analytic::difference(
                    a,
                    &source,
                    &target,
                    &source_state.top_width,
                    &target_state.top_width,
                    &delta,
                    context.evaluation.config.gravity_m_per_s2,
                )?;
                let local_zero = Tensor::zeros(&[nonzero_rows.numel()], DType::F64, c.device())?;
                let local = c
                    .nonzero(&c.ne(&analytic.eligible, &local_zero)?)?
                    .flatten()?;
                let analytic_rows = c.index_select(&nonzero_rows, 0, &local)?;
                let value = PotentialValue {
                    retained: select_expansion(a, &analytic.retained_increment, &local)?,
                    physical: select_interval(a, &analytic.physical_increment, &local)?,
                    analytic: None,
                };
                result = scatter(a, result, &analytic_rows, &value)?;
                analytic_mask = c.index_put(&zero, 0, &nonzero_rows, &analytic.eligible)?;
            }
            let remaining = c.eq(&c.maximum(&equal, &analytic_mask)?, &zero)?;
            let source_dry = exact_zero(context.source.requested_areas_m2())?;
            let target_dry = exact_zero(&target_area)?;
            let forward = c.nonzero(&c.mul(&remaining, &source_dry)?)?.flatten()?;
            let reverse = c.nonzero(&c.mul(&remaining, &target_dry)?)?.flatten()?;
            let wet = c
                .nonzero(&c.mul(
                    &remaining,
                    &c.eq(&c.maximum(&source_dry, &target_dry)?, &zero)?,
                )?)?
                .flatten()?;
            Ok((target_query, result, forward, reverse, wet))
        },
    )?;
    for (rows, query, reverse) in [
        (&dry_forward, &target_query, false),
        (&dry_reverse, &source_query, true),
    ] {
        if rows.numel() == 0 {
            continue;
        }
        let (bands, query, boundaries) = checked(c, "point signed dry prefix selection", |a| {
            Ok((
                context.bands.select(a, rows)?,
                query.select(a, rows)?,
                context.boundaries.select(a, rows)?,
            ))
        })?;
        let mut value = prefix_potential(context, &bands, &query, &boundaries)?;
        if reverse {
            value = checked(c, "point reversed dry prefix", |a| {
                let zero = zero_value(a, rows.numel())?;
                Ok(PotentialValue {
                    retained: a.expansion_sub(&zero.retained, &value.retained)?,
                    physical: a.sub(&zero.physical, &value.physical)?,
                    analytic: None,
                })
            })?;
        }
        result = checked(c, "point signed dry prefix assembly", |a| {
            scatter(a, result, rows, &value)
        })?;
    }
    if wet.numel() != 0 {
        let segments = checked(c, "point signed potential segments", |a| {
            ConditionalBandSegments::prepare(
                a,
                context.bands,
                context.source,
                &target_query,
                context.boundaries,
            )
        })?;
        let value = wet_contributions(context, &segments, &wet)?;
        result = checked(c, "point signed potential assembly", |a| {
            scatter(a, result, &wet, &value)
        })?;
    }
    Ok(result)
}
