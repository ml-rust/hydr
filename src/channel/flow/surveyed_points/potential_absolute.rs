use super::potential_range::{PotentialContext, PotentialValue, scatter, zero_value};
use super::{analytic_bottom, potential_lanes};
use crate::channel::{
    flow::surveyed::{
        derivatives::{integrand, panel_fourth_bound},
        error::{Arithmetic, HydraulicResult, checked},
        integration::assemble_lane_contributions,
        lane_quadrature::numerical_contributions,
    },
    section::{ConditionalBandSegments, PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::{CertifiedExpansion, Interval},
    integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions},
};

pub(super) fn source_potential<R>(
    context: &PotentialContext<'_, R>,
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
    absolute(context, &context.source_query()?)
}

pub(super) fn endpoint_potential<R>(
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
    let query = checked(context.geometry.client(), "point potential endpoint", |a| {
        context.query.endpoint(a, context.bands, endpoint)
    })?;
    absolute(context, &query)
}

fn absolute<R>(
    context: &PotentialContext<'_, R>,
    query: &SupportedAreaQuery<R>,
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
    let (mut result, rows) = checked(c, "point absolute potential dispatch", |a| {
        let raw = query.retained_endpoint_area(a, context.bands, context.boundaries)?;
        let state = context
            .geometry
            .recover_query(a, context.bands, query)?
            .state;
        let analytic = admitted_analytic(
            a,
            context.bands,
            &raw,
            &state.depth,
            context.evaluation.config.gravity_m_per_s2,
        )?;
        let zero = numr::tensor::Tensor::zeros(raw.high().shape(), DType::F64, c.device())?;
        let exact_dry = c.mul(
            &c.eq(raw.high(), &zero)?,
            &c.mul(&c.eq(raw.low(), &zero)?, &c.eq(raw.abs_error(), &zero)?)?,
        )?;
        let completed = c.maximum(&analytic.eligible, &exact_dry)?;
        let rows = c.nonzero(&c.eq(&completed, &zero)?)?.flatten()?;
        let physical = a.expansion_interval(&analytic.potential)?;
        let physical = a.bounds(&c.maximum(physical.lower(), &zero)?, physical.upper())?;
        let retained = a.expansion_bounds(
            analytic.potential.high(),
            analytic.potential.low(),
            analytic.potential.abs_error(),
        )?;
        Ok((
            PotentialValue {
                retained,
                physical,
                analytic: Some(analytic),
            },
            rows,
        ))
    })?;
    if rows.numel() != 0 {
        let (bands, query, boundaries) = checked(c, "point absolute potential selection", |a| {
            Ok((
                context.bands.select(a, &rows)?,
                query.select(a, &rows)?,
                context.boundaries.select(a, &rows)?,
            ))
        })?;
        let numerical = prefix_potential(context, &bands, &query, &boundaries)?;
        result = checked(c, "point absolute potential assembly", |a| {
            scatter(a, result, &rows, &numerical)
        })?;
    }
    Ok(result)
}

/// A true zero-to-endpoint prefix is its own requested integral, never a difference of absolutes.
pub(super) fn prefix_potential<R>(
    context: &PotentialContext<'_, R>,
    bands: &crate::channel::section::GatheredBands<R>,
    query: &SupportedAreaQuery<R>,
    boundaries: &crate::channel::section::BoundaryAreas<R>,
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
    if bands.section_owner.numel() == 0 {
        return checked(c, "point empty potential prefix", |a| zero_value(a, 0));
    }
    let (lanes, zero_gap) = checked(c, "point potential dry prefix", |a| {
        let prefix = ConditionalBandSegments::prefix_from_exact_dry(a, bands, query, boundaries)?;
        let lanes = potential_lanes::build(a, &prefix, context.evaluation.config.gravity_m_per_s2)?;
        let zero = Tensor::zeros(
            prefix.segments.active_mask().shape(),
            DType::F64,
            c.device(),
        )?;
        let zero_gap = c.maximum(
            prefix.segments.zero_gap_mask(),
            &c.eq(prefix.segments.active_mask(), &zero)?,
        )?;
        Ok((lanes, zero_gap))
    })?;
    let left = Tensor::zeros(&[lanes.lanes.band_index.numel()], DType::F64, c.device())?;
    let right = lanes
        .lanes
        .active
        .reshape(&[lanes.lanes.face_owner.numel()])?;
    let numerical = numerical_contributions(
        c,
        &left,
        &right,
        CertifiedSimpsonOptions {
            panels: context.evaluation.panels,
            max_points: context.evaluation.config.max_native_query_points,
        },
        context.evaluation.work,
        |a, owner, x| integrand(a, &lanes.lanes, owner, x),
        |a, owner, x| panel_fourth_bound(a, &lanes.lanes, owner, x),
    )?;
    checked(c, "point prefix potential certificate", |a| {
        let contributions =
            assemble_lane_contributions(a, &lanes.lanes, &lanes.pointed, &numerical)?;
        let zero = Tensor::zeros(contributions.lower().shape(), DType::F64, c.device())?;
        let lower = c.where_cond(&zero_gap, &zero, contributions.lower())?;
        let upper = c.where_cond(&zero_gap, &zero, contributions.upper())?;
        let shape = [lanes.lanes.face_count, lanes.lanes.band_count];
        let lower = lower.reshape(&shape)?;
        let upper = upper.reshape(&shape)?;
        let mut total = zero_value(a, lanes.lanes.face_count)?;
        for band in 0..lanes.lanes.band_count {
            let term = a.bounds(
                &lower.narrow(1, band, 1)?.squeeze(Some(1)),
                &upper.narrow(1, band, 1)?.squeeze(Some(1)),
            )?;
            let retained = a.expansion_from_interval(&term)?;
            total.retained = a.expansion_add(&total.retained, &retained)?;
            total.physical = a.add(&total.physical, &term)?;
        }
        let zero = Tensor::zeros(total.physical.lower().shape(), DType::F64, c.device())?;
        total.physical = a.bounds(
            &c.maximum(total.physical.lower(), &zero)?,
            total.physical.upper(),
        )?;
        Ok(total)
    })
}

/// Unavailable raw area certificates cannot authorize a root, even when their centre is nonnegative.
pub(super) fn admitted_analytic<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &crate::channel::section::GatheredBands<R>,
    raw: &CertifiedExpansion<R>,
    depth: &Interval<R>,
    gravity: f64,
) -> HydraulicResult<analytic_bottom::AnalyticSource<R>>
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
    let c = a.client();
    let zero = Tensor::zeros(raw.high().shape(), DType::F64, c.device())?;
    let bounds = a.expansion_interval(raw)?;
    let admitted = c.mul(&c.ge(bounds.lower(), &zero)?, &c.ge(depth.lower(), &zero)?)?;
    let safe_raw = a.expansion_bounds(
        &c.where_cond(&admitted, raw.high(), &zero)?,
        &c.where_cond(&admitted, raw.low(), &zero)?,
        &c.where_cond(&admitted, raw.abs_error(), &zero)?,
    )?;
    let safe_depth = a.bounds(
        &c.where_cond(&admitted, depth.lower(), &zero)?,
        &c.where_cond(&admitted, depth.upper(), &zero)?,
    )?;
    let mut analytic = analytic_bottom::source(a, bands, &safe_raw, &safe_depth, gravity)?;
    analytic.eligible = c.mul(&analytic.eligible, &admitted)?;
    analytic.rectangular = c.mul(&analytic.rectangular, &admitted)?;
    analytic.pointed = c.mul(&analytic.pointed, &admitted)?;
    Ok(analytic)
}
