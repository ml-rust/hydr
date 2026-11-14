use super::{
    paired_sources::PairedSources,
    point_flux_projection::{center, choose},
    point_flux_types::PointPressure,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::scale,
        error::{Arithmetic, HydraulicResult},
    },
    section::{PhysicalEndpoint, PressureEndpoint, SupportedAreaQuery, pressure_endpoint},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

fn evaluation<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    side: usize,
) -> HydraulicResult<(CertifiedExpansion<R>, PressureEndpoint<R>, Interval<R>)>
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
    let selected = sources.select_rows(parent, query)?;
    let gravity = sources.evaluation.config.gravity_m_per_s2;
    let anchor = scale(a, selected.sources[side].hydrostatic_integral_m3(), gravity)?;
    let source_anchor = a.expansion_from_interval(&anchor)?;
    let increment = pressure_endpoint(
        a,
        sources.geometry,
        &selected.bands,
        query,
        &selected.sources[side],
        &selected.boundaries,
        gravity,
    )?;
    let evaluation = a.add(&anchor, &increment.physical_increment)?;
    let direct = sources
        .geometry
        .recover_query(a, &selected.bands, query)?
        .state;
    let direct_pressure = scale(a, &direct.integral, gravity)?;
    let (evaluation, valid) = super::wet_star::intersection(a, &evaluation, &direct_pressure)?;
    if a.client()
        .nonzero(&a.client().eq(
            &valid,
            &Tensor::zeros(valid.shape(), a.dtype(), a.client().device())?,
        )?)?
        .numel()
        != 0
    {
        return Err(super::point_root_session::invalid(
            "selected pressure anchor and direct geometry are disjoint",
        ));
    }
    let evaluation = super::point_flux_quantum::pressure(
        a,
        query.area.upper(),
        direct.depth.upper(),
        &evaluation,
        gravity,
    )?;
    Ok((source_anchor, increment, evaluation))
}

pub(super) fn selected<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    side: usize,
) -> HydraulicResult<PointPressure<R>>
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
    let (source_anchor, increment, evaluation) = self::evaluation(a, sources, parent, query, side)?;
    let selected_anchor = center(a, &source_anchor)?;
    let selected_increment = center(a, &increment.retained_increment)?;
    let proposal_arithmetic = a.expansion_add(&selected_anchor, &selected_increment)?;
    let projection = choose(a, &proposal_arithmetic, &evaluation, None)?;
    let proposal = projection.proposal;
    let proposal_contained = projection.contained;
    let chosen = projection.chosen;
    let selection_change = projection.change;
    let numerical = a.expansion_interval(&chosen)?;
    let evaluation_discrepancy = a.sub(&evaluation, &numerical)?;
    Ok(PointPressure {
        source_anchor,
        increment,
        evaluation,
        selected_anchor,
        selected_increment,
        proposal_arithmetic,
        proposal,
        proposal_contained,
        chosen,
        selection_change,
        evaluation_discrepancy,
    })
}

pub(super) fn physical<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    side: usize,
) -> HydraulicResult<Interval<R>>
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
    let selected = sources.select_rows(parent, query)?;
    let lower_query = query.endpoint(a, &selected.bands, PhysicalEndpoint::Lower)?;
    let upper_query = query.endpoint(a, &selected.bands, PhysicalEndpoint::Upper)?;
    let (_, _, lower) = evaluation(a, sources, parent, &lower_query, side)?;
    let (_, _, upper) = evaluation(a, sources, parent, &upper_query, side)?;
    Ok(a.bounds(lower.lower(), upper.upper())?)
}
