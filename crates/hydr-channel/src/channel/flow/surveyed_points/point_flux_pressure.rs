use super::{
    paired_sources::PairedSources,
    point_flux_projection::{center, choose},
    point_flux_types::PointPressure,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{choose_expansion, scale},
        error::{Arithmetic, HydraulicResult},
    },
    section::{PhysicalEndpoint, PressureEndpoint, SupportedAreaQuery, pressure_endpoint},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Gravity times the two-part hydrostatic selection value of the base source.
///
/// Bit-equal source areas give bit-equal anchors. The certified bounds never use it.
fn selection_anchor<R>(
    a: &mut Arithmetic<'_, R>,
    point: &CertifiedExpansion<R>,
    gravity: f64,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let gravity = Tensor::full_scalar(
        point.high().shape(),
        a.dtype(),
        gravity,
        a.client().device(),
    )?;
    let gravity = a.expansion_point(&gravity)?;
    Ok(a.expansion_mul(point, &gravity)?)
}

/// Gravity times the two-part model integral at the exact area point of `query`.
fn trace_point<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    let stage = sources.select_rows(parent, query)?;
    let area = query.retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
    let area = center(a, &area)?;
    let point = sources
        .geometry
        .hydrostatic_point_query(a, &stage.bands, query, &area)?;
    selection_anchor(a, &point, sources.evaluation.config.gravity_m_per_s2)
}

/// Certified pressure at `query` from the side's source, with the two-part selection anchor.
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
    let source = &selected.sources[side];
    let anchor = scale(a, source.hydrostatic_integral_m3(), gravity)?;
    let source_anchor = selection_anchor(a, source.hydrostatic_point_m3(), gravity)?;
    let increment = pressure_endpoint(
        a,
        sources.geometry,
        &selected.bands,
        query,
        source,
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

/// Selected pressure of one trace from the base source on `side`.
///
/// A trace at the source area takes the two-part source anchor. Its secant increment is zero.
/// Any other trace takes gravity times the two-part model integral at its own area point.
/// A secant increment from F64 bounds would carry an error on the scale of the whole jump.
/// Bit-equal trace areas give bit-equal proposals.
/// The certified evaluation keeps the certified source integral and secant.
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
    let anchored = a.expansion_add(&selected_anchor, &selected_increment)?;
    let direct = trace_point(a, sources, parent, query)?;
    let c = a.client();
    let zero = Tensor::zeros(anchored.high().shape(), a.dtype(), c.device())?;
    let delta = &increment.delta_area;
    let at_source = c.mul(&c.eq(delta.high(), &zero)?, &c.eq(delta.low(), &zero)?)?;
    let proposal_arithmetic = choose_expansion(a, &at_source, &anchored, &direct)?;
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

/// Hull of the certified pressure over the lower and upper endpoint queries of `query`.
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
