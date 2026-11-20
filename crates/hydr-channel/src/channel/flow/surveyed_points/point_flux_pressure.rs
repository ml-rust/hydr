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
    // The secant and the direct pressure read one recovery of `query`.
    let direct = sources
        .geometry
        .recover_query(a, &selected.bands, query)?
        .state;
    let increment = pressure_endpoint(
        a,
        sources.geometry,
        &selected.bands,
        query,
        source,
        &selected.boundaries,
        gravity,
        Some(&direct),
    )?;
    let evaluation = a.add(&anchor, &increment.physical_increment)?;
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

#[cfg(test)]
mod tests {
    use super::super::{
        chart_fixture, point_flux_pressure,
        source_fixture_support::{self as frozen, Result, hydraulic},
        wet_depth::fixture as support,
    };
    use super::*;
    use crate::channel::flow::surveyed::error::checked;

    fn tiny_complete_pressure<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            1.,
            4.,
            &[[1., 0., 1., 0.]],
            &config,
            |pair, work| {
                let area = 2.0_f64.powi(-164);
                let truth = 2.0_f64.powi(-329);
                let query = chart_fixture::chart(
                    pair,
                    chart_fixture::Anchor::Zero,
                    [1.0_f64.to_bits(), 0],
                    &[[area, area]],
                )?;
                let parent = support::rows::<R>(client, &[0])?;
                let (pressure, numerical) = hydraulic(checked::<R, _>(
                    client,
                    "independent tiny rectangle pressure control",
                    |a| {
                        let pressure = point_flux_pressure::selected(a, pair, &parent, &query, 0)?;
                        let numerical = a.expansion_interval(&pressure.chosen)?;
                        Ok((pressure, numerical))
                    },
                ))?;
                let physical = frozen::endpoints(&pressure.evaluation)?[0];
                let numerical = frozen::endpoints(&numerical)?[0];
                eprintln!(
                    "POINT_FLUX_TINY_PRESSURE\ttruth={truth:e}\tphysical={physical:?}\tnumerical={numerical:?}\tanchor={:?}\tincrement={:?}\tselected_increment={:?}\tsecant={:?}",
                    frozen::components(&pressure.source_anchor)?,
                    frozen::components(&pressure.increment.retained_increment)?,
                    frozen::components(&pressure.selected_increment)?,
                    frozen::endpoints(&pressure.increment.secant)?
                );
                assert!(physical[0] <= truth && truth <= physical[1]);
                // The trace takes the model integral at its own area, not the source anchor and secant.
                assert_eq!(pressure.proposal_contained.try_to_vec::<f64>()?, [1.]);
                assert!(physical[0] <= numerical[0] && numerical[1] <= physical[1]);
                let error = (numerical[0] - truth)
                    .abs()
                    .max((numerical[1] - truth).abs());
                assert!(
                    error / truth < 1e-10,
                    "Selected pressure loses exact rectangle P=A^2/2: {numerical:?}, truth={truth:e}"
                );
                assert_eq!(work.snapshot().root_solves, 0);
                assert_eq!(work.snapshot().residual_points, 0);
                let identity_query =
                    chart_fixture::chart(pair, chart_fixture::Anchor::Source, [0, 0], &[[0., 0.]])?;
                let delta = 2.0_f64.powi(-80);
                let weak_query = chart_fixture::chart(
                    pair,
                    chart_fixture::Anchor::Source,
                    [delta.to_bits(), 0],
                    &[[1., 1.]],
                )?;
                let (identity, weak) = hydraulic(checked::<R, _>(
                    client,
                    "independent source and weak rectangle pressure controls",
                    |a| {
                        Ok((
                            point_flux_pressure::selected(a, pair, &parent, &identity_query, 0)?,
                            point_flux_pressure::selected(a, pair, &parent, &weak_query, 0)?,
                        ))
                    },
                ))?;
                assert_eq!(
                    frozen::components(&identity.increment.retained_increment)?,
                    [[0, 0, 0]]
                );
                let identity_bounds = frozen::endpoints(&identity.evaluation)?[0];
                assert!(identity_bounds[0] <= 0.5 && 0.5 <= identity_bounds[1]);
                // P(1+delta)-P(1)=delta+delta^2/2. These adjacent F64 values enclose that exact dyadic value.
                let expected = [delta, f64::from_bits(delta.to_bits() + 1)];
                let increment = frozen::endpoints(&weak.increment.physical_increment)?[0];
                assert!(increment[0] <= expected[0] && expected[1] <= increment[1]);
                assert!((increment[1] - increment[0]) / delta < 1e-10);
                let retained =
                    frozen::components(&weak.increment.retained_increment)?[0].map(f64::from_bits);
                assert!(retained[0] > 0.);
                assert!((retained[0] + retained[1] - delta).abs() / delta < 1e-10);
                Ok(())
            },
        )
    }

    #[test]
    fn configured_cpu_point_flux_pressure_preserves_tiny_complete_pressure() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        tiny_complete_pressure::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_flux_pressure_preserves_tiny_complete_pressure() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        tiny_complete_pressure::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
