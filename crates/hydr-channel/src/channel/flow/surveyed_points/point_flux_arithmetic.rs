use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Area, discharge, and chosen pressure are exact numerical points. Evaluation uncertainty remains outside this ledger.
pub(super) fn momentum<R>(
    a: &mut Arithmetic<'_, R>,
    area: &CertifiedExpansion<R>,
    discharge: &CertifiedExpansion<R>,
    pressure: &CertifiedExpansion<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let (velocity, advective) = advection(a, area, discharge)?;
    let momentum = a.expansion_add(&advective, pressure)?;
    Ok((velocity, momentum))
}

/// Exact numerical operands retain every quotient and product arithmetic radius.
pub(super) fn advection<R>(
    a: &mut Arithmetic<'_, R>,
    area: &CertifiedExpansion<R>,
    discharge: &CertifiedExpansion<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let velocity = a.expansion_div_positive(discharge, area)?;
    let advective = a.expansion_mul(discharge, &velocity)?;
    Ok((velocity, advective))
}

pub(super) fn intersect_advection<R>(
    a: &mut Arithmetic<'_, R>,
    ordinary: &Interval<R>,
    retained: &CertifiedExpansion<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let retained = a.expansion_interval(retained)?;
    let (intersection, valid) = super::wet_star::intersection(a, ordinary, &retained)?;
    let zero = numr::tensor::Tensor::zeros(valid.shape(), a.dtype(), a.client().device())?;
    if a.client().nonzero(&a.client().eq(&valid, &zero)?)?.numel() != 0 {
        return Err(super::point_root_session::invalid(
            "ordinary and retained advection certificates are disjoint. Discard the inconsistent point",
        ));
    }
    Ok(intersection)
}

#[cfg(test)]
mod tests {
    use super::super::point_flux::fixture::diagnostics;
    use super::super::{
        chart_fixture, source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_point_fluxes_preserve_film_and_source_precision() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        precision::run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_fluxes_preserve_film_and_source_precision() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        precision::run::<CudaRuntime>(&client)
    }

    mod precision {
        use super::super::super::{
            point_candidates::fixture::{fixtures, flux_fixtures},
            point_flux_types::{PointFluxKind, PointFluxStatus},
            point_root_session::PointRootSession,
        };
        use super::{chart_fixture, diagnostics, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::error::checked;
        use frozen::Result;
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let cases = fixtures::executions();
            for id in ["retained_vacuum_inside_film", "tiny_rectangle_compression"] {
                let case = cases
                    .iter()
                    .find(|e| e.id == id)
                    .ok_or("Missing frozen precision execution")?;
                let reference = flux_fixtures::REFERENCES
                    .iter()
                    .find(|r| r.id == id)
                    .ok_or("Missing frozen full flux precision truth")?;
                let config = fixtures::CONFIGS[case.config].build();
                chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                    let source_a = frozen::components(&pair.velocity.input().left_area)?;
                    let source_q = frozen::components(&pair.velocity.input().left_discharge)?;
                    let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    let initial = hydraulic(session.select_point_fluxes(&owners))?;
                    if id == "retained_vacuum_inside_film" {
                        let star = case.star.ok_or("Film lacks independent star area")?;
                        let atol = f64::from_bits(star.area[0]) / cap * 1e-12;
                        assert!(atol > 0. && atol < config.root.atol);
                        hydraulic(
                            session.refine_wet_depth_in_place(CertifiedRootTolerance {
                                atol,
                                rtol: 0.,
                            }),
                        )?;
                        hydraulic(
                            session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }),
                        )?;
                    }
                    let before = work.snapshot();
                    let result = hydraulic(session.select_point_fluxes(&owners))?;
                    diagnostics::check(&session, &result)?;
                    assert_eq!(session.callback_points(), before.residual_points);
                    assert_eq!(work.snapshot().root_solves, before.root_solves);
                    assert_eq!(result.point_limit, initial.point_limit);
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointFluxStatus::Accepted as i64]
                    );
                    let point = result
                        .selected
                        .first()
                        .ok_or("Precision case lacks selected point")?;
                    let (water, momentum) = hydraulic(checked::<R, _>(
                        client,
                        "selected numerical full flux precision",
                        |a| {
                            Ok((
                                a.expansion_interval(&point.discharge)?,
                                a.expansion_interval(&point.momentum)?,
                            ))
                        },
                    ))?;
                    for (actual, expected) in [
                        (
                            frozen::endpoints(&water)?[0],
                            reference.water.ok_or("Missing water truth")?,
                        ),
                        (
                            frozen::endpoints(&momentum)?[0],
                            reference.momentum.ok_or("Missing momentum truth")?,
                        ),
                    ] {
                        let expected = expected.map(f64::from_bits);
                        let scale = expected[0].abs().max(expected[1].abs());
                        assert!(scale > 0.);
                        let error = (actual[0] - expected[1])
                            .abs()
                            .max((actual[1] - expected[0]).abs());
                        eprintln!(
                            "POINT_FLUX_PRECISION\t{id}\tselected={actual:?}\ttruth={expected:?}\trelative_error={:e}",
                            error / scale
                        );
                        assert!(
                            error / scale < 1e-10,
                            "{id} loses original selected full flux precision"
                        );
                    }
                    if id == "tiny_rectangle_compression" {
                        assert!(matches!(point.kind, PointFluxKind::Exterior(_)));
                        assert_eq!(frozen::components(&point.area)?, source_a);
                        assert_eq!(frozen::components(&point.discharge)?, source_q);
                    } else {
                        assert!(matches!(point.kind, PointFluxKind::Fan(_)));
                        let proposal =
                            frozen::components(&point.discharge_proposal)?[0].map(f64::from_bits);
                        assert_eq!([proposal[0], proposal[1]], [0., 0.]);
                        assert_eq!(
                            point.discharge_proposal_contained.try_to_vec::<f64>()?,
                            [0.]
                        );
                        let physical = frozen::endpoints(&point.discharge_physical)?[0];
                        assert!(physical[1] < 0.);
                        let chosen = frozen::components(&point.discharge)?[0];
                        let change = frozen::components(&point.discharge_selection_change)?[0];
                        assert_eq!(&chosen[..2], &change[..2]);
                        let raw =
                            frozen::components(&point.discharge_evaluation)?[0].map(f64::from_bits);
                        assert!(raw[2] > 0.);
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
    }
}
