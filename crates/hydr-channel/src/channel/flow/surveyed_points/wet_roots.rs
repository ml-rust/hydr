#[cfg(test)]
pub(super) use super::wet_root_types::WetRootDomains;
use super::{
    chart_admission, paired_sources::PairedSources, wet_root_direct, wet_root_search,
    wet_root_state::Locator,
};
use super::{root_invocation::RootInvocation, wet_root_types::WetRootResults};
#[cfg(test)]
use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedUpperBracketAlgorithms};

/// Locate complete wet/wet root domains without selecting centers or accepting hydraulic fluxes.
#[cfg(test)]
pub(super) fn locate_wet_roots<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    max_points: usize,
) -> HydraulicResult<WetRootDomains<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = sources.evaluation.config;
    if max_points > config.root.max_points || (parent_rows.numel() != 0 && max_points == 0) {
        return Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: format!(
                "Wet root max_points={max_points}, rows={}, configured ceiling={}. Supply a positive nonempty invocation budget within the ceiling",
                parent_rows.numel(),
                config.root.max_points,
            ),
        });
    }
    let mut invocation = RootInvocation::new(sources.root_witness()?, max_points)?;
    let result = locate_on_invocation(sources, parent_rows, &mut invocation)?;
    Ok(WetRootDomains {
        invocation,
        invocation_rows: result.invocation_rows,
        status: result.status,
        domains: result.domains,
        evidence: result.evidence,
    })
}

pub(super) fn locate_on_invocation<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    invocation: &mut RootInvocation,
) -> HydraulicResult<WetRootResults<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    sources.check_root_witness(invocation.source_witness())?;
    let c = sources.geometry.client();
    let selected = checked::<R, _>(c, "wet root invocation admission", |a| {
        let selected = sources.select_sources(a, parent_rows)?;
        let zero = Tensor::zeros(&[], a.dtype(), c.device())?;
        if c.nonzero(&c.le(&selected.velocity.both_wet, &zero)?)?
            .numel()
            != 0
        {
            return Err(chart_admission::unresolved(
                "Wet root locator requires two wet sources. Classify dry sources before locating domains",
            ));
        }
        Ok(selected)
    })?;
    let rows = checked::<R, _>(c, "wet root invocation ownership", |a| {
        let owned = Tensor::empty(selected.rows.shape(), DType::I64, a.client().device())?;
        a.client().copy_into(&owned, &selected.rows)?;
        Ok(owned)
    })?;
    let mut locator = Locator::new(c, rows, invocation)?;
    wet_root_direct::singleton(sources, &mut locator, &selected, false)?;
    wet_root_direct::dry_limit(sources, &mut locator)?;
    wet_root_direct::singleton(sources, &mut locator, &selected, true)?;
    wet_root_search::local(sources, &mut locator, 0)?;
    wet_root_search::local(sources, &mut locator, 1)?;
    wet_root_search::covered(sources, &mut locator)?;
    Ok(locator.output)
}

#[cfg(test)]
#[path = "wet_root_fixture_cases.rs"]
mod cases;
#[cfg(test)]
#[path = "wet_root_fixture_certificates.rs"]
mod certificates;
#[cfg(test)]
#[path = "wet_root_fixture_support.rs"]
mod support;

#[cfg(test)]
mod tests {
    use super::locate_wet_roots;
    use super::{cases, certificates, support};
    use crate::channel::flow::{
        surveyed::error::{HydraulicError, PrismaticRiemannRejection},
        surveyed_points::{
            chart_fixture,
            corpus_coverage::Coverage,
            source_fixture_support::{self as frozen, Client, Result, hydraulic},
            wet_root_types::WetRootStatus,
        },
    };
    use numr::{
        dtype::DType,
        runtime::{Runtime, RuntimeClient},
        tensor::Tensor,
    };
    use solvr::{
        integrate::CertifiedSimpsonAlgorithms,
        optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    };

    fn controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        admission::run::<R>(client)?;
        selection::run::<R>(client)?;
        oracles::run::<R>(client)
    }

    fn configured_cpu(run: impl FnOnce(&numr::runtime::cpu::CpuClient) -> Result) -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        run(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    fn corpus<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        assert_eq!(frozen::cases().len(), 70);
        let wet = frozen::cases()
            .iter()
            .filter(|case| !case.dry[0] && !case.dry[1]);
        let replayed = coverage.select(wet, |case| {
            (
                cases::declared(case.id).1,
                cases::star(case.id).is_some(),
                case.uniform,
            )
        });
        let selected = replayed.len();
        let mut executed = 0;
        for case in replayed {
            let (config, expected) = cases::declared(case.id);
            chart_fixture::with_pair::<R, _>(client, &[case.id], &config, |pair, work| {
                let source_before = support::source_bits(pair)?;
                let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let result = locate_wet_roots(pair, &parents, config.root.max_points);
                assert_eq!(support::source_bits(pair)?, source_before, "{}", case.id);
                if expected == cases::Expected::RejectedWork {
                    assert!(
                        matches!(
                            result,
                            Err(HydraulicError::Rejected {
                                reason: PrismaticRiemannRejection::WorkLimit,
                                ..
                            })
                        ),
                        "{}: expected declared zero-work rejection",
                        case.id
                    );
                    assert_eq!(work.snapshot().residual_points, 0);
                    return Ok(());
                }
                let result = hydraulic(result)?;
                let status = result.status.item::<i64>()?;
                let exact = match expected {
                    cases::Expected::Located => Some(WetRootStatus::Located),
                    cases::Expected::Vacuum => Some(WetRootStatus::Vacuum),
                    cases::Expected::TouchingDry => Some(WetRootStatus::TouchingDry),
                    cases::Expected::CapExceeded => Some(WetRootStatus::CapExceeded),
                    cases::Expected::LegacyThreshold | cases::Expected::RejectedWork => None,
                };
                if let Some(expected) = exact {
                    assert_eq!(status, expected as i64, "{}", case.id);
                } else {
                    assert!(
                        status == WetRootStatus::Located as i64
                            || status == WetRootStatus::Unresolved as i64
                            || status == WetRootStatus::WorkLimit as i64,
                        "{}: {status}",
                        case.id
                    );
                }
                certificates::check(pair, &result)?;
                assert_eq!(
                    result.callback_points(),
                    work.snapshot().residual_points,
                    "{}",
                    case.id
                );
                assert!(
                    result.callback_points() <= config.root.max_points,
                    "{}",
                    case.id
                );
                if let Some(reference) = cases::star(case.id) {
                    assert_eq!(status, WetRootStatus::Located as i64, "{}", case.id);
                    for domain in &result.domains {
                        let bounds =
                            support::physical_bounds(pair, &domain.parent_rows, &domain.query)?;
                        for actual in bounds {
                            support::star_contains(actual, reference, case.id);
                        }
                    }
                }
                if case.id == "uniform_cap_retained_deficit_dyadic_velocity" {
                    for domain in &result.domains {
                        support::cap_deficit_contains(pair, &domain.parent_rows, &domain.query)?;
                    }
                }
                Ok(())
            })?;
            executed += 1;
        }
        assert_eq!(executed, selected);
        if coverage.is_full() {
            assert_eq!(executed, 61);
        }
        Ok(())
    }

    #[test]
    fn configured_cpu_wet_root_domains_preserve_frozen_outcomes() -> Result {
        configured_cpu(|client| corpus::<numr::runtime::cpu::CpuRuntime>(client, Coverage::Full))
    }

    #[test]
    fn configured_cpu_wet_root_film_and_cap_chains_preserve_retained_roots() -> Result {
        configured_cpu(|client| {
            chains::run::<numr::runtime::cpu::CpuRuntime>(client, chains::FILM_AND_CAP)
        })
    }

    #[test]
    fn configured_cpu_wet_root_jump_and_knot_chains_preserve_retained_roots() -> Result {
        configured_cpu(|client| {
            chains::run::<numr::runtime::cpu::CpuRuntime>(client, chains::JUMPS_AND_KNOTS)
        })
    }

    #[test]
    fn configured_cpu_wet_root_admission_selection_and_oracles() -> Result {
        configured_cpu(controls::<numr::runtime::cpu::CpuRuntime>)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_wet_root_domains_preserve_frozen_outcomes_and_retained_roots() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        corpus::<CudaRuntime>(&client, Coverage::Sample)?;
        chains::run::<CudaRuntime>(&client, &chains::FILM_AND_CAP[..1])?;
        chains::run_pointed::<CudaRuntime>(&client)?;
        controls::<CudaRuntime>(&client)
    }

    #[test]
    fn configured_cpu_pointed_film_locator_preserves_independent_root() -> Result {
        configured_cpu(chains::run_pointed::<numr::runtime::cpu::CpuRuntime>)
    }

    mod admission {
        use crate::channel::flow::{
            surveyed::error::{HydraulicError, PrismaticRiemannRejection},
            surveyed_points::{
                chart_fixture::{self, cache},
                source_fixture_support::{Client, Result, hydraulic},
                wet_root_types::WetRootStatus,
                wet_roots::locate_wet_roots,
            },
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &config,
                |pair, work| {
                    let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let before = (cache(pair)?, work.snapshot());
                    for points in [0, config.root.max_points + 1] {
                        assert!(matches!(
                            locate_wet_roots(pair, &parents, points),
                            Err(HydraulicError::Rejected {
                                reason: PrismaticRiemannRejection::WorkLimit,
                                ..
                            })
                        ));
                        assert_eq!((cache(pair)?, work.snapshot()), before);
                    }
                    for invalid in [
                        Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                        Tensor::from_slice(&[1_i64], &[1], client.device())?,
                        Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
                        Tensor::from_slice(&[0_i64], &[], client.device())?,
                    ] {
                        assert!(locate_wet_roots(pair, &invalid, config.root.max_points).is_err());
                        assert_eq!((cache(pair)?, work.snapshot()), before);
                    }
                    let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
                    let result = hydraulic(locate_wet_roots(pair, &empty, 0))?;
                    assert_eq!(result.status.shape(), [0]);
                    assert_eq!(result.invocation_rows.shape(), [0]);
                    assert!(result.domains.is_empty() && result.evidence.is_empty());
                    assert_eq!(result.callback_points(), 0);
                    assert_eq!((cache(pair)?, work.snapshot()), before);
                    let result = hydraulic(locate_wet_roots(pair, &parents, 1))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [WetRootStatus::Located as i64]
                    );
                    assert_eq!(result.callback_points(), 1);
                    assert_eq!(work.snapshot().residual_points, 1);
                    super::certificates::check(pair, &result)
                },
            )?;
            let mut zero = config.clone();
            zero.max_native_query_points = 0;
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &zero,
                |pair, work| {
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let before = cache(pair)?;
                    assert!(matches!(
                        locate_wet_roots(pair, &rows, zero.root.max_points),
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        })
                    ));
                    assert_eq!(work.snapshot().residual_points, 0);
                    assert_eq!(cache(pair)?, before);
                    Ok(())
                },
            )?;
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_left_dry_sonic"],
                &config,
                |pair, work| {
                    let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                    let before = (cache(pair)?, work.snapshot());
                    assert!(matches!(
                        locate_wet_roots(pair, &rows, config.root.max_points),
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::GeometryUnresolved,
                            ..
                        })
                    ));
                    assert_eq!((cache(pair)?, work.snapshot()), before);
                    Ok(())
                },
            )
        }
    }

    mod chains {
        use super::{cases, certificates, support};
        use crate::channel::flow::{
            surveyed::PrismaticRiemannConfig,
            surveyed_points::{
                chart_fixture,
                chart_roots::solve_chart,
                paired_sources::PairedSources,
                source_fixture_support::{Client, Result, components, endpoints, hydraulic},
                wet_root_types::WetRootStatus,
                wet_roots::locate_wet_roots,
            },
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{
                CertifiedBisectionAlgorithms, CertifiedRootStatus, CertifiedUpperBracketAlgorithms,
            },
        };

        pub(super) const FILM_AND_CAP: &[&str] = &[
            "retained_vacuum_inside_film",
            "uniform_cap_retained_deficit_dyadic_velocity",
            "scalar_rectangle_stationary_shock",
        ];

        pub(super) const JUMPS_AND_KNOTS: &[&str] = &[
            "tiny_rectangle_compression",
            "scalar_rectangle_weak_depth_jump",
            "retained_knot_2_crossing",
            "retained_knot_7_crossing",
        ];

        pub(super) fn run<R>(client: &R::Client, ids: &[&str]) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            for &id in ids {
                let (config, _) = cases::declared(id);
                chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                    chain(pair, &config, id, cases::star(id))?;
                    assert!(work.snapshot().residual_points <= config.root.max_points);
                    Ok(())
                })?;
            }
            Ok(())
        }

        pub(super) fn run_pointed<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let mut config = chart_fixture::config();
            config.gravity_m_per_s2 = 2.0;
            support::pointed_film::<R, _>(client, &config, |pair, _| {
                // A=h², Phi(A)=4*A^(1/4), F(A)=8*A^(1/4)-2^-80.
                chain(
                    pair,
                    &config,
                    "analytic_pointed_film",
                    Some([0x2b30000000000000; 2]),
                )
            })
        }

        fn chain<R>(
            pair: &mut PairedSources<'_, R>,
            config: &PrismaticRiemannConfig,
            id: &str,
            star: Option<[u64; 2]>,
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let client = pair.geometry.client();
            let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            assert_eq!(
                located.status.try_to_vec::<i64>()?,
                [WetRootStatus::Located as i64],
                "{id}: callbacks={}, work={:?}, evidence={:?}",
                located.callback_points(),
                pair.evaluation.work.snapshot(),
                located
                    .evidence
                    .iter()
                    .map(|evidence| Ok((
                        evidence.kind,
                        endpoints(&evidence.value.physical)?,
                        components(&evidence.value.retained)?,
                    )))
                    .collect::<Result<Vec<_>>>()?
            );
            certificates::check(pair, &located)?;
            assert!(!located.domains.is_empty(), "{id}");
            let mut remaining = config
                .root
                .max_points
                .checked_sub(located.callback_points())
                .ok_or("Locator exceeds cumulative root-point budget")?;
            let automatic_film =
                matches!(id, "retained_vacuum_inside_film" | "analytic_pointed_film");
            for domain in &located.domains {
                if automatic_film {
                    let physical =
                        support::physical_bounds(pair, &domain.parent_rows, &domain.query)?;
                    assert!(
                        physical
                            .iter()
                            .all(|bounds| bounds[1] > 0.0 && bounds[1] < 1e-40),
                        "{id}: automatic film domain {physical:?}"
                    );
                }
                let before = pair.evaluation.work.snapshot().residual_points;
                let root = hydraulic(solve_chart(
                    pair,
                    &domain.parent_rows,
                    &domain.query,
                    remaining,
                ))?;
                let after = pair.evaluation.work.snapshot().residual_points;
                assert_eq!(after - before, root.result.evaluated_points(), "{id}");
                remaining = remaining
                    .checked_sub(root.result.evaluated_points())
                    .ok_or("Chart refinement exceeds cumulative root-point budget")?;
                assert_eq!(root.root_rows.try_to_vec::<i64>()?, [0], "{id}");
                assert!(
                    root.result
                        .status()
                        .try_to_vec::<i64>()?
                        .iter()
                        .all(|&code| code == CertifiedRootStatus::Converged as i64
                            || code == CertifiedRootStatus::AccuracyFloor as i64),
                    "{id}: root status {:?}",
                    root.result.status().try_to_vec::<i64>()?
                );
                let physical = support::physical_bounds(pair, &root.root_rows, &root.query)?;
                if let Some(reference) = star {
                    for &bounds in &physical {
                        support::star_contains(bounds, reference, id);
                    }
                }
                if id == "retained_vacuum_inside_film" {
                    let star = f64::from_bits(0x35b0000000000000);
                    let gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
                    for [lower, upper] in physical {
                        assert!(lower <= star && star <= upper);
                        assert!(
                            (upper - lower) / star <= gate,
                            "film [{lower:e}, {upper:e}]"
                        );
                        assert!((star - lower).max(upper - star) / star <= gate);
                    }
                }
                if id == "uniform_cap_retained_deficit_dyadic_velocity" {
                    support::cap_deficit_contains(pair, &root.root_rows, &root.query)?;
                }
            }
            Ok(())
        }
    }

    mod oracles {
        use super::{certificates, support};
        use crate::channel::flow::surveyed_points::{
            chart_fixture,
            source_fixture_support::{Client, Result, components, hydraulic},
            wet_root_types::{WetRootEvidenceKind, WetRootStatus},
            wet_roots::locate_wet_roots,
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            support::uncertain_dry_limit::<R, _>(client, &config, |pair, work| {
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let result = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [WetRootStatus::Unresolved as i64]
                );
                assert!(result.domains.is_empty());
                certificates::check(pair, &result)?;
                let evidence = result
                    .evidence
                    .iter()
                    .find(|evidence| evidence.kind == WetRootEvidenceKind::DryLimit)
                    .ok_or("Unresolved dry limit lacks its original residual evidence")?;
                // Exact du=10/3 and PhiL+PhiR=10/3 establish mathematical touching.
                // Independent thirds certificates cannot establish the required structural zero.
                assert!(evidence.value.physical.lower().item::<f64>()? <= 0.0);
                assert!(evidence.value.physical.upper().item::<f64>()? >= 0.0);
                assert!(
                    components(&evidence.value.retained)?
                        .iter()
                        .all(|point| f64::from_bits(point[2]) > 0.0)
                );
                assert!(result.callback_points() > 0);
                assert!(result.callback_points() <= config.root.max_points);
                assert_eq!(work.snapshot().residual_points, result.callback_points());
                Ok(())
            })
        }
    }

    mod selection {
        use crate::channel::flow::surveyed_points::{
            chart_fixture::{self, cache},
            source_fixture_support::{Client, Result, hydraulic},
            wet_root_types::WetRootStatus,
            wet_roots::locate_wet_roots,
        };
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::{
            integrate::CertifiedSimpsonAlgorithms,
            optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            mapping::<R>(client)?;
            subsets::<R>(client)?;
            budgets::<R>(client)?;
            multi_chart_budget::<R>(client)
        }

        fn mapping<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            let ids = [
                "retained_vacuum_inside_film",
                "scalar_rectangle_vacuum",
                "scalar_rectangle_vacuum_threshold",
                "scalar_three_bands_cap_compression",
                "scalar_rectangle_uniform_rest",
                "scalar_rectangle_weak_depth_jump",
            ];
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let parent_values = [4_i64, 1, 4, 0, 2, 3];
                let storage = Tensor::from_slice(
                    &[99_i64, 4, 99, 1, 99, 4, 99, 0, 99, 2, 99, 3],
                    &[6, 2],
                    client.device(),
                )?;
                let parents = storage.narrow(1, 1, 1)?.squeeze(Some(1));
                let result = hydraulic(locate_wet_roots(pair, &parents, config.root.max_points))?;
                assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [
                        WetRootStatus::Located as i64,
                        WetRootStatus::Vacuum as i64,
                        WetRootStatus::Located as i64,
                        WetRootStatus::Located as i64,
                        WetRootStatus::TouchingDry as i64,
                        WetRootStatus::CapExceeded as i64
                    ]
                );
                super::certificates::check(pair, &result)?;
                assert_eq!(work.snapshot().residual_points, result.callback_points());
                let physical = result
                    .domains
                    .iter()
                    .map(|domain| {
                        super::support::physical_bounds(pair, &domain.parent_rows, &domain.query)
                    })
                    .collect::<Result<Vec<_>>>()?;
                client.copy_into(&parents, &Tensor::zeros(&[6], DType::I64, client.device())?)?;
                assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
                super::certificates::check(pair, &result)?;
                for (domain, before) in result.domains.iter().zip(physical) {
                    assert_eq!(
                        super::support::physical_bounds(pair, &domain.parent_rows, &domain.query)?,
                        before
                    );
                }
                assert!(cache(pair)?.iter().all(|entry| entry.ready[5] == 0.0));
                Ok(())
            })
        }

        fn subsets<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            let ids = [
                "retained_vacuum_inside_film",
                "scalar_rectangle_depth_jump",
                "scalar_rectangle_two_shocks",
                "scalar_rectangle_uniform_rest",
            ];
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let rows = Tensor::from_slice(&[0_i64, 0], &[2], client.device())?;
                let before = cache(pair)?;
                let mut cold_simpson = None;
                for _ in 0..2 {
                    let before_work = work.snapshot();
                    let result = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [WetRootStatus::Located as i64; 2]
                    );
                    super::certificates::check(pair, &result)?;
                    let after = work.snapshot();
                    assert_eq!(
                        after.residual_points - before_work.residual_points,
                        result.callback_points()
                    );
                    assert!(result.callback_points() >= 2);
                    let current = cache(pair)?;
                    for (entry, original) in current.iter().zip(&before) {
                        assert_eq!(entry.ready[1..], original.ready[1..]);
                    }
                    let simpson = after.simpson_value_points - before_work.simpson_value_points;
                    if let Some(cold) = cold_simpson {
                        assert!(simpson <= cold);
                    } else {
                        cold_simpson = Some(simpson);
                    }
                }
                Ok(())
            })
        }

        fn budgets<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            let ids = [
                "scalar_rectangle_uniform_rest",
                "scalar_rectangle_vacuum",
                "retained_vacuum_inside_film",
            ];
            for budget in [1, 2, 3] {
                chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                    let rows = Tensor::from_slice(&[0_i64, 1, 2], &[3], client.device())?;
                    let result = hydraulic(locate_wet_roots(pair, &rows, budget))?;
                    let expected = [
                        WetRootStatus::Located as i64,
                        if budget == 3 {
                            WetRootStatus::Vacuum as i64
                        } else {
                            WetRootStatus::WorkLimit as i64
                        },
                        WetRootStatus::WorkLimit as i64,
                    ];
                    assert_eq!(result.status.try_to_vec::<i64>()?, expected);
                    assert_eq!(result.callback_points(), if budget == 3 { 3 } else { 1 });
                    assert_eq!(work.snapshot().residual_points, result.callback_points());
                    super::certificates::check(pair, &result)
                })?;
            }
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_uniform_rest"],
                &config,
                |pair, work| {
                    let rows = Tensor::from_slice(&[0_i64, 0, 0], &[3], client.device())?;
                    let result = hydraulic(locate_wet_roots(pair, &rows, 2))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [WetRootStatus::WorkLimit as i64; 3]
                    );
                    assert_eq!(result.callback_points(), 0);
                    assert_eq!(work.snapshot().residual_points, 0);
                    assert!(result.domains.is_empty() && result.evidence.is_empty());
                    Ok(())
                },
            )
        }

        fn multi_chart_budget<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>
                + CertifiedSimpsonAlgorithms<R>
                + CertifiedBisectionAlgorithms<R>
                + CertifiedUpperBracketAlgorithms<R>,
        {
            let config = chart_fixture::config();
            for budget in [4, config.root.max_points] {
                chart_fixture::with_pair::<R, _>(
                    client,
                    &["retained_vacuum_inside_film"],
                    &config,
                    |pair, work| {
                        let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                        let result = hydraulic(locate_wet_roots(pair, &rows, budget))?;
                        let counts = work.snapshot();
                        assert_eq!(result.callback_points(), counts.residual_points);
                        assert!(result.callback_points() <= budget);
                        assert!(counts.bracket_solves > 0);
                        if budget == 4 {
                            assert_eq!(
                                result.status.try_to_vec::<i64>()?,
                                [WetRootStatus::WorkLimit as i64]
                            );
                            assert!(result.domains.is_empty());
                        } else {
                            assert_eq!(
                                result.status.try_to_vec::<i64>()?,
                                [WetRootStatus::Located as i64]
                            );
                            assert!(counts.bracket_solves >= 2);
                            super::certificates::check(pair, &result)?;
                        }
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
    }
}
