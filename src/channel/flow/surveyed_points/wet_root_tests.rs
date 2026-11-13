use super::locate_wet_roots;
use crate::channel::flow::{
    surveyed::error::{HydraulicError, PrismaticRiemannRejection},
    surveyed_points::{
        chart_test_support,
        source_tests::support::{self as frozen, Client, Result, hydraulic},
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

#[path = "wet_root_tests_admission.rs"]
mod admission;
#[path = "wet_root_tests_cases.rs"]
mod cases;
#[path = "wet_root_tests_certificates.rs"]
mod certificates;
#[path = "wet_root_tests_chains.rs"]
mod chains;
#[path = "wet_root_tests_oracles.rs"]
mod oracles;
#[path = "wet_root_tests_selection.rs"]
mod selection;
#[path = "wet_root_tests_support.rs"]
mod support;

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    corpus::<R>(client)?;
    chains::run::<R>(client)?;
    admission::run::<R>(client)?;
    selection::run::<R>(client)?;
    oracles::run::<R>(client)
}

fn corpus<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    assert_eq!(frozen::cases().len(), 70);
    let cases = frozen::cases()
        .iter()
        .filter(|case| !case.dry[0] && !case.dry[1]);
    let mut executed = 0;
    for case in cases {
        let (config, expected) = cases::declared(case.id);
        chart_test_support::with_pair::<R, _>(client, &[case.id], &config, |pair, work| {
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
                result.callback_points,
                work.snapshot().residual_points,
                "{}",
                case.id
            );
            assert!(
                result.callback_points <= config.root.max_points,
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
    assert_eq!(executed, 61);
    Ok(())
}

#[test]
fn configured_cpu_wet_root_domains_preserve_frozen_outcomes_and_retained_roots() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    matrix::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_wet_root_domains_preserve_frozen_outcomes_and_retained_roots() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}

#[test]
fn configured_cpu_pointed_film_locator_preserves_independent_root() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    chains::run_pointed::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}
