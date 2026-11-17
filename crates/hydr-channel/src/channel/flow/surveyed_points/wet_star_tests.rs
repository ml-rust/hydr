use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
use crate::channel::flow::surveyed_points::{
    chart_test_support, source_tests::support as frozen, wet_depth::refine_wet_depth,
    wet_roots::locate_wet_roots, wet_star_types::WetStarStatus,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

use crate::channel::flow::surveyed_points::wet_depth::tests::support;
#[path = "wet_star_tests_admission.rs"]
mod admission;
#[path = "wet_star_tests_diagnostics.rs"]
mod diagnostics;
#[path = "wet_star_tests_ranges.rs"]
mod ranges;
use diagnostics::check_diagnostics;

fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    exact_rectangular_star::<R>(client)?;
    ranges::full_domain_discharge::<R>(client)?;
    ranges::empty_intersections::<R>(client)?;
    frozen_star_states::<R>(client, coverage)?;
    admission::run::<R>(client)
}

fn exact_rectangular_star<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    // Phi(A)=2*sqrt(A). Sources (4,-4),(4,12) give A*=1,u*=1,Q*=1.
    support::with_rectangle::<R, _>(
        client,
        1.0,
        8.0,
        &[[4.0, -4.0, 4.0, 12.0], [1.0, 0.125, 1.0, 0.125]],
        &config,
        |pair, work| {
            let rows = support::rows::<R>(client, &[1, 0, 1, 0])?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut session = hydraulic(refine_wet_depth(pair, located))?;
            let points = session.callback_points();
            let statuses = session.locator_status().try_to_vec::<i64>()?;
            let mut identities = Vec::new();
            for domain in session.domains() {
                identities.push((
                    domain.invocation_owner.try_to_vec::<i64>()?,
                    domain.parent_rows.try_to_vec::<i64>()?,
                    domain.query.query.original_rows.try_to_vec::<i64>()?,
                    frozen::endpoints(&domain.query.query.parameter)?,
                ));
            }
            let before = work.snapshot();
            let certificates = hydraulic(session.certify_wet_star())?;
            let after = work.snapshot();
            assert_eq!(after.residual_points, before.residual_points);
            assert_eq!(after.root_iterations, before.root_iterations);
            assert_eq!(after.root_solves, before.root_solves);
            assert_eq!(session.callback_points(), points);
            assert_eq!(session.locator_status().try_to_vec::<i64>()?, statuses);
            assert_eq!(certificates.len(), identities.len());
            for (domain, (owners, parents, original_rows, parameter)) in
                certificates.iter().zip(identities)
            {
                check_diagnostics(session.sources(), domain)?;
                assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, owners);
                assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, parents);
                assert_eq!(
                    domain.query.query.original_rows.try_to_vec::<i64>()?,
                    original_rows
                );
                assert_eq!(frozen::endpoints(&domain.query.query.parameter)?, parameter);
                assert!(
                    domain
                        .status
                        .try_to_vec::<i64>()?
                        .iter()
                        .all(|&x| x == WetStarStatus::Certified as i64)
                );
                let velocity = frozen::endpoints(&domain.velocity)?;
                let discharge = frozen::endpoints(&domain.discharge)?;
                for (index, parent) in parents.into_iter().enumerate() {
                    let expected = if parent == 0 { 1.0 } else { 0.125 };
                    assert!(velocity[index][0] <= expected && expected <= velocity[index][1]);
                    assert!(discharge[index][0] <= expected && expected <= discharge[index][1]);
                    if parent == 1 {
                        assert_eq!(discharge[index], [0.125; 2]);
                    }
                }
            }
            let cache = chart_test_support::cache(session.sources())?;
            let repeated = hydraulic(session.certify_wet_star())?;
            assert_eq!(repeated.len(), certificates.len());
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            assert_eq!(work.snapshot(), after);
            // Certificate construction leaves continuation available for stricter physical stopping.
            let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            assert!(session.callback_points() >= points);
            Ok(())
        },
    )
}

fn frozen_star_states<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    // references.json supplies new_independent_truth.star and weak-cap reused_truths_verbatim.truth.star.
    // Weak-cap rational star bounds project outward to these binary64 bounds. Film sampled flux is sonic.
    let truths = [
        (
            "retained_vacuum_inside_film",
            [0xbae0000000000000, 0xbae0000000000000],
            [0xb0a0000000000000, 0xb0a0000000000000],
        ),
        (
            "uniform_cap_retained_deficit_dyadic_velocity",
            [0x3fc0000000000000, 0x3fc0000000000000],
            [0x3ffbffffffffffff, 0x3ffc000000000000],
        ),
        (
            "trace_origin_after_retained_history_face_193",
            [0x3c8b706c91b2f8d9, 0x3c8b706c91b2f8da],
            [0x3cc8025eff7c99be, 0x3cc8025eff7c99bf],
        ),
        (
            "retained_knot_2_crossing",
            [0xbada20bd700c2c3e, 0xbada20bd700c2c3d],
            [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
        ),
    ];
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(client, &truths.map(|t| t.0), &config, |pair, work| {
        // The device replays the knot jump and the film, one star of each kind.
        let parents: &[i64] = if coverage.is_full() {
            &[3, 0, 1, 3, 2]
        } else {
            &[3, 0]
        };
        let rows = support::rows::<R>(client, parents)?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        let session = hydraulic(refine_wet_depth(pair, located))?;
        let mut session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        }))?;
        let before = work.snapshot();
        let certificates = hydraulic(session.certify_wet_star())?;
        assert_eq!(work.snapshot().residual_points, before.residual_points);
        assert_eq!(work.snapshot().root_iterations, before.root_iterations);
        let mut count = 0;
        for domain in &certificates {
            check_diagnostics(session.sources(), domain)?;
            diagnostics::check_frozen_differences(session.sources(), domain)?;
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            let velocity = frozen::endpoints(&domain.velocity)?;
            let discharge = frozen::endpoints(&domain.discharge)?;
            assert!(
                domain
                    .status
                    .try_to_vec::<i64>()?
                    .iter()
                    .all(|&s| s == WetStarStatus::Certified as i64)
            );
            for (index, parent) in parents.into_iter().enumerate() {
                let truth = truths[usize::try_from(parent)?];
                for (actual, expected) in [(velocity[index], truth.1), (discharge[index], truth.2)]
                {
                    let expected = expected.map(f64::from_bits);
                    assert!(
                        actual[0] <= expected[0] && actual[1] >= expected[1],
                        "{}: {actual:?}, {expected:?}",
                        truth.0
                    );
                    if parent == 0 {
                        assert!(
                            actual[1] - actual[0] < expected[0].abs() * 1e-10,
                            "film precision: {actual:?}"
                        );
                    }
                }
                count += 1;
            }
        }
        assert_eq!(count, parents.len());
        Ok(())
    })
}

#[test]
fn configured_cpu_wet_star_certificates_preserve_root_state_and_anchors() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    matrix::<CpuRuntime>(&client, Coverage::Full)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_wet_star_certificates_preserve_root_state_and_anchors() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
}
