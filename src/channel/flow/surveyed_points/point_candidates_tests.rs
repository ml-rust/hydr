use super::super::{
    chart_test_support,
    point_candidate_types::{PointCandidate, PointCandidateStatus, PointFanAuthority, PointRegime},
    point_root_session::PointRootSession,
    source_tests::support as frozen,
    wet_depth::tests::support,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "point_candidates_tests_contacts.rs"]
mod contacts;
#[path = "point_candidates_tests_corpus.rs"]
mod corpus;
#[path = "point_candidates_tests_demand.rs"]
mod demand;
#[path = "point_candidates_tests_diagnostics.rs"]
mod diagnostics;
#[path = "point_candidates_fixture_export.rs"]
mod fixture_export;
#[path = "point_candidates_fixture_model.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixture_model;
#[path = "point_candidates_fixtures.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixtures;
#[path = "point_flux_fixture_export.rs"]
mod flux_fixture_export;
#[path = "point_flux_fixture_model.rs"]
pub(in crate::channel::flow::surveyed_points) mod flux_fixture_model;
#[path = "point_flux_fixtures.rs"]
pub(in crate::channel::flow::surveyed_points) mod flux_fixtures;
#[path = "point_candidates_tests_membership.rs"]
mod membership;
#[path = "point_candidates_tests_support.rs"]
mod observed;
#[path = "point_candidates_tests_ownership.rs"]
mod ownership;
#[path = "point_candidates_tests_precision.rs"]
mod precision;
#[path = "point_candidates_tests_shocks.rs"]
mod shocks;
#[path = "point_candidates_tests_telemetry.rs"]
mod telemetry;
#[path = "point_candidates_tests_wave_references.rs"]
mod wave_references;
#[path = "point_candidates_tests_work.rs"]
mod work;

#[test]
fn point_candidate_reference_fixture_export() -> Result {
    fixture_export::emit(&fixtures::executions())
}

#[test]
fn point_flux_reference_fixture_export() -> Result {
    flux_fixture_export::emit()
}

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 1., 0.], [0., 0., 0., 0.], [1., 0., 0., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 2, 0, 1])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let owners = support::rows::<R>(client, &[3, 2, 1, 0])?;
            let result = hydraulic(session.certify_point_candidates(&owners))?;
            hydraulic(session.invocation.check(&result.invocation_witness))?;
            assert_eq!(result.invocation_owner.try_to_vec::<i64>()?, [3, 2, 1, 0]);
            assert_eq!(result.parent_rows.try_to_vec::<i64>()?, [1, 0, 2, 0]);
            assert_eq!(
                result.regime.try_to_vec::<i64>()?,
                [
                    PointRegime::BothDry as i64,
                    PointRegime::Wet as i64,
                    PointRegime::LeftWet as i64,
                    PointRegime::Wet as i64
                ]
            );
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointCandidateStatus::Complete as i64; 4]
            );
            let mut starowners = Vec::new();
            let mut fanowners = Vec::new();
            let mut dryowners = Vec::new();
            for candidate in &result.candidates {
                match candidate {
                    PointCandidate::Star(star) => {
                        let wave = &result.waves[star.wave];
                        let owners = star.invocation_owner.try_to_vec::<i64>()?;
                        let rows = star.rows.try_to_vec::<i64>()?;
                        let waveowners = wave.star.invocation_owner.try_to_vec::<i64>()?;
                        let q = frozen::endpoints(&wave.star.discharge)?;
                        for (&owner, &row) in owners.iter().zip(&rows) {
                            assert_eq!(waveowners[usize::try_from(row)?], owner);
                            assert_eq!(q[usize::try_from(row)?], [0., 0.]);
                        }
                        starowners.extend(owners);
                    }
                    PointCandidate::Fan(fan) => {
                        assert!(matches!(fan.authority, PointFanAuthority::DryVacuum));
                        fanowners.extend(fan.certificate.invocation_owner.try_to_vec::<i64>()?);
                        assert_eq!(fan.active.try_to_vec::<f64>()?, [1.]);
                    }
                    PointCandidate::Dry(dry) => {
                        dryowners.extend(dry.invocation_owner.try_to_vec::<i64>()?)
                    }
                    PointCandidate::Exterior(_) => {
                        return Err("Resting uniform source produces exterior candidate".into());
                    }
                    PointCandidate::Pending(_) => {
                        return Err("Exact mixed control retains pending candidate".into());
                    }
                }
            }
            starowners.sort();
            assert_eq!(starowners, [0, 2]);
            assert_eq!(fanowners, [1]);
            assert_eq!(dryowners, [3]);
            let points = session.callback_points();
            let before = work.snapshot().residual_points;
            let repeated = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(repeated.status.try_to_vec::<i64>()?, [0; 4]);
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before);
            Ok(())
        },
    )?;
    corpus::run::<R>(client)?;
    contacts::run::<R>(client)?;
    precision::run::<R>(client)?;
    demand::run::<R>(client)?;
    membership::run::<R>(client)?;
    ownership::run::<R>(client)?;
    work::run::<R>(client)?;
    shocks::run::<R>(client)
}
#[test]
fn configured_cpu_point_candidates_preserve_all_regime_alternatives() -> Result {
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
fn actual_cuda_point_candidates_preserve_all_regime_alternatives() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
