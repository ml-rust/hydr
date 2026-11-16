use super::super::{
    chart_test_support,
    corpus_coverage::Coverage,
    point_candidates::tests::{fixtures, flux_fixtures},
    point_flux_types::{PointFluxKind, PointFluxStatus},
    point_root_session::PointRootSession,
    source_sonic_quadrature_types::SourceSonicQuadratureProgress,
    source_sonic_types::SonicSide,
    source_tests::support as frozen,
    wet_depth::tests::support,
};
use crate::channel::flow::surveyed::{PrismaticRiemannConfig, work::PrismaticRiemannWork};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "source_sonic_quadrature_tests_admission.rs"]
mod admission;
#[path = "source_sonic_quadrature_tests_controls.rs"]
mod controls;
#[path = "source_sonic_quadrature_tests_corpus.rs"]
mod corpus;
#[path = "source_sonic_quadrature_tests_ownership.rs"]
mod ownership;
#[path = "source_sonic_quadrature_tests_support.rs"]
mod snapshots;

pub(in crate::channel::flow::surveyed_points) const CASES: [&str; 7] = [
    "scalar_trapezoid_depth_jump",
    "scalar_three_bands_depth_jump",
    "scalar_three_bands_both_knots_dry_fan",
    "scalar_three_bands_both_knots_dry_fan_reverse",
    "scalar_planned_three_bands_both_knots_dry_fan",
    "scalar_planned_three_bands_both_knots_near_pure_fan",
    "scalar_three_bands_loose_recovery_allowance",
];

fn config(id: &str) -> Result<PrismaticRiemannConfig> {
    let rows = fixtures::executions();
    let row = rows
        .iter()
        .find(|row| row.id == id)
        .ok_or("Missing frozen sonic quadrature execution")?;
    Ok(fixtures::CONFIGS[row.config].build())
}

fn side(id: &str) -> Result<SonicSide> {
    let rows = fixtures::executions();
    let row = rows
        .iter()
        .find(|row| row.id == id)
        .ok_or("Missing frozen sonic quadrature execution")?;
    match row.traces.first().map(|trace| trace.label) {
        Some("left-sonic") => Ok(SonicSide::Left),
        Some("right-sonic") => Ok(SonicSide::Right),
        _ => Err("Frozen sonic quadrature execution lacks its sampled sonic trace".into()),
    }
}

fn mapped<R: Runtime<DType = DType>>(session: &PointRootSession<'_, '_, R>) -> usize {
    session
        .sonic
        .requests
        .iter()
        .flat_map(|request| &request.domains)
        .map(|domain| domain.mapped_points)
        .sum()
}

fn work_delta<R>(
    progress: &SourceSonicQuadratureProgress<R>,
    before: PrismaticRiemannWork,
    after: PrismaticRiemannWork,
    old_mapped: usize,
    new_mapped: usize,
) where
    R: Runtime<DType = DType>,
{
    assert_eq!(
        after.residual_points.checked_sub(before.residual_points),
        Some(progress.additional_residual_points)
    );
    assert_eq!(
        after
            .simpson_value_points
            .checked_sub(before.simpson_value_points),
        Some(progress.additional_simpson_value_points)
    );
    assert_eq!(
        after
            .simpson_derivative_points
            .checked_sub(before.simpson_derivative_points),
        Some(progress.additional_simpson_derivative_points)
    );
    assert_eq!(
        new_mapped.checked_sub(old_mapped),
        Some(progress.additional_mapped_points)
    );
}

fn contains(actual: [f64; 2], expected: [u64; 2], id: &str) {
    let expected = expected.map(f64::from_bits);
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{id}: refined sonic flux {actual:?} excludes independent truth {expected:?}"
    );
}

/// Ownership, admission, and level controls, checked on the CPU.
fn session_controls<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    ownership::shared_parent::<R>(client)?;
    admission::provenance::<R>(client)?;
    admission::stale_generation::<R>(client)?;
    controls::bounded_levels::<R>(client)?;
    controls::exhausted::<R>(client)?;
    controls::charged_error::<R>(client)?;
    Ok(())
}

#[test]
fn configured_cpu_source_sonic_quadrature_narrows_frozen_fluxes() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    corpus::run::<CpuRuntime>(&client, Coverage::Full)?;
    session_controls::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_source_sonic_quadrature_narrows_frozen_fluxes() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    corpus::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
}
