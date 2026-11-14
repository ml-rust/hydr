use super::super::{
    chart_test_support,
    point_flux_types::{PointFluxKind, PointFluxStatus},
    point_root_session::PointRootSession,
    source_tests::support as frozen,
    wet_depth::tests::support,
};
use frozen::{Result, hydraulic};
use numr::{dtype::DType, runtime::Runtime};

#[path = "point_flux_tests_controls.rs"]
mod controls;
#[path = "point_flux_tests_corpus.rs"]
mod corpus;
#[path = "point_flux_tests_diagnostics.rs"]
mod diagnostics;
#[path = "point_flux_tests_support.rs"]
mod positive_support;
#[path = "point_flux_tests_precision.rs"]
mod precision;
#[path = "point_flux_tests_pressure.rs"]
mod pressure;
#[path = "point_flux_tests_quantum.rs"]
mod quantum;

fn basic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 1., 1., 1.], [0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[1, 0, 0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let demand = support::rows::<R>(client, &[2, 0, 1])?;
            let result = hydraulic(session.select_point_fluxes(&demand))?;
            hydraulic(
                session
                    .invocation
                    .check(&result.candidates.invocation_witness),
            )?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointFluxStatus::Accepted as i64; 3]
            );
            let water = frozen::endpoints(&result.physical_water)?;
            let momentum = frozen::endpoints(&result.physical_momentum)?;
            for (row, q, j) in [(0, 1., 1.5), (1, 0., 0.), (2, 1., 1.5)] {
                assert!(water[row][0] <= q && q <= water[row][1]);
                assert!(momentum[row][0] <= j && j <= momentum[row][1]);
            }
            let mut count = 0;
            for point in &result.selected {
                let area = frozen::components(&point.area)?;
                let q = frozen::components(&point.discharge)?;
                for row in 0..area.len() {
                    let value = if point.kind == PointFluxKind::Dry {
                        0.0_f64
                    } else {
                        1.0_f64
                    };
                    assert_eq!(area[row], [value.to_bits(), 0, 0]);
                    assert_eq!(q[row], [value.to_bits(), 0, 0]);
                    count += 1;
                }
            }
            assert_eq!(count, 3);
            assert_eq!(result.callback_points, session.callback_points());
            assert_eq!(result.point_limit, config.root.max_points);
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            Ok(())
        },
    )
}

#[test]
fn configured_cpu_point_fluxes_select_coherent_conserved_states() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    basic::<CpuRuntime>(&client)?;
    corpus::run::<CpuRuntime>(&client)?;
    controls::run::<CpuRuntime>(&client)?;
    precision::run::<CpuRuntime>(&client)?;
    positive_support::run::<CpuRuntime>(&client)?;
    quantum::run::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_point_fluxes_select_coherent_conserved_states() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    basic::<CudaRuntime>(&client)?;
    corpus::run::<CudaRuntime>(&client)?;
    controls::run::<CudaRuntime>(&client)?;
    precision::run::<CudaRuntime>(&client)?;
    positive_support::run::<CudaRuntime>(&client)?;
    quantum::run::<CudaRuntime>(&client)
}
