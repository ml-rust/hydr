use super::solve_chart;
use crate::channel::flow::surveyed::error;
use crate::channel::flow::surveyed_points::{
    potential_range_tests::frozen::original_film,
    source_tests::support::{Client, Result, hydraulic},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedRootStatus},
};

#[path = "chart_root_tests_controls.rs"]
mod controls;
#[path = "chart_root_tests_domains.rs"]
mod domains;
#[path = "chart_root_tests_scaling.rs"]
mod scaling;
#[path = "chart_root_tests_support.rs"]
mod support;

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
{
    let config = support::config();
    let ids = [
        "scalar_rectangle_uniform_rest",
        "scalar_triangle_uniform_flow",
        "scalar_rectangle_uniform_rest",
    ];
    support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let chart = support::chart(
            pair,
            support::Anchor::Source,
            [1.0_f64.to_bits(), 0],
            &[[-0.25, 0.25], [0.25, 0.5], [-0.25, 0.25]],
        )?;
        let parent_values = [2_i64, 0, 2, 1];
        let parents = Tensor::from_slice(&parent_values, &[4], client.device())?;
        let chart = support::select(pair, &chart, &parents)?;
        let result = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
        assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
        let statuses = result.result.status().try_to_vec::<i64>()?;
        assert_eq!(statuses.len(), 4);
        assert_eq!(statuses[3], CertifiedRootStatus::NoRoot as i64);
        assert!(statuses[..3].iter().all(|&code| matches!(
            CertifiedRootStatus::from_code(code),
            Some(
                CertifiedRootStatus::Converged
                    | CertifiedRootStatus::AccuracyFloor
                    | CertifiedRootStatus::WorkLimit
            )
        )));
        assert_eq!(result.root_rows.try_to_vec::<i64>()?, [2, 0, 2]);
        assert_eq!(result.result.root_owner().try_to_vec::<i64>()?, [0, 1, 2]);
        assert_eq!(
            result.query.query.original_rows.try_to_vec::<i64>()?,
            [103, 101, 103]
        );
        let expected_owner = pair.sources[0].owners().try_to_vec::<i64>()?[0];
        assert_eq!(
            result.query.query.owners.try_to_vec::<i64>()?,
            [expected_owner; 3]
        );
        let lower = result.result.root_lower().try_to_vec::<f64>()?;
        let upper = result.result.root_upper().try_to_vec::<f64>()?;
        for (&lo, &hi) in lower.iter().zip(&upper) {
            assert!(lo <= 0.0 && hi >= 0.0);
        }
        assert_eq!(work.snapshot().root_solves, 1);
        assert_eq!(work.snapshot().root_iterations, result.result.iterations());
        assert_eq!(
            work.snapshot().residual_points,
            result.result.evaluated_points()
        );
        client.copy_into(&parents, &Tensor::zeros(&[4], DType::I64, client.device())?)?;
        assert_eq!(result.invocation_rows.try_to_vec::<i64>()?, parent_values);
        assert_eq!(result.root_rows.try_to_vec::<i64>()?, [2, 0, 2]);
        assert_eq!(
            result.query.query.original_rows.try_to_vec::<i64>()?,
            [103, 101, 103]
        );
        Ok(())
    })?;
    support::with_pair::<R, _>(
        client,
        &["retained_vacuum_inside_film"],
        &config,
        |pair, work| {
            let scale = original_film::SONIC_BRACKET.scale_components;
            let upper = f64::from_bits(original_film::SONIC_BRACKET.parameter_bounds[0]);
            let chart = support::chart(pair, support::Anchor::Zero, scale, &[[0.0, upper]])?;
            let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let roots = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
            assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
            let star = f64::from_bits(original_film::STAR.area_bounds[0]);
            assert_eq!(
                original_film::STAR.area_bounds[0],
                original_film::STAR.area_bounds[1]
            );
            let [lower, upper] = support::physical_endpoints(pair, &roots)?;
            let (lo, hi) = hydraulic(error::checked::<R, _>(
                client,
                "returned film root physical endpoints",
                |a| {
                    Ok((
                        a.expansion_interval(&lower)?.lower().item::<f64>()?,
                        a.expansion_interval(&upper)?.upper().item::<f64>()?,
                    ))
                },
            ))?;
            assert!(
                lo <= star && star <= hi,
                "film root [{lo:e}, {hi:e}] excludes {star:e}"
            );
            let relative_gate = f64::from_bits(0x3ddb7cdfd9d7bdba);
            assert!(
                (hi - lo) / star <= relative_gate,
                "film width: [{lo:e}, {hi:e}]"
            );
            assert!((star - lo).max(hi - star) / star <= relative_gate);
            assert_eq!(
                work.snapshot().residual_points,
                roots.result.evaluated_points()
            );
            Ok(())
        },
    )?;
    support::with_pair::<R, _>(
        client,
        &["uniform_cap_retained_deficit_dyadic_velocity"],
        &config,
        |pair, _| {
            let deficit = f64::from_bits(0x3af0000000000000);
            let chart = support::chart(
                pair,
                support::Anchor::Cap,
                [1.0_f64.to_bits(), 0],
                &[[0.0, 2.0 * deficit]],
            )?;
            let parents = Tensor::from_slice(&[0_i64], &[1], client.device())?;
            let roots = hydraulic(solve_chart(pair, &parents, &chart, config.root.max_points))?;
            assert_eq!(roots.root_rows.try_to_vec::<i64>()?, [0]);
            assert!(roots.result.root_lower().item::<f64>()? <= deficit);
            assert!(roots.result.root_upper().item::<f64>()? >= deficit);
            let [lower, upper] = support::physical_endpoints(pair, &roots)?;
            hydraulic(error::checked::<R, _>(
                client,
                "returned deficit root retained physical endpoints",
                |a| {
                    let high = Tensor::from_slice(&[14.0], &[1], client.device())?;
                    let low = Tensor::from_slice(&[-deficit], &[1], client.device())?;
                    let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                    let truth = a.expansion_bounds(&high, &low, &zero)?;
                    let lower_difference = a.expansion_sub(&lower, &truth)?;
                    let upper_difference = a.expansion_sub(&upper, &truth)?;
                    assert!(
                        a.expansion_interval(&lower_difference)?
                            .lower()
                            .item::<f64>()?
                            <= 0.0
                    );
                    assert!(
                        a.expansion_interval(&upper_difference)?
                            .upper()
                            .item::<f64>()?
                            >= 0.0
                    );
                    Ok(())
                },
            ))?;
            Ok(())
        },
    )?;
    controls::run::<R>(client)
}

#[test]
fn configured_cpu_chart_roots_preserve_certificates_and_film_precision() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let base = CpuClient::new(CpuDevice::new());
    let before = base.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = base.with_parallelism(config);
    matrix::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(base.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_chart_roots_preserve_certificates_and_film_precision() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
