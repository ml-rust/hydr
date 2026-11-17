use super::PrismaticRiemannConfig;
use super::acceptance::{accept, acceptance_mask};
use super::error::{
    HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked, into_public,
};
use super::sample_merge::Samples;
use super::types::CapturedInput;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn gates<R>(client: &R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let config = PrismaticRiemannConfig {
        water_flux_atol_m3_per_s: 1.0,
        momentum_flux_atol_m4_per_s2: 2.0,
        flux_rtol: 0.0,
        ..Default::default()
    };
    checked::<R, _>(
        client,
        "per-face acceptance requires both outward width gates",
        |a| {
            let mut solution = Samples::new(a, 5)?.solution;
            let lower = Tensor::from_slice(&[0.0; 5], &[5], client.device())?;
            solution.water_flux = a.bounds(
                &lower,
                &Tensor::from_slice(
                    &[1.0, 1.0_f64.next_up(), 0.5, 1.0_f64.next_down(), 0.0],
                    &[5],
                    client.device(),
                )?,
            )?;
            solution.momentum_flux = a.bounds(
                &lower,
                &Tensor::from_slice(
                    &[2.0, 1.0, 2.0_f64.next_up(), 2.0_f64.next_down(), 0.0],
                    &[5],
                    client.device(),
                )?,
            )?;
            let mask = acceptance_mask(a, &solution, &config)?;
            assert_eq!(mask.dtype(), DType::F64);
            assert_eq!(mask.try_to_vec::<f64>()?, [1.0, 0.0, 0.0, 1.0, 1.0]);
            let empty = Samples::new(a, 0)?.solution;
            assert_eq!(acceptance_mask(a, &empty, &config)?.shape(), &[0]);
            let input = CapturedInput {
                owners: Tensor::from_slice(&[2_i64, 1, 2, 1, 2], &[5], client.device())?,
                left_area: lower.clone(),
                left_discharge: lower.clone(),
                right_area: lower.clone(),
                right_discharge: lower,
            };
            let error = accept(a, &solution, &input, &config);
            assert!(matches!(
                error,
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::FluxBudgetExceeded,
                    query: Some(1),
                    section_owner: Some(1),
                    ..
                })
            ));
            Ok(())
        },
    )
}

#[test]
fn acceptance_masks_require_both_flux_gates_cpu() -> Result<(), super::PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    gates::<CpuRuntime>(&client).map_err(|error| into_public(error, Default::default()))
}

#[cfg(feature = "cuda")]
#[test]
fn acceptance_masks_require_both_flux_gates_cuda() -> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    gates::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}

#[path = "refinement_public_tests.rs"]
mod public;

#[path = "refinement_scatter_tests.rs"]
mod scatter;
