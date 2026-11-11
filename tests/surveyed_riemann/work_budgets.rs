use super::data::CASE_GROUPS;
use super::support::{self, Client, Result};
use hydr::channel::{PrismaticRiemannError, PrismaticRiemannRejection};
use numr::dtype::DType;
use numr::runtime::Runtime;

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(&client)?;
    let case = CASE_GROUPS
        .iter()
        .flat_map(|group| group.iter())
        .find(|case| case.id == "three_bands_depth_jump")
        .ok_or("Missing frozen multi-band work reference")?;
    let input = support::inputs::<R>(&client, &[case], false)?;
    let mut config = support::config(case);
    let accepted = batch.riemann_flux_tensor(
        &input[0], &input[1], &input[2], &input[3], &input[4], &config,
    )?;
    super::accuracy::check(&accepted, &[case], &config)?;
    let work = accepted.work();
    let total = work
        .residual_points
        .checked_add(work.simpson_value_points)
        .and_then(|n| n.checked_add(work.simpson_derivative_points))
        .ok_or("Executed work overflow")?;
    assert!(total > 1);
    assert!(work.root_solves > 0);
    assert!(work.residual_points > 0);
    assert!(work.simpson_value_points > 0);
    assert!(work.simpson_derivative_points > 0);
    config.max_native_query_points = total - 1;
    match batch.riemann_flux_tensor(
        &input[0], &input[1], &input[2], &input[3], &input[4], &config,
    ) {
        Err(PrismaticRiemannError::Rejected {
            reason,
            work,
            context,
            ..
        }) => {
            assert_eq!(reason, PrismaticRiemannRejection::WorkLimit, "{context}");
            support::work(work, &config)?;
            let executed =
                work.residual_points + work.simpson_value_points + work.simpson_derivative_points;
            assert!(executed < total);
        }
        Err(error) => return Err(format!("Cumulative work ceiling returned {error}").into()),
        Ok(_) => {
            return Err("Accepted less work than the unchanged successful solve executes".into());
        }
    }
    super::accuracy::check(&accepted, &[case], &support::config(case))?;
    Ok(())
}

#[test]
fn configured_cpu_charges_every_native_callback_before_execution() -> Result {
    run::<numr::runtime::cpu::CpuRuntime>(support::cpu())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_charges_every_native_callback_before_execution() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
