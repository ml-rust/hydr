use super::support::{self, Client, Result};
use hydr::channel::{CrossSectionError, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn numerical<T>(result: std::result::Result<T, CrossSectionError>) -> Result {
    let error = result
        .err()
        .ok_or("Host projection accepted invalid native contact bounds")?;
    assert!(
        matches!(error, CrossSectionError::Numerical { .. }),
        "{error}"
    );
    Ok(())
}

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(&client)?;
    let config = PrismaticRecoveryConfig::default();
    for value in [0.0, 1.0] {
        let (owners, values) = support::inputs::<R>(&client, &[0], &[value], false)?;
        for field in 0..4 {
            let query = batch.evaluate_depths_tensor(&owners, &values)?;
            let recovered = batch.recover_depths_tensor(&owners, &values, &config)?;
            let query_fields = [
                query.wetted_perimeter_m().lower(),
                query.wetted_perimeter_m().upper(),
                query.hydraulic_radius_m().lower(),
                query.hydraulic_radius_m().upper(),
            ];
            let recovery_fields = [
                recovered.wetted_perimeter_m().lower(),
                recovered.wetted_perimeter_m().upper(),
                recovered.hydraulic_radius_m().lower(),
                recovered.hydraulic_radius_m().upper(),
            ];
            let replacement = Tensor::from_slice(&[f64::NAN], &[1], client.device())?;
            for input in [query_fields[field], recovery_fields[field]] {
                client.copy_into(input, &replacement)?;
            }
            numerical(query.properties())?;
            numerical(recovered.properties())?;
        }
        let query = batch.evaluate_depths_tensor(&owners, &values)?;
        let recovered = batch.recover_depths_tensor(&owners, &values, &config)?;
        let invalid = if value == 0.0 { 1.0 } else { 0.0 };
        let replacement = Tensor::from_slice(&[invalid], &[1], client.device())?;
        for interval in [query.wetted_perimeter_m(), recovered.wetted_perimeter_m()] {
            client.copy_into(interval.lower(), &replacement)?;
            client.copy_into(interval.upper(), &replacement)?;
        }
        numerical(query.properties())?;
        numerical(recovered.properties())?;
    }
    Ok(())
}

#[test]
fn cpu_host_projection_reports_invalid_native_contact_bounds() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_host_projection_reports_invalid_native_contact_bounds() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
