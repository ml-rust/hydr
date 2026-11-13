use super::refine;
use crate::channel::flow::{
    surveyed::error,
    surveyed_points::source_tests::support::{Client, Result, components, hydraulic},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let (result, fallback) = hydraulic(error::checked::<R, _>(
        client,
        "optional pointed depth identity controls",
        |a| {
            let tiny = f64::from_bits(1);
            // m*h²/2=2^-1074, while its half-area 2^-1075 cannot be represented.
            let depth = f64::from_bits(0x1e60000000000000);
            let perturbation = f64::from_bits(0x3af0000000000000);
            let n = 8;
            let tensor = |values: &[f64]| Tensor::from_slice(values, &[n], client.device());
            let pointed = tensor(&[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;
            let area = a.expansion_bounds(
                &tensor(&[1.0, 1.0, 1.0, 1.0, tiny, f64::MAX, f64::MAX, f64::MAX])?,
                &tensor(&[0.0, perturbation, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])?,
                &tensor(&[0.0, 0.0, perturbation, 0.0, 0.0, 0.0, 0.0, 1.0])?,
            )?;
            let depths = a.bounds(
                &tensor(&[0.75, 0.75, 0.75, 0.75, 0.75 * depth, 1e200, 1e200, 1e200])?,
                &tensor(&[
                    1.25,
                    1.25,
                    1.25,
                    1.25,
                    1.25 * depth,
                    1.25e200,
                    1.25e200,
                    1.25e200,
                ])?,
            )?;
            let slopes = a.bounds(
                &tensor(&[2.0, 2.0, 2.0, 1.5, 2.0, f64::MAX, 1e200, f64::MAX])?,
                &tensor(&[2.0, 2.0, 2.0, 2.5, 2.0, f64::MAX, 1.25e200, f64::MAX])?,
            )?;
            let end_depth = a.point(&tensor(&[
                2.0,
                2.0,
                2.0,
                2.0,
                2.0 * depth,
                1.5e200,
                1.5e200,
                1.5e200,
            ])?)?;
            let fallback = a.expansion_from_interval(&depths)?;
            let result = refine(a, &pointed, &area, &depths, &slopes, &end_depth)?;
            let physical = a.expansion_interval(&result)?;
            let lower = physical.lower().try_to_vec::<f64>()?;
            let upper = physical.upper().try_to_vec::<f64>()?;
            assert!(lower[4] < upper[4]);
            assert!(lower[4] <= depth && depth <= upper[4]);
            Ok((result, fallback))
        },
    ))?;
    let expected = components(&fallback)?;
    let actual = components(&result)?;
    assert_eq!(actual[0], [1.0_f64.to_bits(), 0, 0]);
    let labels = [
        "exact pointed identity",
        "retained area perturbation",
        "area uncertainty",
        "non-singleton slope",
        "half-area underflow",
        "inactive pointed mask",
        "inactive non-singleton huge slope",
        "inactive uncertain huge area",
    ];
    for row in 1..labels.len() {
        assert_eq!(actual[row], expected[row], "{}", labels[row]);
    }
    assert!(f64::from_bits(actual[4][2]) > 0.0);
    Ok(())
}

#[test]
fn configured_cpu_pointed_depth_identity_retains_exactness_and_safe_fallbacks() -> Result {
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
fn actual_cuda_pointed_depth_identity_retains_exactness_and_safe_fallbacks() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
