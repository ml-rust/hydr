use super::pointed_depth;
use crate::channel::section::prismatic::query_input::QueryError;
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

type Result = std::result::Result<(), Box<dyn std::error::Error>>;

struct FormulaCase {
    id: &'static str,
    slope: u64,
    area: u64,
    depth: [u64; 2],
}

#[path = "../../../../tests/prismatic_area_points/formula_data.rs"]
mod formula_data;

fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    for case in formula_data::FORMULAS {
        let area = Tensor::<R>::from_slice(&[f64::from_bits(case.area)], &[1], client.device())?;
        let slope = Tensor::<R>::from_slice(&[f64::from_bits(case.slope)], &[1], client.device())?;
        let depth = with_interval_arithmetic::<R, _, _, _>(
            &client,
            DType::F64,
            "pointed inverse range control",
            |a| {
                let area = a.point(&area)?;
                let slope = a.point(&slope)?;
                Ok(pointed_depth(a, &area, &slope))
            },
        )?;
        let depth = match depth {
            Ok(depth) => depth,
            Err(QueryError::Geometry(error)) => return Err(error.into()),
            Err(QueryError::Arithmetic(error)) => return Err(error.into()),
        };
        let lower = depth.lower().item::<f64>()?;
        let upper = depth.upper().item::<f64>()?;
        let reference = case.depth.map(f64::from_bits);
        assert!(lower.is_finite() && upper.is_finite(), "{}", case.id);
        assert!(
            0.0 < lower && lower <= reference[0] && reference[1] <= upper,
            "{}",
            case.id
        );
        assert!(upper - lower <= reference[0] * 1e-10, "{}", case.id);
    }
    Ok(())
}

#[test]
fn cpu_pointed_inverse_retains_finite_tiny_slope_depth() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    run::<CpuRuntime>(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pointed_inverse_retains_finite_tiny_slope_depth() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
