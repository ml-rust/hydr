use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

/// Retain a depth point only when its complete pointed-area identity proves exact equality.
pub(super) fn refine<R>(
    a: &mut Arithmetic<'_, R>,
    pointed: &Tensor<R>,
    area: &CertifiedExpansion<R>,
    depth: &Interval<R>,
    slope: &Interval<R>,
    end_depth: &Interval<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(area.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(area.high().shape(), a.dtype(), c.device())?;
    let candidate = c.midpoint(depth.lower(), depth.upper())?;
    let singleton_slope = c.mul(
        &c.eq(slope.lower(), slope.upper())?,
        &c.gt(slope.lower(), &zero)?,
    )?;
    let inside = c.mul(
        &c.ge(&candidate, depth.lower())?,
        &c.mul(
            &c.le(&candidate, depth.upper())?,
            &c.le(&candidate, end_depth.lower())?,
        )?,
    )?;
    let active = c.mul(
        pointed,
        &c.mul(
            &singleton_slope,
            &c.mul(
                &inside,
                &c.mul(&c.gt(&candidate, &zero)?, &c.eq(area.abs_error(), &zero)?)?,
            )?,
        )?,
    )?;

    let candidate_point = a.expansion_point(&c.where_cond(&active, &candidate, &zero)?)?;
    let slope_point = a.expansion_point(&c.where_cond(&active, slope.lower(), &one)?)?;
    let proof_area = a.expansion_bounds(
        &c.where_cond(&active, area.high(), &zero)?,
        &c.where_cond(&active, area.low(), &zero)?,
        &c.where_cond(&active, area.abs_error(), &zero)?,
    )?;
    let half = a.expansion_point(&c.mul_scalar(&one, 0.5)?)?;
    let half_depth = a.expansion_mul(&candidate_point, &half)?;
    // The proved band bounds give m*h <= end width and m*h²/2 <= end area.
    // Halving before each product leaves headroom without changing the identity.
    let half_width = a.expansion_mul(&slope_point, &half_depth)?;
    let quarter_product = a.expansion_mul(&half_width, &half_depth)?;
    let half_area = a.expansion_mul(&proof_area, &half)?;
    let residual = a.expansion_sub(&quarter_product, &half_area)?;
    let mut exact = c.mul(
        &active,
        &c.mul(
            &c.eq(residual.high(), &zero)?,
            &c.eq(residual.low(), &zero)?,
        )?,
    )?;
    for value in [
        &half_depth,
        &half_width,
        &quarter_product,
        &half_area,
        &residual,
    ] {
        exact = c.mul(&exact, &c.eq(value.abs_error(), &zero)?)?;
    }

    let fallback = a.expansion_from_interval(depth)?;
    Ok(a.expansion_bounds(
        &c.where_cond(&exact, &candidate, fallback.high())?,
        &c.where_cond(&exact, &zero, fallback.low())?,
        &c.where_cond(&exact, &zero, fallback.abs_error())?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::{
        surveyed::error,
        surveyed_points::source_tests::support::{Client, Result, components, hydraulic},
    };
    use numr::runtime::RuntimeClient;

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
}
