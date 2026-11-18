use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ReduceOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

use super::{
    config::PrismaticRiemannConfig,
    error::{Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection},
    sampling::SampledSolution,
    types::CapturedInput,
};

pub(crate) fn accept<R>(
    a: &mut Arithmetic<'_, R>,
    solution: &SampledSolution<R>,
    input: &CapturedInput<R>,
    config: &PrismaticRiemannConfig,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    for (name, flux, atol) in [
        (
            "water_flux_m3_per_s",
            &solution.water_flux,
            config.water_flux_atol_m3_per_s,
        ),
        (
            "momentum_flux_m4_per_s2",
            &solution.momentum_flux,
            config.momentum_flux_atol_m4_per_s2,
        ),
    ] {
        check_width(a, flux, input, atol, config.flux_rtol, name)?;
    }
    Ok(())
}

pub(crate) fn acceptance_mask<R>(
    a: &mut Arithmetic<'_, R>,
    solution: &SampledSolution<R>,
    config: &PrismaticRiemannConfig,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    flux_acceptance_mask(a, &solution.water_flux, &solution.momentum_flux, config)
}

pub(crate) fn flux_acceptance_mask<R>(
    a: &mut Arithmetic<'_, R>,
    water: &Interval<R>,
    momentum: &Interval<R>,
    config: &PrismaticRiemannConfig,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    if water.lower().numel() == 0 {
        return Ok(Tensor::zeros(&[0], DType::F64, c.device())?);
    }
    let (water_width, water_allowance) =
        width_bounds(a, water, config.water_flux_atol_m3_per_s, config.flux_rtol)?;
    let (momentum_width, momentum_allowance) = width_bounds(
        a,
        momentum,
        config.momentum_flux_atol_m4_per_s2,
        config.flux_rtol,
    )?;
    let water = c.le(water_width.upper(), water_allowance.lower())?;
    let momentum = c.le(momentum_width.upper(), momentum_allowance.lower())?;
    Ok(c.cast(&c.mul(&water, &momentum)?, DType::F64)?)
}

fn width_bounds<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    atol: f64,
    rtol: f64,
) -> HydraulicResult<(Interval<R>, Interval<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let lo = a.point(value.lower())?;
    let hi = a.point(value.upper())?;
    let width = a.sub(&hi, &lo)?;
    let magnitude = c.maximum(&c.abs(value.lower())?, &c.abs(value.upper())?)?;
    let magnitude = a.point(&magnitude)?;
    let relative = a.point(&Tensor::from_slice(&[rtol], &[], c.device())?)?;
    let absolute = a.point(&Tensor::from_slice(&[atol], &[], c.device())?)?;
    let relative = a.mul(&magnitude, &relative)?;
    let allowance = a.add(&absolute, &relative)?;
    Ok((width, allowance))
}

fn check_width<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    input: &CapturedInput<R>,
    atol: f64,
    rtol: f64,
    name: &str,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if value.lower().numel() == 0 {
        return Ok(());
    }
    let c = a.client();
    let (width, allowance) = width_bounds(a, value, atol, rtol)?;
    let rejected = c.gt(width.upper(), allowance.lower())?;
    if c.cast(&c.any(&rejected, &[], false)?, DType::F64)?
        .item::<f64>()?
        != 0.0
    {
        let rejected_rows = c.nonzero(&rejected)?.flatten()?;
        let row = rejected_rows.narrow(0, 0, 1)?.item::<i64>()?;
        let row = usize::try_from(row).map_err(|_| HydraulicError::Numerical {
            context: format!("Rejected flux row {row} is invalid. Rebuild the query"),
        })?;
        let owner = input.owners.narrow(0, row, 1)?.item::<i64>()?;
        let owner = usize::try_from(owner).map_err(|_| HydraulicError::Numerical {
            context: format!("Rejected flux owner {owner} is invalid. Rebuild the query"),
        })?;
        let read = |tensor: &Tensor<R>| tensor.narrow(0, row, 1)?.item::<f64>();
        let lower = read(value.lower())?;
        let upper = read(value.upper())?;
        let width_upper = read(width.upper())?;
        let allowance_lower = read(allowance.lower())?;
        let left_area = read(&input.left_area)?;
        let left_discharge = read(&input.left_discharge)?;
        let right_area = read(&input.right_area)?;
        let right_discharge = read(&input.right_discharge)?;
        return Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            query: Some(row),
            section_owner: Some(owner),
            context: format!(
                "{name} bounds=[{lower:?}, {upper:?}], width_upper={width_upper:?}, allowance_lower={allowance_lower:?}, atol={atol:?}, rtol={rtol:?}, left_area_m2={left_area:?}, left_discharge_m3_per_s={left_discharge:?}, right_area_m2={right_area:?}, right_discharge_m3_per_s={right_discharge:?}: refine geometry, root, or quadrature bounds"
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::error::{checked, into_public};
    use super::super::sample_merge::Samples;
    use super::*;

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
    fn acceptance_masks_require_both_flux_gates_cpu()
    -> Result<(), super::super::PrismaticRiemannError> {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
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
}
