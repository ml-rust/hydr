use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

use super::{
    config::PrismaticRiemannConfig,
    error::{Arithmetic, HydraulicError, HydraulicResult, PrismaticRiemannRejection},
    sampling::SampledSolution,
};

pub(crate) fn accept<R>(
    a: &mut Arithmetic<'_, R>,
    solution: &SampledSolution<R>,
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
        check_width(a, flux, atol, config.flux_rtol, name)?;
    }
    Ok(())
}

fn check_width<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
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
    let lo = a.point(value.lower())?;
    let hi = a.point(value.upper())?;
    let width = a.sub(&hi, &lo)?;
    let magnitude = c.maximum(&c.abs(value.lower())?, &c.abs(value.upper())?)?;
    let magnitude = a.point(&magnitude)?;
    let relative = a.point(&Tensor::from_slice(&[rtol], &[], c.device())?)?;
    let absolute = a.point(&Tensor::from_slice(&[atol], &[], c.device())?)?;
    let relative = a.mul(&magnitude, &relative)?;
    let allowance = a.add(&absolute, &relative)?;
    let rejected = c.gt(width.upper(), allowance.lower())?;
    if c.cast(&c.any(&rejected, &[], false)?, DType::F64)?
        .item::<f64>()?
        != 0.0
    {
        return Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            query: None,
            section_owner: None,
            context: format!(
                "{name} enclosure exceeds atol={atol:?}, rtol={rtol:?}, with remaining geometry, arithmetic, and quadrature uncertainty. Review hydraulic conditioning"
            ),
        });
    }
    Ok(())
}
