use super::band_lanes::IntegrationLanes;
use super::depth_potential::DepthPotential;
use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn panel_fourth_bound<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &IntegrationLanes<R>,
    owner: &Tensor<R>,
    x: &Interval<R>,
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
    let fourth = fourth_derivative(a, lanes, owner, x)?;
    let c = a.client();
    Ok(c.maximum(&c.abs(fourth.lower())?, &c.abs(fourth.upper())?)?)
}

pub(crate) fn integrand<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &IntegrationLanes<R>,
    owner: &Tensor<R>,
    x: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    DepthPotential::from_lanes(lanes, owner, x).value(a)
}

pub(crate) fn fourth_derivative<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &IntegrationLanes<R>,
    owner: &Tensor<R>,
    x: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    DepthPotential::from_lanes(lanes, owner, x).fourth_derivative(a)
}
