use super::{
    error::{Arithmetic, HydraulicResult},
    pressure::pressure,
    types::SideState,
};
use crate::channel::section::Constitutive;
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

pub(crate) struct Candidate<R: Runtime<DType = DType>> {
    pub(crate) possible: Tensor<R>,
    pub(crate) area: Interval<R>,
    pub(crate) discharge: Interval<R>,
    pub(crate) depth: Interval<R>,
    pub(crate) water_flux: Interval<R>,
    pub(crate) momentum_flux: Interval<R>,
    pub(crate) dry_possible: Tensor<R>,
    pub(crate) wet_possible: Tensor<R>,
}

pub(crate) fn from_state<R>(
    a: &mut Arithmetic<'_, R>,
    state: &Constitutive<R>,
    velocity: &Interval<R>,
    discharge: Option<&Tensor<R>>,
    possible: Tensor<R>,
    gravity: f64,
) -> HydraulicResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let q = match discharge {
        Some(q) => a.point(q)?,
        None => a.mul(&state.area, velocity)?,
    };
    let advective = a.mul(&q, velocity)?;
    let hydrostatic = pressure(a, state, gravity)?;
    let momentum = a.add(&advective, &hydrostatic)?;
    Ok(Candidate {
        possible,
        area: a.bounds(state.area.lower(), state.area.upper())?,
        discharge: a.bounds(q.lower(), q.upper())?,
        depth: a.bounds(state.depth.lower(), state.depth.upper())?,
        water_flux: q,
        momentum_flux: momentum,
        dry_possible: state.dry_possible.clone(),
        wet_possible: state.wet_possible.clone(),
    })
}

pub(crate) fn exterior<R>(
    a: &mut Arithmetic<'_, R>,
    side: &SideState<R>,
    possible: Tensor<R>,
    gravity: f64,
) -> HydraulicResult<Candidate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    from_state(
        a,
        &side.geometry,
        &side.velocity,
        Some(&side.original_discharge),
        possible,
        gravity,
    )
}

pub(crate) fn dry<R>(
    a: &mut Arithmetic<'_, R>,
    possible: Tensor<R>,
) -> HydraulicResult<Candidate<R>>
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
    let zero = Tensor::zeros(possible.shape(), DType::F64, c.device())?;
    Ok(Candidate {
        area: a.point(&zero)?,
        discharge: a.point(&zero)?,
        depth: a.point(&zero)?,
        water_flux: a.point(&zero)?,
        momentum_flux: a.point(&zero)?,
        dry_possible: Tensor::ones(possible.shape(), DType::F64, c.device())?,
        wet_possible: zero,
        possible,
    })
}
