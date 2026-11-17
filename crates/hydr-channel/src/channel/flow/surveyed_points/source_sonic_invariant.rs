use super::{
    potential_absolute::endpoint_potential, potential_increment::endpoint_increment,
    potential_range::PotentialContext, source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::PhysicalEndpoint,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

/// The invariant is evaluated at the supplied endpoint, without imposing the sonic equation there.
pub(super) fn endpoint<R>(
    context: &PotentialContext<'_, R>,
    side: SonicSide,
    epsilon: &CertifiedExpansion<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let potential = endpoint_potential(context, PhysicalEndpoint::Lower)?;
    let increment = endpoint_increment(context, PhysicalEndpoint::Lower)?;
    let c = context.geometry.client();
    checked::<R, _>(c, "source sonic retained trace invariant", |a| {
        if side == SonicSide::Left {
            let zero = a.expansion_point(&Tensor::zeros(
                epsilon.high().shape(),
                a.dtype(),
                c.device(),
            )?)?;
            Ok((
                a.expansion_sub(epsilon, &potential.retained)?,
                a.expansion_sub(&zero, &increment.retained)?,
            ))
        } else {
            Ok((
                a.expansion_sub(&potential.retained, epsilon)?,
                increment.retained,
            ))
        }
    })
}
