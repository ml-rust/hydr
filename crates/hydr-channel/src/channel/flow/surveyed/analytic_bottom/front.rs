use super::AnalyticBottom;
use crate::channel::flow::surveyed::equation_ops::{choose, scalar, scale};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use crate::channel::flow::surveyed::types::{SideState, WaveSide};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(crate) fn vacuum<R>(
    a: &mut Arithmetic<'_, R>,
    left: &SideState<R>,
    right: &SideState<R>,
    left_proof: &AnalyticBottom<R>,
    right_proof: &AnalyticBottom<R>,
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
    let zero = scalar(a, 0.0)?;
    let eligible = c.mul(&left_proof.eligible, &right_proof.eligible)?;
    let difference = a.sub(&right.velocity, &left.velocity)?;
    let difference = a.point(difference.lower())?;
    let difference = choose(a, &eligible, &difference, &zero)?;
    let left_squared = a.point(left_proof.potential_squared.upper())?;
    let left_squared = choose(a, &eligible, &left_squared, &zero)?;
    let right_squared = a.point(right_proof.potential_squared.upper())?;
    let right_squared = choose(a, &eligible, &right_squared, &zero)?;
    let square = a.mul(&difference, &difference)?;
    let remainder = a.sub(&square, &left_squared)?;
    let remainder = a.sub(&remainder, &right_squared)?;
    // Only singleton conservative endpoints enter squaring, preserving exact equality certificates.
    let remainder_lower = a.point(remainder.lower())?;
    let square_remainder = a.mul(&remainder_lower, &remainder_lower)?;
    let product = a.mul(&left_squared, &right_squared)?;
    let product = scale(a, &product, 4.0)?;
    let discriminant = a.sub(&square_remainder, &product)?;
    let nonnegative = c.mul(
        &c.ge(difference.lower(), zero.upper())?,
        &c.mul(
            &c.ge(remainder.lower(), zero.upper())?,
            &c.ge(discriminant.lower(), zero.upper())?,
        )?,
    )?;
    Ok(c.mul(&eligible, &nonnegative)?)
}

pub(crate) fn dry_front<R>(
    a: &mut Arithmetic<'_, R>,
    side: &SideState<R>,
    wave_side: WaveSide,
    proof: &AnalyticBottom<R>,
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
    let zero = scalar(a, 0.0)?;
    let (endpoint, correct_sign) = match wave_side {
        WaveSide::Left => (
            side.velocity.upper(),
            c.le(side.velocity.upper(), zero.lower())?,
        ),
        WaveSide::Right => (
            side.velocity.lower(),
            c.ge(side.velocity.lower(), zero.upper())?,
        ),
    };
    let endpoint = a.point(endpoint)?;
    let endpoint = choose(a, &proof.eligible, &endpoint, &zero)?;
    let square = a.mul(&endpoint, &endpoint)?;
    let magnitude = c.ge(square.lower(), proof.potential_squared.upper())?;
    Ok(c.mul(&proof.eligible, &c.mul(&correct_sign, &magnitude)?)?)
}
