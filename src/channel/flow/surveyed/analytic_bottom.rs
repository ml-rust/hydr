use super::equation_ops::{choose, scalar, scale};
use super::error::{Arithmetic, HydraulicResult};
use super::types::{SideState, WaveSide};
use crate::channel::section::GatheredBands;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) struct AnalyticBottom<R: Runtime<DType = DType>> {
    pub(crate) eligible: Tensor<R>,
    pub(crate) potential_squared: Interval<R>,
}

pub(crate) fn kinds<R>(
    a: &mut Arithmetic<'_, R>,
    bottom: &Tensor<R>,
    pointed: &Tensor<R>,
    width: &Interval<R>,
    slope: &Interval<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
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
    let rectangular = c.mul(
        bottom,
        &c.mul(
            &c.gt(width.lower(), zero.upper())?,
            &c.mul(
                &c.eq(slope.lower(), zero.lower())?,
                &c.eq(slope.upper(), zero.upper())?,
            )?,
        )?,
    )?;
    let pointed = c.mul(
        bottom,
        &c.mul(
            pointed,
            &c.mul(
                &c.gt(slope.lower(), zero.upper())?,
                &c.mul(
                    &c.eq(width.lower(), zero.lower())?,
                    &c.eq(width.upper(), zero.upper())?,
                )?,
            )?,
        )?,
    )?;
    Ok((rectangular, pointed))
}

pub(crate) fn potential_squared<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    side: &SideState<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticBottom<R>>
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
    let column = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
    };
    let width = a.bounds(
        &column(bands.start_width.lower())?,
        &column(bands.start_width.upper())?,
    )?;
    let slope = a.bounds(&column(bands.slope.lower())?, &column(bands.slope.upper())?)?;
    let (rectangular, pointed) = kinds(
        a,
        &column(&bands.valid)?,
        &column(&bands.pointed)?,
        &width,
        &slope,
    )?;
    let supported = c.le(&side.original_area, &column(bands.end_area.lower())?)?;
    let rectangular = c.mul(&rectangular, &supported)?;
    let pointed = c.mul(&pointed, &supported)?;
    let eligible = c.maximum(&rectangular, &pointed)?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    // The original conserved area determines rectangular potential without recovering a square root.
    let area = a.point(&side.original_area)?;
    let area = choose(a, &rectangular, &area, &zero)?;
    let denominator = choose(a, &rectangular, &width, &one)?;
    let numerator = scale(a, &area, 4.0)?;
    let numerator = scale(a, &numerator, gravity)?;
    let rectangular_squared = a.div_positive_interval(&numerator, &denominator)?;
    let depth = choose(a, &pointed, &side.geometry.depth, &zero)?;
    let pointed_squared = scale(a, &depth, 8.0)?;
    let pointed_squared = scale(a, &pointed_squared, gravity)?;
    let potential_squared = a.add(&rectangular_squared, &pointed_squared)?;
    Ok(AnalyticBottom {
        eligible,
        potential_squared,
    })
}

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
