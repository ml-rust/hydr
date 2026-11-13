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
#[path = "pointed_depth_identity_tests.rs"]
mod tests;
