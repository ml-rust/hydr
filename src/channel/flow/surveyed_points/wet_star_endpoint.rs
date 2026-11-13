use super::{
    paired_endpoint,
    paired_sources::PairedSources,
    wet_star_types::{WetStarEndpoint, WetStarSideEndpoint},
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult, checked},
    section::{PhysicalEndpoint, SupportedAreaQuery, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn evaluate<R>(
    sources: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<WetStarEndpoint<R>>
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
    let terms = paired_endpoint::prepare(sources, rows, query, endpoint)?;
    let c = sources.geometry.client();
    checked(c, "wet star anchored endpoint", |a| {
        let stage = &terms.stage;
        let area = terms
            .target_query
            .retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
        let zero_tensor = Tensor::zeros(area.high().shape(), a.dtype(), c.device())?;
        let input = stage.velocity.input();
        let mut sides = Vec::with_capacity(2);
        for side in 0..2 {
            let source_velocity = if side == 0 {
                &stage.velocity.left
            } else {
                &stage.velocity.right
            };
            let source_discharge = if side == 0 {
                &input.left_discharge
            } else {
                &input.right_discharge
            };
            let source_area = if side == 0 {
                &input.left_area
            } else {
                &input.right_area
            };
            let target = &terms.target_value.retained;
            let target = a.expansion_bounds(
                &c.where_cond(&terms.far[side], target.high(), &zero_tensor)?,
                &c.where_cond(&terms.far[side], target.low(), &zero_tensor)?,
                &c.where_cond(&terms.far[side], target.abs_error(), &zero_tensor)?,
            )?;
            let source = &terms.far_sources[side].retained;
            let local = &terms.local[side].retained;
            // Compose the source invariant before adding tiny far-target terms.
            let (velocity, velocity_change) = if side == 0 {
                let invariant = a.expansion_add(source_velocity, source)?;
                let invariant = a.expansion_sub(&invariant, local)?;
                let velocity = a.expansion_sub(&invariant, &target)?;
                let change = a.expansion_sub(source, local)?;
                (velocity, a.expansion_sub(&change, &target)?)
            } else {
                let invariant = a.expansion_sub(source_velocity, source)?;
                let invariant = a.expansion_add(&invariant, local)?;
                let velocity = a.expansion_add(&invariant, &target)?;
                let change = a.expansion_sub(local, source)?;
                (velocity, a.expansion_add(&change, &target)?)
            };
            let area_difference = source_difference(
                a,
                &stage.bands,
                &terms.target_query,
                &stage.sources[side],
                &stage.boundaries,
            )?
            .lower;
            let unchanged = c.mul(
                &exact_zero(a, &area_difference)?,
                &exact_zero(a, &velocity_change)?,
            )?;
            let area_term = a.expansion_mul(&area_difference, source_velocity)?;
            let velocity_term = a.expansion_mul(&area, &velocity_change)?;
            let discharge_change = a.expansion_add(&area_term, &velocity_term)?;
            let discharge = a.expansion_add(source_discharge, &discharge_change)?;
            // Complete expression identity preserves captured discharge without quotient reconstruction.
            let discharge = a.expansion_bounds(
                &c.where_cond(&unchanged, source_discharge.high(), discharge.high())?,
                &c.where_cond(&unchanged, source_discharge.low(), discharge.low())?,
                &c.where_cond(
                    &unchanged,
                    source_discharge.abs_error(),
                    discharge.abs_error(),
                )?,
            )?;
            sides.push(WetStarSideEndpoint {
                source_area: a.expansion_bounds(
                    source_area.high(),
                    source_area.low(),
                    source_area.abs_error(),
                )?,
                source_velocity: a.expansion_bounds(
                    source_velocity.high(),
                    source_velocity.low(),
                    source_velocity.abs_error(),
                )?,
                source_discharge: a.expansion_bounds(
                    source_discharge.high(),
                    source_discharge.low(),
                    source_discharge.abs_error(),
                )?,
                area_difference,
                velocity_change,
                velocity,
                discharge_change,
                discharge,
                unchanged,
                available: c.mul(
                    &stage.velocity.both_wet,
                    &c.maximum(&terms.far[side], &terms.local[side].available)?,
                )?,
            });
        }
        let mut iter = sides.into_iter();
        let left = iter.next().ok_or_else(|| {
            crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                context: "Wet star left endpoint diagnostic is absent. Rebuild the paired endpoint"
                    .into(),
            }
        })?;
        let right = iter.next().ok_or_else(|| {
            crate::channel::flow::surveyed::error::HydraulicError::Numerical {
                context:
                    "Wet star right endpoint diagnostic is absent. Rebuild the paired endpoint"
                        .into(),
            }
        })?;
        Ok(WetStarEndpoint {
            area,
            sides: [left, right],
        })
    })
}

fn exact_zero<R>(a: &Arithmetic<'_, R>, value: &CertifiedExpansion<R>) -> HydraulicResult<Tensor<R>>
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
    let zero = Tensor::zeros(value.high().shape(), a.dtype(), c.device())?;
    Ok(c.mul(
        &c.eq(value.high(), &zero)?,
        &c.mul(&c.eq(value.low(), &zero)?, &c.eq(value.abs_error(), &zero)?)?,
    )?)
}
