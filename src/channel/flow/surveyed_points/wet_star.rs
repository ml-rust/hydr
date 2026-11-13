use super::{
    wet_depth_types::WetDepthSession,
    wet_star_endpoint,
    wet_star_types::{WetStarDomain, WetStarSideRange, WetStarStatus},
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult, checked},
    section::PhysicalEndpoint,
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
use solvr::{common::arithmetic::Interval, integrate::CertifiedSimpsonAlgorithms};

impl<R> WetDepthSession<'_, '_, R>
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
    /// Enclose each established wet root while retaining depth evidence and continuation unchanged.
    pub(super) fn certify_wet_star(&mut self) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet certificates remain sealed after an earlier numerical error",
            ));
        }
        self.check_domains()?;
        let mut output = Vec::with_capacity(self.wet.domains.len());
        for domain in &self.wet.domains {
            let lower = wet_star_endpoint::evaluate(
                self.sources,
                &domain.parent_rows,
                &domain.query,
                PhysicalEndpoint::Lower,
            )?;
            let upper = wet_star_endpoint::evaluate(
                self.sources,
                &domain.parent_rows,
                &domain.query,
                PhysicalEndpoint::Upper,
            )?;
            let c = self.sources.geometry.client();
            output.push(checked(c, "wet star physical root-state enclosure", |a| {
                let stage = self
                    .sources
                    .select_rows(&domain.parent_rows, &domain.query)?;
                let query = domain.query.with_parameter_bounds(
                    a,
                    &stage.bands,
                    &domain.query.query.parameter,
                )?;
                let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                    let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
                    c.copy_into(&owned, value)?;
                    Ok(owned)
                };
                let mut available = stage.velocity.both_wet.clone();
                for side in 0..2 {
                    available = c.mul(
                        &available,
                        &c.mul(&lower.sides[side].available, &upper.sides[side].available)?,
                    )?;
                }
                let low_left = a.expansion_interval(&lower.sides[0].velocity)?;
                let high_left = a.expansion_interval(&upper.sides[0].velocity)?;
                let low_right = a.expansion_interval(&lower.sides[1].velocity)?;
                let high_right = a.expansion_interval(&upper.sides[1].velocity)?;
                // Left velocity decreases with area. Right velocity increases with area.
                let (left_velocity, left_valid) =
                    ordered_bounds(a, high_left.lower(), low_left.upper(), &available)?;
                let (right_velocity, right_valid) =
                    ordered_bounds(a, low_right.lower(), high_right.upper(), &available)?;
                let side_velocity = [left_velocity, right_velocity];
                let (velocity, intersection_valid) =
                    intersection(a, &side_velocity[0], &side_velocity[1])?;
                let velocity_valid =
                    c.mul(&intersection_valid, &c.mul(&left_valid, &right_valid)?)?;
                let area = a.expansion_from_interval(&query.area)?;
                let mut operands = Vec::with_capacity(2);
                let mut discharges = Vec::with_capacity(2);
                for side in 0..2 {
                    let first = &lower.sides[side];
                    let last = &upper.sides[side];
                    let low_delta = a.expansion_interval(&first.area_difference)?;
                    let high_delta = a.expansion_interval(&last.area_difference)?;
                    let delta = a.bounds(low_delta.lower(), high_delta.upper())?;
                    let area_difference = a.expansion_from_interval(&delta)?;
                    let first_change = a.expansion_interval(&first.velocity_change)?;
                    let last_change = a.expansion_interval(&last.velocity_change)?;
                    let change = a.bounds(
                        &c.minimum(first_change.lower(), last_change.lower())?,
                        &c.maximum(first_change.upper(), last_change.upper())?,
                    )?;
                    let velocity_change = a.expansion_from_interval(&change)?;
                    // Products consume full-domain operands. Discharge has no endpoint-extrema proof.
                    let area_term = a.expansion_mul(&area_difference, &first.source_velocity)?;
                    let velocity_term = a.expansion_mul(&area, &velocity_change)?;
                    let discharge_change = a.expansion_add(&area_term, &velocity_term)?;
                    let discharge = a.expansion_add(&first.source_discharge, &discharge_change)?;
                    discharges.push(a.expansion_interval(&discharge)?);
                    operands.push(WetStarSideRange {
                        area_difference,
                        velocity_change,
                        discharge_change,
                        discharge,
                    });
                }
                let [left_operands, right_operands] = take_pair(operands)?;
                let [left_discharge, right_discharge] = take_pair(discharges)?;
                let (anchored, anchored_valid) =
                    intersection(a, &left_discharge, &right_discharge)?;
                // The physical product also encloses Q*. It limits source-scale cancellation uncertainty.
                let product = a.mul(&query.area, &velocity)?;
                let (discharge, product_valid) = intersection(a, &anchored, &product)?;
                let valid = c.mul(&velocity_valid, &c.mul(&anchored_valid, &product_valid)?)?;
                let certified = c.mul(&available, &valid)?;
                let zero = Tensor::zeros(available.shape(), a.dtype(), c.device())?;
                let status = c.where_cond(
                    &available,
                    &c.where_cond(
                        &valid,
                        &Tensor::full_scalar(
                            available.shape(),
                            DType::I64,
                            WetStarStatus::Certified as i64 as f64,
                            c.device(),
                        )?,
                        &Tensor::full_scalar(
                            available.shape(),
                            DType::I64,
                            WetStarStatus::Unresolved as i64 as f64,
                            c.device(),
                        )?,
                    )?,
                    &Tensor::full_scalar(
                        available.shape(),
                        DType::I64,
                        WetStarStatus::Unavailable as i64 as f64,
                        c.device(),
                    )?,
                )?;
                // Captured uniform identity and strict wet residual uniqueness prove the unchanged root state.
                let uniform = c.mul(&stage.velocity.uniform, &certified)?;
                let source_velocity = a.expansion_interval(&stage.velocity.left)?;
                let source_discharge =
                    a.expansion_interval(&stage.velocity.input().left_discharge)?;
                let velocity = a.bounds(
                    &c.where_cond(
                        &certified,
                        &c.where_cond(&uniform, source_velocity.lower(), velocity.lower())?,
                        &zero,
                    )?,
                    &c.where_cond(
                        &certified,
                        &c.where_cond(&uniform, source_velocity.upper(), velocity.upper())?,
                        &zero,
                    )?,
                )?;
                let discharge = a.bounds(
                    &c.where_cond(
                        &certified,
                        &c.where_cond(&uniform, source_discharge.lower(), discharge.lower())?,
                        &zero,
                    )?,
                    &c.where_cond(
                        &certified,
                        &c.where_cond(&uniform, source_discharge.upper(), discharge.upper())?,
                        &zero,
                    )?,
                )?;
                Ok(WetStarDomain {
                    invocation_owner: copy(&domain.invocation_owner)?,
                    parent_rows: copy(&domain.parent_rows)?,
                    query,
                    endpoints: [lower, upper],
                    side_operands: [left_operands, right_operands],
                    side_velocity,
                    side_discharge: [left_discharge, right_discharge],
                    velocity,
                    discharge,
                    status,
                })
            })?);
        }
        Ok(output)
    }
}

fn take_pair<T>(values: Vec<T>) -> HydraulicResult<[T; 2]> {
    values.try_into().map_err(|_: Vec<T>| {
        crate::channel::flow::surveyed::error::HydraulicError::Numerical {
            context: "Wet star side count differs from two. Rebuild the paired certificate".into(),
        }
    })
}

fn ordered_bounds<R>(
    a: &mut Arithmetic<'_, R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    available: &Tensor<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
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
    let valid = c.le(lower, upper)?;
    let active = c.mul(available, &valid)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&active, lower, &zero)?,
            &c.where_cond(&active, upper, &zero)?,
        )?,
        valid,
    ))
}

pub(super) fn intersection<R>(
    a: &mut Arithmetic<'_, R>,
    first: &Interval<R>,
    second: &Interval<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
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
    let lower = c.maximum(first.lower(), second.lower())?;
    let upper = c.minimum(first.upper(), second.upper())?;
    let valid = c.le(&lower, &upper)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&valid, &lower, &zero)?,
            &c.where_cond(&valid, &upper, &zero)?,
        )?,
        valid,
    ))
}

#[cfg(test)]
#[path = "wet_star_tests.rs"]
mod tests;
