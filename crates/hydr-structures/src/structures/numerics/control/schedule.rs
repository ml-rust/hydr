//! Exact original schedule and declared initial-observation cadence.

use crate::structures::tensor::control::TensorControlContext;
use crate::structures::{StructureError, StructureInputErrorKind};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ScalarOps, TwoSumOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(crate) fn check_schedule<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let schedule = &context.schedule;
    let mut starts = Vec::with_capacity(schedule.decision_times_s.len() + 1);
    let mut ends = Vec::with_capacity(starts.capacity());
    let mut previous = schedule.initial_time_s;
    for &time in &schedule.decision_times_s {
        starts.push(previous);
        ends.push(time);
        previous = time;
    }
    starts.push(previous);
    ends.push(schedule.end_s);
    check(context, &starts, &ends, false)?;
    let first = schedule
        .decision_times_s
        .first()
        .copied()
        .unwrap_or(schedule.end_s);
    let starts = context
        .initial_states
        .iter()
        .map(|s| {
            s.antecedent_sample
                .as_ref()
                .map_or(schedule.initial_time_s, |sample| sample.observed_at_s)
        })
        .collect::<Vec<_>>();
    check(context, &starts, &vec![first; starts.len()], true)
}
fn check<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    starts: &[f64],
    ends: &[f64],
    per_controller: bool,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.structures.client;
    let starts = context.tensor(starts)?;
    let ends = context.tensor(ends)?;
    let zero = c.mul_scalar(&starts, 0.0)?;
    let maximum = c.add_scalar(&zero, context.timing.maximum_decision_gap_s)?;
    let high = c.sub(&ends, &starts)?;
    let reject = |mask: &Tensor<R>| -> Result<(), StructureError> {
        if per_controller {
            context.reject(
                mask,
                Some(StructureInputErrorKind::Definition),
                "initial observation gap exceeds declared finite decision cadence",
            )
        } else if context.structures.any(mask)? {
            Err(StructureError::input(
                StructureInputErrorKind::Definition,
                "original schedule gap exceeds declared finite decision cadence",
            ))
        } else {
            Ok(())
        }
    };
    let finite = super::timers::finite(context, &high)?;
    reject(&c.add_scalar(&c.neg(&finite)?, 1.0)?)?;
    let (high, low) = c.two_sum(&ends, &c.neg(&starts)?)?;
    reject(&c.add_scalar(&c.neg(&super::timers::finite(context, &low)?)?, 1.0)?)?;
    reject(&c.maximum(
        &c.gt(&high, &maximum)?,
        &c.mul(&c.eq(&high, &maximum)?, &c.gt(&low, &zero)?)?,
    )?)
}
