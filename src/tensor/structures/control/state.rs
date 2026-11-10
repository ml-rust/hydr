//! Complete persistent relay and bounded last-transition tensors.

use super::{TensorControlContext, TensorControlSample, TensorControlTimer};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::structures::{ControlLatch, ControlState, StructureError};

#[derive(Clone, Debug)]
pub(crate) struct TensorControlTransition<R: Runtime<DType = DType>> {
    pub valid: Tensor<R>,
    pub decision: Tensor<R>,
    pub from: Tensor<R>,
    pub sample: TensorControlSample<R>,
    pub timer: TensorControlTimer<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorControlState<R: Runtime<DType = DType>> {
    pub high: Tensor<R>,
    pub pending: Tensor<R>,
    pub since: Tensor<R>,
    pub sample: TensorControlSample<R>,
    pub transition_valid: Tensor<R>,
    pub transition_time: Tensor<R>,
    pub count: Tensor<R>,
    pub last_transition: TensorControlTransition<R>,
}
impl<R: Runtime<DType = DType>> TensorControlState<R> {
    pub(crate) fn initial(context: &TensorControlContext<R>) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let initial = &context.initial_states;
        let state = Self {
            high: context.tensor(&initial.iter().map(|s| latch(s.latch)).collect::<Vec<_>>())?,
            pending: context.tensor(
                &initial
                    .iter()
                    .map(|s| if s.pending.is_some() { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )?,
            since: context.tensor(
                &initial
                    .iter()
                    .map(|s| s.pending.map_or(0.0, |p| p.since_s))
                    .collect::<Vec<_>>(),
            )?,
            sample: TensorControlSample::from_optional(
                context,
                &initial
                    .iter()
                    .map(|s| s.antecedent_sample.as_ref())
                    .collect::<Vec<_>>(),
            )?,
            transition_valid: context.tensor(
                &initial
                    .iter()
                    .map(|s| {
                        if s.last_transition_at_s.is_some() {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>(),
            )?,
            transition_time: context.tensor(
                &initial
                    .iter()
                    .map(|s| s.last_transition_at_s.unwrap_or(0.0))
                    .collect::<Vec<_>>(),
            )?,
            count: context.zero.clone(),
            last_transition: TensorControlTransition {
                valid: context.zero.clone(),
                decision: context.zero.clone(),
                from: context.zero.clone(),
                sample: TensorControlSample::zero(context),
                timer: TensorControlTimer::zero(context),
            },
        };
        crate::numerics::structures::control::check_initial(context, &state)?;
        Ok(state)
    }
    pub(crate) fn from_values(
        context: &TensorControlContext<R>,
        states: &[ControlState],
    ) -> Result<Self, StructureError> {
        let values =
            |f: fn(&ControlState) -> f64| context.tensor(&states.iter().map(f).collect::<Vec<_>>());
        let sample = TensorControlSample::from_optional(
            context,
            &states
                .iter()
                .map(|s| s.last_sample.as_ref())
                .collect::<Vec<_>>(),
        )?;
        let transition_samples = states
            .iter()
            .map(|s| s.last_transition.as_ref().map(|r| &r.sample))
            .collect::<Vec<_>>();
        let last_transition = TensorControlTransition {
            valid: values(|s| {
                if s.last_transition.is_some() {
                    1.0
                } else {
                    0.0
                }
            })?,
            decision: values(|s| {
                s.last_transition
                    .as_ref()
                    .map_or(0.0, |r| r.decision_index as f64)
            })?,
            from: values(|s| s.last_transition.as_ref().map_or(0.0, |r| latch(r.from)))?,
            sample: TensorControlSample::from_optional(context, &transition_samples)?,
            timer: super::import::timers(
                context,
                &states
                    .iter()
                    .map(|s| s.last_transition.as_ref().map(|r| r.timer.clone()))
                    .collect::<Vec<_>>(),
            )?,
        };
        Ok(Self {
            high: values(|s| latch(s.latch))?,
            pending: values(|s| if s.pending.is_some() { 1.0 } else { 0.0 })?,
            since: values(|s| s.pending.map_or(0.0, |p| p.since_s))?,
            sample,
            transition_valid: values(|s| {
                if s.last_transition_at_s.is_some() {
                    1.0
                } else {
                    0.0
                }
            })?,
            transition_time: values(|s| s.last_transition_at_s.unwrap_or(0.0))?,
            count: values(|s| s.transition_count as f64)?,
            last_transition,
        })
    }
}
pub(crate) fn latch(value: ControlLatch) -> f64 {
    match value {
        ControlLatch::Low => 0.0,
        ControlLatch::High => 1.0,
    }
}
