//! Complete persistent relay and bounded last-transition tensors.

use super::{CpuControlContext, CpuControlSample, CpuControlTimer};
use crate::cpu::CpuTensor;
use crate::structures::{ControlLatch, ControlState, StructureError};

#[derive(Clone, Debug)]
pub(crate) struct CpuControlTransition {
    pub valid: CpuTensor,
    pub decision: CpuTensor,
    pub from: CpuTensor,
    pub sample: CpuControlSample,
    pub timer: CpuControlTimer,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuControlState {
    pub high: CpuTensor,
    pub pending: CpuTensor,
    pub since: CpuTensor,
    pub sample: CpuControlSample,
    pub transition_valid: CpuTensor,
    pub transition_time: CpuTensor,
    pub count: CpuTensor,
    pub last_transition: CpuControlTransition,
}
impl CpuControlState {
    pub(crate) fn initial(context: &CpuControlContext) -> Result<Self, StructureError> {
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
            sample: CpuControlSample::from_optional(
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
            last_transition: CpuControlTransition {
                valid: context.zero.clone(),
                decision: context.zero.clone(),
                from: context.zero.clone(),
                sample: CpuControlSample::zero(context),
                timer: CpuControlTimer::zero(context),
            },
        };
        crate::numerics::structures::control::check_initial(context, &state)?;
        Ok(state)
    }
    pub(crate) fn from_values(
        context: &CpuControlContext,
        states: &[ControlState],
    ) -> Result<Self, StructureError> {
        let values =
            |f: fn(&ControlState) -> f64| context.tensor(&states.iter().map(f).collect::<Vec<_>>());
        let sample = CpuControlSample::from_optional(
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
        let last_transition = CpuControlTransition {
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
            sample: CpuControlSample::from_optional(context, &transition_samples)?,
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
