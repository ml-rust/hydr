//! Exact original sample identity, chronology, and causal availability checks.

use super::{CpuControlContext, CpuControlSample};
use crate::structures::{
    ControlInitialState, ControlSample, OperatingControlDefinition, StructureError,
    StructureInputErrorKind,
};

pub(crate) fn check_sample(
    definition: &OperatingControlDefinition,
    sample: &ControlSample,
) -> Result<(), StructureError> {
    if sample.controller_id != definition.controller_id
        || sample.signal_id != definition.signal.signal_id
        || sample.support_id != definition.signal.support_id
    {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "control sample identity differs from original controller/signal/support",
        ));
    }
    if sample.datum != definition.signal.datum {
        return Err(StructureError::input(
            StructureInputErrorKind::Datum,
            "control sample datum differs from declared signal datum",
        ));
    }
    if !sample.observed_at_s.is_finite()
        || !sample.available_at_s.is_finite()
        || !sample.stage_m.is_finite()
    {
        return Err(StructureError::input(
            StructureInputErrorKind::NonfiniteInput,
            "control sample requires finite stage, observation, and availability times",
        ));
    }
    if sample.available_at_s > sample.observed_at_s {
        return Err(StructureError::input(
            StructureInputErrorKind::UnsupportedRegime,
            "control sample availability follows its original observation: supply causal input",
        ));
    }
    Ok(())
}
pub(super) fn check_initial_static(
    definition: &OperatingControlDefinition,
    state: &ControlInitialState,
    time: f64,
) -> Result<(), StructureError> {
    let error = || {
        StructureError::input(
            StructureInputErrorKind::Definition,
            "initial control antecedent chronology or pending qualification is inconsistent",
        )
    };
    if let Some(sample) = &state.antecedent_sample {
        check_sample(definition, sample)?;
        if sample.observed_at_s > time {
            return Err(error());
        }
    }
    if let Some(transition) = state.last_transition_at_s
        && (!transition.is_finite()
            || transition > time
            || state
                .antecedent_sample
                .as_ref()
                .is_some_and(|sample| transition > sample.observed_at_s))
    {
        return Err(error());
    }
    if let Some(pending) = state.pending {
        let sample = state.antecedent_sample.as_ref().ok_or_else(error)?;
        if pending.target == state.latch
            || !pending.since_s.is_finite()
            || pending.since_s > sample.observed_at_s
            || state
                .last_transition_at_s
                .is_some_and(|transition| pending.since_s <= transition)
        {
            return Err(error());
        }
    }
    Ok(())
}
pub(crate) fn decision_samples(
    context: &CpuControlContext,
    samples: &[ControlSample],
    time: f64,
) -> Result<CpuControlSample, StructureError> {
    if samples.len() != context.definitions.len() {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            format!(
                "control sample count={},expected={}: supply complete ordered decision inputs",
                samples.len(),
                context.definitions.len()
            ),
        ));
    }
    for ((definition, sample), &slot) in context.definitions.iter().zip(samples).zip(&context.slots)
    {
        check_sample(definition, sample).map_err(|error| error.at(slot))?;
        if sample.observed_at_s != time {
            return Err(StructureError::input(StructureInputErrorKind::UnsupportedRegime,format!("control observation={},decision={time}: supply original sample at this decision",sample.observed_at_s)).at(slot));
        }
    }
    CpuControlSample::from_optional(context, &samples.iter().map(Some).collect::<Vec<_>>())
}
