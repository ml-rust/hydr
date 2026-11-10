//! Static ordered binding and original command applicability admission.

use crate::cpu::structures::CpuStructureContext;
use crate::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, OperatingControlDefinition,
    SampledControlTiming, StructureError, StructureInputErrorKind, StructureLaw,
};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn check(
    context: &CpuStructureContext,
    definitions: &[OperatingControlDefinition],
    initial: &[ControlInitialState],
    schedule: &ControlDecisionSchedule,
    timing: &SampledControlTiming,
) -> Result<Vec<usize>, StructureError> {
    let invalid =
        |message: &str| StructureError::input(StructureInputErrorKind::Definition, message);
    if definitions.is_empty() {
        return Err(invalid(
            "control definitions require a nonempty ordered subset",
        ));
    }
    if initial.len() != definitions.len() {
        return Err(StructureError::input(
            StructureInputErrorKind::Identity,
            "control initial states do not match the complete ordered controller subset",
        ));
    }
    if definitions.len() as u128 > super::context::MAX_EXACT_COUNT as u128
        || schedule.decision_times_s.len() as u128 > super::context::MAX_EXACT_COUNT as u128
    {
        return Err(invalid(
            "control count or decision index exceeds exact F64 integer capacity",
        ));
    }
    crate::structures::definition::text(&timing.sampled_applicability, "sampled_applicability")?;
    if !timing.maximum_decision_gap_s.is_finite()
        || timing.maximum_decision_gap_s <= 0.0
        || !timing.maximum_timer_lateness_s.is_finite()
        || timing.maximum_timer_lateness_s < 0.0
    {
        return Err(invalid(
            "control timing requires finite positive cadence and finite nonnegative lateness",
        ));
    }
    if !schedule.initial_time_s.is_finite()
        || schedule.initial_time_s < 0.0
        || !schedule.end_s.is_finite()
        || schedule.end_s <= schedule.initial_time_s
    {
        return Err(invalid(
            "control schedule requires finite nonnegative start and later finite horizon",
        ));
    }
    let mut previous = schedule.initial_time_s;
    for &time in &schedule.decision_times_s {
        if !time.is_finite() || time <= previous || time > schedule.end_s {
            return Err(invalid(
                "control decisions require strict original order inside declared finite support",
            ));
        }
        previous = time;
    }
    let lookup = context
        .elements
        .iter()
        .enumerate()
        .map(|(i, e)| (e.id.as_str(), i))
        .collect::<BTreeMap<_, _>>();
    let mut ids = BTreeSet::new();
    let mut slots = Vec::with_capacity(definitions.len());
    let mut previous_slot = None;
    for (definition, state) in definitions.iter().zip(initial) {
        for (value, name) in [
            (&definition.controller_id, "controller_id"),
            (&definition.structure_id, "structure_id"),
            (&definition.signal.signal_id, "signal_id"),
            (&definition.signal.support_id, "support_id"),
            (&definition.signal.datum, "signal datum"),
            (&definition.provenance, "control provenance"),
        ] {
            crate::structures::definition::text(value, name)?;
        }
        let slot = *lookup
            .get(definition.structure_id.as_str())
            .ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Identity,
                    "control structure is absent from actual batch",
                )
            })?;
        if !ids.insert(&definition.controller_id)
            || previous_slot.is_some_and(|before| slot <= before)
            || state.controller_id != definition.controller_id
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "control IDs, initial IDs, or controlled subset order differ",
            )
            .at(slot));
        }
        previous_slot = Some(slot);
        if definition.signal.datum != context.elements[slot].datum {
            return Err(StructureError::input(
                StructureInputErrorKind::Datum,
                "control signal datum differs from actual structure datum",
            )
            .at(slot));
        }
        if !definition.lower_stage_m.is_finite()
            || !definition.upper_stage_m.is_finite()
            || definition.lower_stage_m >= definition.upper_stage_m
            || [
                definition.high_qualification_delay_s,
                definition.low_qualification_delay_s,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err(invalid(
                "control thresholds require finite increasing stages and finite nonnegative delays",
            )
            .at(slot));
        }
        for command in [definition.low_command, definition.high_command] {
            check_command(&context.elements[slot].law, command).map_err(|error| error.at(slot))?;
        }
        super::input::check_initial_static(definition, state, schedule.initial_time_s)
            .map_err(|error| error.at(slot))?;
        slots.push(slot);
    }
    Ok(slots)
}
fn check_command(law: &StructureLaw, command: ControlCommand) -> Result<(), StructureError> {
    let admitted = match (law, command) {
        (StructureLaw::Pump(_), ControlCommand::PumpStoppedIsolated) => true,
        (StructureLaw::Pump(parameters), ControlCommand::PumpRunning { speed_ratio }) => {
            let (lo, hi) = parameters.speed_policy().range();
            speed_ratio.is_finite() && speed_ratio > 0.0 && speed_ratio >= lo && speed_ratio <= hi
        }
        (
            StructureLaw::HydrostaticRectangularGate(parameters),
            ControlCommand::GateTarget { opening_m },
        ) => {
            opening_m.is_finite() && opening_m >= 0.0 && opening_m <= parameters.maximum_opening_m()
        }
        _ => false,
    };
    if !admitted {
        return Err(StructureError::input(
            StructureInputErrorKind::Definition,
            format!(
                "control command={command:?}: supply a law-compatible command within actual device bounds"
            ),
        ));
    }
    Ok(())
}
