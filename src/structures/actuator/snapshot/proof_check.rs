//! Bounded Numr recomputation of retained completion and target-change proofs.

use super::super::*;
use super::compatibility::{as_definition, require};
use super::receipt_bits::ReceiptBits;
use crate::structures::numerics::actuator as math;
use crate::structures::tensor::actuator::{
    TensorActuatorContext, TensorActuatorState, import_values, record::TensorRecord,
};
use crate::structures::{StructureError, StructureLaw};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

fn hold<R: Runtime<DType = DType>>(c: &TensorActuatorContext<R>, i: usize) -> GateActuatorState {
    GateActuatorState {
        actuator_id: c.definitions[i].actuator_id.clone(),
        anchor: GateActuatorAnchor {
            time_s: c.schedule.initial_time_s,
            position: GateActuatorPosition {
                opening_m: 0.0,
                lower_opening_m: 0.0,
                upper_opening_m: 0.0,
                absolute_error_m: 0.0,
            },
        },
        target_opening_m: 0.0,
        position: GateActuatorPosition {
            opening_m: 0.0,
            lower_opening_m: 0.0,
            upper_opening_m: 0.0,
            absolute_error_m: 0.0,
        },
        phase: GateActuatorPhase::Complete,
        arrival: None,
        target_change_count: 0,
        episode_count: 0,
        arrival_count: 0,
        active_episode_index: None,
        latest_update: None,
        last_target_change: None,
        last_completion: None,
    }
}
fn arrival<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    state: &mut TensorActuatorState<R>,
) -> Result<(), StructureError>
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
    state.arrival =
        math::arrival::arrival(c, &state.anchor, &state.target).map_err(as_definition)?;
    Ok(())
}
pub(super) fn completions<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    receipts: &[Option<&GateActuatorCompletionReceipt>],
) -> Result<(), StructureError>
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
    require(
        receipts.len() == c.definitions.len(),
        "retained actuator completion vector has incompatible length",
    )?;
    let original = super::super::projection::states(c, &math::initial(c).map_err(as_definition)?)?;
    let mut times = Vec::with_capacity(receipts.len());
    let mut rows = Vec::with_capacity(receipts.len());
    for (i, receipt) in receipts.iter().enumerate() {
        let mut row = hold(c, i);
        if let Some(r) = receipt {
            let maximum = match &c.structures.elements[c.slots[i]].law {
                StructureLaw::HydrostaticRectangularGate(gate) => gate.maximum_opening_m(),
                _ => {
                    return Err(StructureError::numerical(
                        "actuator context contains a non-gate",
                    ));
                }
            };
            require(
                r.original_arrival.target_opening_m.is_finite()
                    && r.original_arrival.target_opening_m >= 0.0
                    && r.original_arrival.target_opening_m <= maximum,
                "retained actuator completion target lies outside its actual gate range",
            )?;
            require(
                r.episode_index.checked_add(1).is_some_and(|episodes| {
                    r.target_generation
                        .checked_add(original[i].episode_count)
                        .is_some_and(|capacity| episodes <= capacity)
                }) && (r.target_generation == 0 || r.episode_index >= original[i].episode_count),
                "retained actuator completion episode exceeds its target-generation capacity",
            )?;
            if r.target_generation > 0 {
                require(
                    c.schedule
                        .target_times_s
                        .iter()
                        .enumerate()
                        .any(|(index, time)| {
                            time.same_bits(&r.original_arrival.anchor.time_s)
                                && index
                                    .checked_add(1)
                                    .is_some_and(|events| events >= r.target_generation)
                        }),
                    "retained actuator completion activation differs from its original target calendar",
                )?;
            }
            if r.target_generation == 0 {
                require(
                    r.episode_index == 0
                        && r.original_arrival.anchor.same_bits(&original[i].anchor)
                        && r.original_arrival
                            .target_opening_m
                            .same_bits(&c.initial_states[i].target_opening_m),
                    "generation-zero completion differs from canonical original initial episode",
                )?;
            }
            require(
                r.confirmed_at_s.is_finite()
                    && r.confirmed_at_s > r.original_arrival.anchor.time_s
                    && r.confirmed_at_s <= c.schedule.end_s,
                "retained actuator completion has an invalid confirmation time",
            )?;
            row.anchor = r.original_arrival.anchor;
            row.position = row.anchor.position;
            row.target_opening_m = r.original_arrival.target_opening_m;
            row.arrival = Some(r.original_arrival.clone());
            row.phase = GateActuatorPhase::BeforeWindow;
            row.target_change_count = r.target_generation;
            row.episode_count = r.episode_index.checked_add(1).ok_or_else(|| {
                StructureError::input(
                    crate::structures::StructureInputErrorKind::Definition,
                    "completion episode count overflows",
                )
            })?;
            row.active_episode_index = Some(r.episode_index);
            times.push(r.confirmed_at_s);
        } else {
            times.push(c.schedule.initial_time_s);
        }
        rows.push(row);
    }
    anchors(c, &rows.iter().map(|r| r.anchor).collect::<Vec<_>>())?;
    let mut state = import_values::states(c, &rows)?;
    arrival(c, &mut state)?;
    let recomputed = state.arrival.project()?;
    for (i, receipt) in receipts.iter().enumerate() {
        if let Some(r) = receipt {
            let actual = super::super::projection_values::arrival(
                &recomputed,
                r.original_arrival.anchor,
                r.original_arrival.target_opening_m,
                i,
            )?;
            require(
                actual
                    .as_ref()
                    .is_some_and(|a| a.same_bits(&r.original_arrival)),
                "retained actuator original arrival differs from Numr recomputation",
            )?;
        }
    }
    let evaluated = math::evaluate_at(c, &state, &c.tensor(&times)?).map_err(as_definition)?;
    let completion = evaluated.completion.project()?;
    for (i, receipt) in receipts.iter().enumerate() {
        let actual = super::super::projection_values::completion(&completion, i)?;
        require(
            match (actual.as_ref(), receipt) {
                (Some(a), Some(b)) => a.same_bits(b),
                (None, None) => true,
                _ => false,
            },
            "retained actuator completion differs from independently recomputed confirmation",
        )?;
    }
    Ok(())
}
pub(super) fn changes<R: Runtime<DType = DType>>(
    c: &Arc<TensorActuatorContext<R>>,
    rows: &[GateActuatorState],
) -> Result<(), StructureError>
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
    let previous = rows
        .iter()
        .map(|s| {
            s.last_target_change
                .as_ref()
                .and_then(|r| r.previous_completion.as_ref())
        })
        .collect::<Vec<_>>();
    completions(c, &previous)?;
    let mut seeds = Vec::with_capacity(rows.len());
    let mut times = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let mut seed = hold(c, i);
        if let Some(change) = &row.last_target_change {
            seed.anchor = change.previous_anchor;
            seed.position = seed.anchor.position;
            seed.target_opening_m = change.previous_target_opening_m;
            seed.target_change_count =
                change.target_generation.checked_sub(1).ok_or_else(|| {
                    StructureError::input(
                        crate::structures::StructureInputErrorKind::Definition,
                        "changed actuator target has zero generation",
                    )
                })?;
            seed.phase = GateActuatorPhase::BeforeWindow;
            if let Some(done) = &change.previous_completion {
                seed.last_completion = Some(done.clone());
                seed.active_episode_index = Some(done.episode_index);
            }
            times.push(change.time_s);
        } else {
            times.push(c.schedule.initial_time_s);
        }
        seeds.push(seed);
    }
    anchors(c, &seeds.iter().map(|r| r.anchor).collect::<Vec<_>>())?;
    let mut state = import_values::states(c, &seeds)?;
    arrival(c, &mut state)?;
    let left = math::evaluate_at(c, &state, &c.tensor(&times)?)
        .map_err(as_definition)?
        .position
        .project()?;
    for (i, row) in rows.iter().enumerate() {
        if let Some(change) = &row.last_target_change {
            let actual = super::super::projection_values::position(&left, i)?;
            require(
                actual.same_bits(&change.left_position),
                "retained actuator target-change left position differs from Numr recomputation",
            )?;
        }
    }
    Ok(())
}

pub(super) fn anchors<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    anchors: &[GateActuatorAnchor],
) -> Result<(), StructureError>
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
    require(
        anchors.len() == c.definitions.len(),
        "imported actuator anchor vector has incompatible length",
    )?;
    let mut rows = Vec::with_capacity(anchors.len());
    for (i, anchor) in anchors.iter().enumerate() {
        let mut row = hold(c, i);
        row.anchor = *anchor;
        row.position = anchor.position;
        row.target_opening_m = anchor.position.opening_m;
        rows.push(row);
    }
    let state = import_values::states(c, &rows)?;
    let proof = math::forward(c, &state.anchor, &state.target, &state.anchor.epoch, &c.one)
        .map_err(as_definition)?
        .position
        .project()?;
    for (i, a) in anchors.iter().enumerate() {
        require(
            super::super::projection_values::position(&proof, i)?.same_bits(&a.position),
            "imported actuator anchor error differs from Numr endpoint-distance evidence",
        )?;
    }
    Ok(())
}
