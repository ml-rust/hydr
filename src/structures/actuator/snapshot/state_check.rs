//! Complete bounded continuation admission without command-history replay.

use super::super::*;
use super::compatibility::{as_definition, require};
use super::receipt_bits::ReceiptBits;
use crate::numerics::structures::actuator as math;
use crate::structures::{StructureError, StructureLaw};
use crate::tensor::structures::actuator::{TensorActuatorState, import_values};
use numr::dtype::DType;
use numr::runtime::Runtime;

fn activated_position(p: GateActuatorPosition, target: f64) -> GateActuatorPosition {
    if p.lower_opening_m == target && p.upper_opening_m == target {
        GateActuatorPosition {
            opening_m: target,
            lower_opening_m: target,
            upper_opening_m: target,
            absolute_error_m: 0.0,
        }
    } else {
        p
    }
}
fn exact_count(n: usize) -> bool {
    n as u128 <= (1u128 << 53)
}
fn position(p: &GateActuatorPosition, maximum: f64) -> bool {
    [
        p.opening_m,
        p.lower_opening_m,
        p.upper_opening_m,
        p.absolute_error_m,
    ]
    .iter()
    .all(|v| v.is_finite())
        && p.lower_opening_m >= 0.0
        && p.lower_opening_m <= p.opening_m
        && p.opening_m <= p.upper_opening_m
        && p.upper_opening_m <= maximum
        && p.absolute_error_m >= 0.0
        && (p.upper_opening_m == 0.0 || p.opening_m > 0.0)
}
fn anchor(a: &GateActuatorAnchor, maximum: f64, start: f64, time: f64) -> bool {
    a.time_s.is_finite() && a.time_s >= start && a.time_s <= time && position(&a.position, maximum)
}
fn packet<R: Runtime<DType = DType>>(
    batch: &GateActuatorBatch<R>,
    i: usize,
    p: &GateTargetUpdate,
    time: f64,
) -> bool {
    let d = &batch.definitions()[i];
    p.actuator_id == d.actuator_id
        && p.structure_id == d.structure_id
        && p.target_opening_m.is_finite()
        && p.target_opening_m >= 0.0
        && p.issued_at_s.is_finite()
        && p.available_at_s.is_finite()
        && p.available_at_s <= p.issued_at_s
        && p.issued_at_s <= time
}
pub(super) fn check<R: Runtime<DType = DType>>(
    batch: &GateActuatorBatch<R>,
    rows: &[GateActuatorState],
    time: f64,
    processed: usize,
    steps: usize,
) -> Result<TensorActuatorState<R>, StructureError>
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
        rows.len() == batch.definitions().len(),
        "imported actuator state count differs from ordered actual subset",
    )?;
    let initial = math::initial(&batch.context).map_err(as_definition)?;
    let initial = super::super::projection::states(&batch.context, &initial)?;
    for (i, s) in rows.iter().enumerate() {
        let slot = batch.context.slots[i];
        let maximum = match &batch.elements()[slot].law {
            StructureLaw::HydrostaticRectangularGate(g) => g.maximum_opening_m(),
            _ => {
                return Err(StructureError::numerical(
                    "actuator context contains a non-gate",
                ));
            }
        };
        require(
            s.actuator_id == batch.definitions()[i].actuator_id
                && position(&s.position, maximum)
                && anchor(&s.anchor, maximum, batch.schedule().initial_time_s, time)
                && s.target_opening_m.is_finite()
                && s.target_opening_m >= 0.0
                && s.target_opening_m <= maximum,
            "imported actuator state has incompatible identity, position, anchor, or target",
        )?;
        require(
            [s.target_change_count, s.episode_count, s.arrival_count]
                .into_iter()
                .all(exact_count)
                && s.target_change_count <= processed
                && s.episode_count >= initial[i].episode_count
                && (s.target_change_count == 0 || s.episode_count > 0)
                && s.arrival_count <= s.episode_count
                && s.arrival_count <= steps
                && s.target_change_count
                    .checked_add(initial[i].episode_count)
                    .is_some_and(|n| s.episode_count <= n),
            "imported actuator counters contradict accepted events or exact integer capacity",
        )?;
        require(
            s.active_episode_index
                .is_none_or(|n| exact_count(n) && n.checked_add(1) == Some(s.episode_count)),
            "imported active actuator episode is not the latest counted episode",
        )?;
        let hold = s.anchor.position.lower_opening_m == s.target_opening_m
            && s.anchor.position.upper_opening_m == s.target_opening_m;
        require(
            s.active_episode_index.is_some() != hold,
            "imported actuator active episode contradicts exact target hold",
        )?;
        require(
            (s.arrival_count == 0) == s.last_completion.is_none(),
            "imported actuator arrival count contradicts retained completion presence",
        )?;
        if let Some(done) = &s.last_completion {
            require(
                exact_count(done.episode_index)
                    && exact_count(done.target_generation)
                    && done.episode_index < s.episode_count
                    && done.target_generation <= s.target_change_count
                    && done
                        .episode_index
                        .checked_add(1)
                        .is_some_and(|n| s.arrival_count <= n)
                    && done.confirmed_at_s <= time
                    && anchor(
                        &done.original_arrival.anchor,
                        maximum,
                        batch.schedule().initial_time_s,
                        done.confirmed_at_s,
                    ),
                "imported actuator completion counters, anchor, or chronology are inconsistent",
            )?;
            if done.target_generation == s.target_change_count {
                require(
                    Some(done.episode_index) == s.active_episode_index
                        && done.original_arrival.anchor.same_bits(&s.anchor)
                        && done
                            .original_arrival
                            .target_opening_m
                            .same_bits(&s.target_opening_m),
                    "current-generation completion differs from the current active episode or original anchor",
                )?;
            } else {
                let superseded_at = s.target_change_count.checked_sub(1).and_then(|previous| {
                    if done.target_generation == previous {
                        Some(s.anchor.time_s)
                    } else if done.target_generation < previous {
                        s.last_target_change
                            .as_ref()
                            .map(|change| change.previous_anchor.time_s)
                    } else {
                        None
                    }
                });
                require(
                    superseded_at.is_some_and(|time| done.confirmed_at_s <= time),
                    "historical actuator completion follows the retained activation that supersedes its target",
                )?;
                if let Some(active) = s.active_episode_index {
                    require(
                        done.episode_index < active,
                        "historical completion reuses the current active episode index",
                    )?;
                }
            }
        }
        if s.target_change_count == 0 {
            require(
                s.last_target_change.is_none()
                    && s.anchor.same_bits(&initial[i].anchor)
                    && s.target_opening_m
                        .same_bits(&batch.initial_states()[i].target_opening_m),
                "generation-zero actuator anchor or original target changed",
            )?;
        } else {
            let change = s.last_target_change.as_ref().ok_or_else(|| {
                StructureError::input(
                    crate::structures::StructureInputErrorKind::Definition,
                    "changed actuator target lacks its retained change receipt",
                )
            })?;
            require(
                exact_count(change.target_index)
                    && exact_count(change.target_generation)
                    && change.target_generation == s.target_change_count
                    && change.target_index < processed
                    && change
                        .target_index
                        .checked_add(1)
                        .is_some_and(|n| s.target_change_count <= n)
                    && batch
                        .schedule()
                        .target_times_s
                        .get(change.target_index)
                        .is_some_and(|t| t.same_bits(&change.time_s))
                    && change.time_s <= time
                    && packet(batch, i, &change.update, time)
                    && change.update.issued_at_s.same_bits(&change.time_s)
                    && change
                        .update
                        .target_opening_m
                        .same_bits(&s.target_opening_m)
                    && s.target_opening_m != change.previous_target_opening_m
                    && change.new_anchor.same_bits(&s.anchor)
                    && change.new_anchor.time_s.same_bits(&change.time_s)
                    && change.new_anchor.position.same_bits(&activated_position(
                        change.left_position,
                        s.target_opening_m,
                    ))
                    && anchor(
                        &change.previous_anchor,
                        maximum,
                        batch.schedule().initial_time_s,
                        change.time_s,
                    )
                    && change.previous_anchor.time_s < change.time_s
                    && change.previous_target_opening_m.is_finite()
                    && change.previous_target_opening_m >= 0.0
                    && change.previous_target_opening_m <= maximum,
                "retained actuator target change contradicts original calendar, active target, or complete anchor",
            )?;
            require(
                change.new_episode_index == s.active_episode_index,
                "retained actuator target change disagrees with current active episode",
            )?;
            if let Some(last) = &s.last_completion
                && last.target_generation.checked_add(1) == Some(change.target_generation)
            {
                require(
                    change
                        .previous_completion
                        .as_ref()
                        .is_some_and(|previous| previous.same_bits(last)),
                    "retained previous completion omits or differs from the latest old-target completion authority",
                )?;
            }
            if let Some(done) = &change.previous_completion {
                require(
                    exact_count(done.episode_index)
                        && exact_count(done.target_generation)
                        && done.target_generation.checked_add(1) == Some(change.target_generation)
                        && done
                            .original_arrival
                            .anchor
                            .same_bits(&change.previous_anchor)
                        && done
                            .original_arrival
                            .target_opening_m
                            .same_bits(&change.previous_target_opening_m)
                        && done.confirmed_at_s <= change.time_s
                        && done.episode_index < s.episode_count,
                    "retained previous completion differs from the old target generation or original anchor",
                )?;
                require(
                    s.arrival_count > 0
                        && s.last_completion.as_ref().is_some_and(|last| {
                            last.episode_index >= done.episode_index
                                && last.target_generation >= done.target_generation
                                && last.confirmed_at_s >= done.confirmed_at_s
                        }),
                    "retained previous completion contradicts accepted arrival count or latest completion chronology",
                )?;
            }
            if change.target_generation > 1 {
                require(
                    batch.schedule().target_times_s[..change.target_index]
                        .iter()
                        .any(|t| t.same_bits(&change.previous_anchor.time_s)),
                    "retained previous anchor does not match an earlier original target event",
                )?;
            }
            if change.target_generation == 1 {
                require(
                    change.previous_anchor.same_bits(&initial[i].anchor)
                        && change
                            .previous_target_opening_m
                            .same_bits(&batch.initial_states()[i].target_opening_m),
                    "first actuator target change loses its original initial anchor or target",
                )?;
            }
        }
        if processed == 0 {
            require(
                s.latest_update.is_none(),
                "unprocessed actuator target calendar contains a latest packet",
            )?;
        } else {
            let latest = s.latest_update.as_ref().ok_or_else(|| {
                StructureError::input(
                    crate::structures::StructureInputErrorKind::Definition,
                    "processed actuator targets lack their latest causal packet",
                )
            })?;
            require(
                packet(batch, i, latest, time)
                    && latest.target_opening_m <= maximum
                    && latest.target_opening_m == s.target_opening_m
                    && latest
                        .issued_at_s
                        .same_bits(&batch.schedule().target_times_s[processed - 1]),
                "latest actuator packet differs from the latest processed event or active numerical target",
            )?;
            if let Some(change) = &s.last_target_change
                && change.target_index == processed - 1
            {
                require(
                    latest.same_bits(&change.update),
                    "latest actuator change and latest packet differ at the same original event",
                )?;
            }
        }
    }
    super::proof_check::anchors(
        &batch.context,
        &rows.iter().map(|s| s.anchor).collect::<Vec<_>>(),
    )?;
    super::proof_check::completions(
        &batch.context,
        &rows
            .iter()
            .map(|s| s.last_completion.as_ref())
            .collect::<Vec<_>>(),
    )?;
    super::proof_check::changes(&batch.context, rows)?;
    let mut state = import_values::states(&batch.context, rows)?;
    state.arrival = math::arrival::arrival(&batch.context, &state.anchor, &state.target)
        .map_err(as_definition)?;
    let arrived = super::super::projection::states(&batch.context, &state)?;
    for (a, b) in arrived.iter().zip(rows) {
        require(
            a.arrival.same_bits(&b.arrival),
            "imported actuator current arrival differs from Numr recomputation",
        )?;
    }
    let evaluated = math::evaluate(&batch.context, &state, time).map_err(as_definition)?;
    state.position = evaluated.position;
    state.phase = evaluated.phase;
    state.last_completion = evaluated.completion;
    require(
        super::super::projection::states(&batch.context, &state)?
            .as_slice()
            .same_bits(rows),
        "imported actuator current position, phase, or completion differs from Numr recomputation",
    )?;
    Ok(state)
}
