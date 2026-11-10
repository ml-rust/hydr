//! Complete caller-owned receipt transposition into batched tensor records.

use super::record::TensorRecord;
use super::*;
use crate::structures::*;
use numr::dtype::DType;
use numr::runtime::Runtime;

type Rows<'a, T> = [Option<&'a T>];
fn scalar<T>(rows: &Rows<'_, T>, f: impl Fn(&T) -> f64) -> Vec<f64> {
    rows.iter().map(|r| r.map_or(0.0, &f)).collect()
}
fn nested<'a, T, U>(rows: &Rows<'a, T>, f: impl Fn(&'a T) -> Option<&'a U>) -> Vec<Option<&'a U>> {
    rows.iter().map(|r| r.and_then(&f)).collect()
}
fn valid<T>(rows: &Rows<'_, T>) -> Vec<f64> {
    rows.iter().map(|r| f64::from(r.is_some())).collect()
}

fn position(rows: &Rows<'_, GateActuatorPosition>) -> TensorActuatorPositionValues {
    TensorActuatorPositionValues {
        center: scalar(rows, |r| r.opening_m),
        lower: scalar(rows, |r| r.lower_opening_m),
        upper: scalar(rows, |r| r.upper_opening_m),
        error: scalar(rows, |r| r.absolute_error_m),
    }
}
fn anchor(rows: &Rows<'_, GateActuatorAnchor>) -> TensorActuatorAnchorValues {
    TensorActuatorAnchorValues {
        epoch: scalar(rows, |r| r.time_s),
        position: position(&nested(rows, |r| Some(&r.position))),
    }
}
fn arrival(rows: &Rows<'_, GateActuatorArrivalCertificate>) -> TensorActuatorArrivalValues {
    TensorActuatorArrivalValues {
        kind: valid(rows),
        lower: scalar(rows, |r| r.lower_s),
        upper: scalar(rows, |r| r.upper_s),
        width_high: scalar(rows, |r| r.width_s.high_s),
        width_low: scalar(rows, |r| r.width_s.low_s),
    }
}
fn fma(rows: &Rows<'_, GateActuatorFmaStageReceipt>) -> TensorActuatorFmaValues {
    TensorActuatorFmaValues {
        valid: valid(rows),
        rate: scalar(rows, |r| r.signed_rate_m_per_s),
        time: scalar(rows, |r| r.time_component_s),
        addend: scalar(rows, |r| r.addend_m),
        result: scalar(rows, |r| r.result_m),
        error: scalar(rows, |r| r.rounding_error_m),
        zero_grid: scalar(rows, |r| f64::from(r.zero_grid_certificate)),
        binary_valid: scalar(rows, |r| f64::from(r.exact_binary_witness.is_some())),
        binary_product: scalar(rows, |r| {
            r.exact_binary_witness.as_ref().map_or(0.0, |w| w.product_m)
        }),
        binary_high: scalar(rows, |r| {
            r.exact_binary_witness
                .as_ref()
                .map_or(0.0, |w| w.sum_high_m)
        }),
        binary_low: scalar(rows, |r| {
            r.exact_binary_witness.as_ref().map_or(0.0, |w| w.sum_low_m)
        }),
    }
}
fn motion(rows: &Rows<'_, GateActuatorMotionReceipt>) -> TensorActuatorMotionValues {
    TensorActuatorMotionValues {
        valid: valid(rows),
        direction: scalar(rows, |r| {
            f64::from(r.direction == GateActuatorDirection::Opening)
        }),
        rate: scalar(rows, |r| r.rate_m_per_s),
        primary: fma(&nested(rows, |r| Some(&r.primary))),
        low: fma(&nested(rows, |r| r.elapsed_low.as_ref())),
        error: scalar(rows, |r| r.raw_error_m),
        raw_lower_valid: scalar(rows, |r| f64::from(r.raw_lower_m.is_some())),
        raw_lower: scalar(rows, |r| r.raw_lower_m.unwrap_or(0.0)),
        raw_upper_valid: scalar(rows, |r| f64::from(r.raw_upper_m.is_some())),
        raw_upper: scalar(rows, |r| r.raw_upper_m.unwrap_or(0.0)),
    }
}
fn endpoint(rows: &Rows<'_, GateActuatorEndpointReceipt>) -> TensorActuatorEndpointValues {
    TensorActuatorEndpointValues {
        anchor: scalar(rows, |r| r.anchor_opening_m),
        motion: motion(&nested(rows, |r| r.motion.as_ref())),
        lower: scalar(rows, |r| r.lower_opening_m),
        upper: scalar(rows, |r| r.upper_opening_m),
    }
}
fn forward(rows: &Rows<'_, GateActuatorForwardReceipt>) -> TensorActuatorForwardValues {
    TensorActuatorForwardValues {
        valid: valid(rows),
        anchor: anchor(&nested(rows, |r| Some(&r.anchor))),
        query: scalar(rows, |r| r.query_time_s),
        target: scalar(rows, |r| r.target_opening_m),
        elapsed_high: scalar(rows, |r| r.elapsed_s.high_s),
        elapsed_low: scalar(rows, |r| r.elapsed_s.low_s),
        lower_endpoint: endpoint(&nested(rows, |r| Some(&r.lower_endpoint))),
        upper_endpoint: endpoint(&nested(rows, |r| Some(&r.upper_endpoint))),
        center_motion: motion(&nested(rows, |r| r.center_motion.as_ref())),
        position: position(&nested(rows, |r| Some(&r.position))),
    }
}
fn completion(rows: &Rows<'_, GateActuatorCompletionReceipt>) -> TensorActuatorCompletionValues {
    TensorActuatorCompletionValues {
        valid: valid(rows),
        episode: scalar(rows, |r| r.episode_index as f64),
        generation: scalar(rows, |r| r.target_generation as f64),
        confirmed: scalar(rows, |r| r.confirmed_at_s),
        anchor: anchor(&nested(rows, |r| Some(&r.original_arrival.anchor))),
        target: scalar(rows, |r| r.original_arrival.target_opening_m),
        arrival: arrival(&nested(rows, |r| Some(&r.original_arrival))),
        effective_upper: scalar(rows, |r| r.effective_upper_s),
        width_high: scalar(rows, |r| r.effective_width_s.high_s),
        width_low: scalar(rows, |r| r.effective_width_s.low_s),
        forward: forward(&nested(rows, |r| r.forward.as_ref())),
    }
}
fn update(rows: &Rows<'_, GateTargetUpdate>) -> TensorActuatorUpdateValues {
    TensorActuatorUpdateValues {
        valid: valid(rows),
        target: scalar(rows, |r| r.target_opening_m),
        issued: scalar(rows, |r| r.issued_at_s),
        available: scalar(rows, |r| r.available_at_s),
    }
}
fn change(rows: &Rows<'_, GateActuatorTargetChangeReceipt>) -> TensorActuatorChangeValues {
    TensorActuatorChangeValues {
        valid: valid(rows),
        index: scalar(rows, |r| r.target_index as f64),
        time: scalar(rows, |r| r.time_s),
        previous_target: scalar(rows, |r| r.previous_target_opening_m),
        previous_anchor: anchor(&nested(rows, |r| Some(&r.previous_anchor))),
        previous_completion: completion(&nested(rows, |r| r.previous_completion.as_ref())),
        update: update(&nested(rows, |r| Some(&r.update))),
        left: position(&nested(rows, |r| Some(&r.left_position))),
        new_anchor: anchor(&nested(rows, |r| Some(&r.new_anchor))),
        generation: scalar(rows, |r| r.target_generation as f64),
        new_episode_valid: scalar(rows, |r| f64::from(r.new_episode_index.is_some())),
        new_episode: scalar(rows, |r| r.new_episode_index.unwrap_or(0) as f64),
    }
}
fn phase(p: GateActuatorPhase) -> f64 {
    match p {
        GateActuatorPhase::BeforeWindow => 0.0,
        GateActuatorPhase::InsideWindow => 1.0,
        GateActuatorPhase::Complete => 2.0,
        GateActuatorPhase::BeyondHorizon => 3.0,
        GateActuatorPhase::Stalled => 4.0,
        GateActuatorPhase::UnresolvedWithStalledSupport => 5.0,
    }
}
pub(crate) fn states<R: Runtime<DType = DType>>(
    context: &TensorActuatorContext<R>,
    states: &[GateActuatorState],
) -> Result<TensorActuatorState<R>, StructureError> {
    let rows = states.iter().map(Some).collect::<Vec<_>>();
    let mut a = arrival(&nested(&rows, |r| r.arrival.as_ref()));
    for (kind, state) in a.kind.iter_mut().zip(states) {
        if state.arrival.is_none() {
            *kind = match state.phase {
                GateActuatorPhase::BeyondHorizon => 2.0,
                GateActuatorPhase::Stalled => 3.0,
                GateActuatorPhase::UnresolvedWithStalledSupport => 4.0,
                _ => 0.0,
            };
        }
    }
    TensorActuatorState::from_values(
        context,
        &TensorActuatorStateValues {
            anchor: anchor(&nested(&rows, |r| Some(&r.anchor))),
            target: scalar(&rows, |r| r.target_opening_m),
            position: position(&nested(&rows, |r| Some(&r.position))),
            phase: scalar(&rows, |r| phase(r.phase)),
            arrival: a,
            generation: scalar(&rows, |r| r.target_change_count as f64),
            episodes: scalar(&rows, |r| r.episode_count as f64),
            arrivals: scalar(&rows, |r| r.arrival_count as f64),
            active_valid: scalar(&rows, |r| f64::from(r.active_episode_index.is_some())),
            active_index: scalar(&rows, |r| r.active_episode_index.unwrap_or(0) as f64),
            latest_update: update(&nested(&rows, |r| r.latest_update.as_ref())),
            last_change: change(&nested(&rows, |r| r.last_target_change.as_ref())),
            last_completion: completion(&nested(&rows, |r| r.last_completion.as_ref())),
        },
    )
}
