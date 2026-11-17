//! Explicit scalar receipt projection from complete batched actuator records.

use super::*;
use crate::structures::StructureError;
use crate::structures::tensor::actuator::*;
use numr::dtype::DType;
use numr::runtime::Runtime;

pub(super) fn value(values: &[f64], lane: usize) -> Result<f64, StructureError> {
    values
        .get(lane)
        .copied()
        .filter(|v| v.is_finite())
        .ok_or_else(|| {
            StructureError::numerical("actuator projection has a missing or nonfinite lane")
        })
}
pub(super) fn index(value: f64) -> Result<usize, StructureError> {
    if value < 0.0
        || !value.is_finite()
        || value > (1u64 << 53) as f64
        || value.fract() != 0.0
        || value as u128 > usize::MAX as u128
    {
        return Err(StructureError::numerical(
            "actuator counter exceeds exact supported integer range",
        ));
    }
    Ok(value as usize)
}
pub(super) fn flag(values: &[f64], lane: usize) -> Result<bool, StructureError> {
    match value(values, lane)? {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(StructureError::numerical(
            "actuator projection mask is not binary",
        )),
    }
}
pub(crate) fn position(
    v: &TensorActuatorPositionValues,
    i: usize,
) -> Result<GateActuatorPosition, StructureError> {
    Ok(GateActuatorPosition {
        opening_m: value(&v.center, i)?,
        lower_opening_m: value(&v.lower, i)?,
        upper_opening_m: value(&v.upper, i)?,
        absolute_error_m: value(&v.error, i)?,
    })
}
pub(super) fn anchor(
    v: &TensorActuatorAnchorValues,
    i: usize,
) -> Result<GateActuatorAnchor, StructureError> {
    Ok(GateActuatorAnchor {
        time_s: value(&v.epoch, i)?,
        position: position(&v.position, i)?,
    })
}
pub(crate) fn phase(value: f64) -> Result<GateActuatorPhase, StructureError> {
    match value {
        0.0 => Ok(GateActuatorPhase::BeforeWindow),
        1.0 => Ok(GateActuatorPhase::InsideWindow),
        2.0 => Ok(GateActuatorPhase::Complete),
        3.0 => Ok(GateActuatorPhase::BeyondHorizon),
        4.0 => Ok(GateActuatorPhase::Stalled),
        5.0 => Ok(GateActuatorPhase::UnresolvedWithStalledSupport),
        _ => Err(StructureError::numerical("actuator phase code is invalid")),
    }
}
pub(super) fn arrival(
    v: &TensorActuatorArrivalValues,
    a: GateActuatorAnchor,
    target: f64,
    i: usize,
) -> Result<Option<GateActuatorArrivalCertificate>, StructureError> {
    if value(&v.kind, i)? != 1.0 {
        return Ok(None);
    }
    Ok(Some(GateActuatorArrivalCertificate {
        anchor: a,
        target_opening_m: target,
        lower_s: value(&v.lower, i)?,
        upper_s: value(&v.upper, i)?,
        width_s: GateActuatorTimeExpansion {
            high_s: value(&v.width_high, i)?,
            low_s: value(&v.width_low, i)?,
        },
    }))
}
fn fma(
    v: &TensorActuatorFmaValues,
    i: usize,
) -> Result<Option<GateActuatorFmaStageReceipt>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    Ok(Some(GateActuatorFmaStageReceipt {
        signed_rate_m_per_s: value(&v.rate, i)?,
        time_component_s: value(&v.time, i)?,
        addend_m: value(&v.addend, i)?,
        result_m: value(&v.result, i)?,
        rounding_error_m: value(&v.error, i)?,
        zero_grid_certificate: flag(&v.zero_grid, i)?,
        exact_binary_witness: if flag(&v.binary_valid, i)? {
            Some(GateActuatorBinaryWitness {
                product_m: value(&v.binary_product, i)?,
                sum_high_m: value(&v.binary_high, i)?,
                sum_low_m: value(&v.binary_low, i)?,
            })
        } else {
            None
        },
    }))
}
fn motion(
    v: &TensorActuatorMotionValues,
    i: usize,
) -> Result<Option<GateActuatorMotionReceipt>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    let primary = fma(&v.primary, i)?
        .ok_or_else(|| StructureError::numerical("actuator motion lacks primary fused evidence"))?;
    Ok(Some(GateActuatorMotionReceipt {
        direction: if flag(&v.direction, i)? {
            GateActuatorDirection::Opening
        } else {
            GateActuatorDirection::Closing
        },
        rate_m_per_s: value(&v.rate, i)?,
        primary,
        elapsed_low: fma(&v.low, i)?,
        raw_error_m: value(&v.error, i)?,
        raw_lower_m: if flag(&v.raw_lower_valid, i)? {
            Some(value(&v.raw_lower, i)?)
        } else {
            None
        },
        raw_upper_m: if flag(&v.raw_upper_valid, i)? {
            Some(value(&v.raw_upper, i)?)
        } else {
            None
        },
    }))
}
fn endpoint(
    v: &TensorActuatorEndpointValues,
    i: usize,
) -> Result<GateActuatorEndpointReceipt, StructureError> {
    Ok(GateActuatorEndpointReceipt {
        anchor_opening_m: value(&v.anchor, i)?,
        motion: motion(&v.motion, i)?,
        lower_opening_m: value(&v.lower, i)?,
        upper_opening_m: value(&v.upper, i)?,
    })
}
pub(super) fn forward(
    v: &TensorActuatorForwardValues,
    i: usize,
) -> Result<Option<GateActuatorForwardReceipt>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    Ok(Some(GateActuatorForwardReceipt {
        anchor: anchor(&v.anchor, i)?,
        query_time_s: value(&v.query, i)?,
        target_opening_m: value(&v.target, i)?,
        elapsed_s: GateActuatorTimeExpansion {
            high_s: value(&v.elapsed_high, i)?,
            low_s: value(&v.elapsed_low, i)?,
        },
        lower_endpoint: endpoint(&v.lower_endpoint, i)?,
        upper_endpoint: endpoint(&v.upper_endpoint, i)?,
        center_motion: motion(&v.center_motion, i)?,
        position: position(&v.position, i)?,
    }))
}
pub(super) fn completion(
    v: &TensorActuatorCompletionValues,
    i: usize,
) -> Result<Option<GateActuatorCompletionReceipt>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    let a = anchor(&v.anchor, i)?;
    let target = value(&v.target, i)?;
    let original_arrival = arrival(&v.arrival, a, target, i)?.ok_or_else(|| {
        StructureError::numerical("actuator completion lacks original finite arrival")
    })?;
    Ok(Some(GateActuatorCompletionReceipt {
        episode_index: index(value(&v.episode, i)?)?,
        target_generation: index(value(&v.generation, i)?)?,
        confirmed_at_s: value(&v.confirmed, i)?,
        original_arrival,
        effective_upper_s: value(&v.effective_upper, i)?,
        effective_width_s: GateActuatorTimeExpansion {
            high_s: value(&v.width_high, i)?,
            low_s: value(&v.width_low, i)?,
        },
        forward: forward(&v.forward, i)?,
    }))
}
pub(super) fn update<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    v: &TensorActuatorUpdateValues,
    i: usize,
) -> Result<Option<GateTargetUpdate>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    let d = c
        .definitions
        .get(i)
        .ok_or_else(|| StructureError::numerical("actuator update lane lacks its definition"))?;
    Ok(Some(GateTargetUpdate {
        actuator_id: d.actuator_id.clone(),
        structure_id: d.structure_id.clone(),
        target_opening_m: value(&v.target, i)?,
        issued_at_s: value(&v.issued, i)?,
        available_at_s: value(&v.available, i)?,
    }))
}
fn change<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    v: &TensorActuatorChangeValues,
    i: usize,
) -> Result<Option<GateActuatorTargetChangeReceipt>, StructureError> {
    if !flag(&v.valid, i)? {
        return Ok(None);
    }
    let update = update(c, &v.update, i)?.ok_or_else(|| {
        StructureError::numerical("actuator target change lacks its original update")
    })?;
    Ok(Some(GateActuatorTargetChangeReceipt {
        target_index: index(value(&v.index, i)?)?,
        time_s: value(&v.time, i)?,
        previous_target_opening_m: value(&v.previous_target, i)?,
        previous_anchor: anchor(&v.previous_anchor, i)?,
        previous_completion: completion(&v.previous_completion, i)?,
        update,
        left_position: position(&v.left, i)?,
        new_anchor: anchor(&v.new_anchor, i)?,
        target_generation: index(value(&v.generation, i)?)?,
        new_episode_index: if flag(&v.new_episode_valid, i)? {
            Some(index(value(&v.new_episode, i)?)?)
        } else {
            None
        },
    }))
}
pub(super) fn state<R: Runtime<DType = DType>>(
    c: &TensorActuatorContext<R>,
    v: &TensorActuatorStateValues,
    i: usize,
) -> Result<GateActuatorState, StructureError> {
    let a = anchor(&v.anchor, i)?;
    let target = value(&v.target, i)?;
    let d = c
        .definitions
        .get(i)
        .ok_or_else(|| StructureError::numerical("actuator state lane lacks its definition"))?;
    Ok(GateActuatorState {
        actuator_id: d.actuator_id.clone(),
        anchor: a,
        target_opening_m: target,
        position: position(&v.position, i)?,
        phase: phase(value(&v.phase, i)?)?,
        arrival: arrival(&v.arrival, a, target, i)?,
        target_change_count: index(value(&v.generation, i)?)?,
        episode_count: index(value(&v.episodes, i)?)?,
        arrival_count: index(value(&v.arrivals, i)?)?,
        active_episode_index: if flag(&v.active_valid, i)? {
            Some(index(value(&v.active_index, i)?)?)
        } else {
            None
        },
        latest_update: update(c, &v.latest_update, i)?,
        last_target_change: change(c, &v.last_change, i)?,
        last_completion: completion(&v.last_completion, i)?,
    })
}
