//! Original packet admission before actuator motion evaluation.

use crate::cpu::structures::actuator::{
    CpuActuatorContext, CpuActuatorUpdate, record::TensorRecord,
};
use crate::structures::{GateTargetUpdate, StructureError, StructureInputErrorKind};

pub(crate) fn admit_update(
    ctx: &CpuActuatorContext,
    updates: &[GateTargetUpdate],
    time: f64,
    expected: bool,
) -> Result<CpuActuatorUpdate, StructureError> {
    if !expected {
        if !updates.is_empty() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "non-target actuator interval requires an empty update slice",
            ));
        }
        return Ok(CpuActuatorUpdate::zero(ctx));
    }
    if updates.len() != ctx.definitions.len() {
        return Err(StructureError::input(
            StructureInputErrorKind::OperatingPoint,
            "target event requires one complete ordered update per actuator",
        ));
    }
    for ((update, definition), &slot) in updates.iter().zip(&ctx.definitions).zip(&ctx.slots) {
        if update.actuator_id != definition.actuator_id
            || update.structure_id != definition.structure_id
        {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "target packet identity differs from its ordered actual actuator",
            )
            .at(slot));
        }
        if [
            update.target_opening_m,
            update.issued_at_s,
            update.available_at_s,
        ]
        .iter()
        .any(|v| !v.is_finite())
        {
            return Err(StructureError::input(
                StructureInputErrorKind::NonfiniteInput,
                "target packet requires finite opening and original timestamps",
            )
            .at(slot));
        }
        if update.issued_at_s != time || update.available_at_s > update.issued_at_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "target packet must be issued at the original event with causal availability",
            )
            .at(slot));
        }
        let limit = match &ctx.structures.elements[slot].law {
            crate::structures::StructureLaw::HydrostaticRectangularGate(gate) => {
                gate.maximum_opening_m()
            }
            _ => {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    "actuator lost its actual gate identity",
                )
                .at(slot));
            }
        };
        if update.target_opening_m < 0.0 || update.target_opening_m > limit {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "target opening lies outside actual gate range",
            )
            .at(slot));
        }
    }
    Ok(CpuActuatorUpdate {
        valid: ctx.one.clone(),
        target: ctx.tensor(
            &updates
                .iter()
                .map(|u| u.target_opening_m)
                .collect::<Vec<_>>(),
        )?,
        issued: ctx.tensor(&updates.iter().map(|u| u.issued_at_s).collect::<Vec<_>>())?,
        available: ctx.tensor(&updates.iter().map(|u| u.available_at_s).collect::<Vec<_>>())?,
    })
}
