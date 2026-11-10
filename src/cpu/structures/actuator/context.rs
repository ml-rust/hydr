//! Immutable actual-gate binding and original finite-rate inputs.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::{
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, StructureError,
    StructureInputErrorKind, StructureLaw,
};
use numr::dtype::DType;
use numr::runtime::{FloatArithmeticContract, FusedMulAddContract, Runtime, cpu::CpuRuntime};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Debug)]
pub(crate) struct CpuActuatorContext {
    pub structures: Arc<CpuStructureContext>,
    pub definitions: Vec<GateActuatorDefinition>,
    pub initial_states: Vec<GateActuatorInitialState>,
    pub schedule: GateActuatorSchedule,
    pub slots: Vec<usize>,
    pub zero: CpuTensor,
    pub one: CpuTensor,
    pub maximum: CpuTensor,
    pub opening_rate: CpuTensor,
    pub closing_rate: CpuTensor,
    pub opening_power: CpuTensor,
    pub closing_power: CpuTensor,
    pub absolute: CpuTensor,
    pub relative: CpuTensor,
    pub window: CpuTensor,
}
impl CpuActuatorContext {
    pub(crate) fn new(
        structures: Arc<CpuStructureContext>,
        definitions: Vec<GateActuatorDefinition>,
        initial_states: Vec<GateActuatorInitialState>,
        schedule: GateActuatorSchedule,
    ) -> Result<Self, StructureError> {
        let invalid = |message| StructureError::input(StructureInputErrorKind::Definition, message);
        if CpuRuntime::float_arithmetic_contract(DType::F64)
            != FloatArithmeticContract::Ieee754Basic
            || CpuRuntime::fused_mul_add_contract(DType::F64) != FusedMulAddContract::Ieee754Fused
        {
            return Err(invalid(
                "actual gate actuation requires strict CPU F64 basic and fused arithmetic contracts",
            ));
        }
        if definitions.is_empty() || definitions.len() != initial_states.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "actuator definitions require a nonempty complete ordered initial state",
            ));
        }
        if !schedule.initial_time_s.is_finite()
            || !schedule.end_s.is_finite()
            || schedule.initial_time_s < 0.0
            || schedule.end_s <= schedule.initial_time_s
        {
            return Err(invalid(
                "actuator schedule requires finite nonnegative initial time before terminal support",
            ));
        }
        let mut previous = schedule.initial_time_s;
        for &time in &schedule.target_times_s {
            if !time.is_finite() || time <= previous || time > schedule.end_s {
                return Err(invalid(
                    "actuator target times must increase strictly inside original finite support",
                ));
            }
            previous = time;
        }
        let index: BTreeMap<_, _> = structures
            .elements
            .iter()
            .enumerate()
            .map(|(i, e)| (e.id.as_str(), i))
            .collect();
        let mut ids = BTreeSet::new();
        let mut slots = Vec::with_capacity(definitions.len());
        let mut maximum = Vec::with_capacity(definitions.len());
        for (definition, initial) in definitions.iter().zip(&initial_states) {
            let identity =
                |message| StructureError::input(StructureInputErrorKind::Identity, message);
            if definition.actuator_id.trim().is_empty()
                || definition.actuator_id.chars().any(char::is_control)
                || !ids.insert(&definition.actuator_id)
                || initial.actuator_id != definition.actuator_id
            {
                return Err(identity(
                    "actuator IDs must be unique, nonempty, and match ordered initial state",
                ));
            }
            let slot = index
                .get(definition.structure_id.as_str())
                .copied()
                .ok_or_else(|| identity("actuator references an unknown actual structure"))?;
            if slots.last().is_some_and(|previous| *previous >= slot) {
                return Err(identity(
                    "actuator subset must follow unique original structure order",
                )
                .at(slot));
            }
            let limit = match &structures.elements[slot].law {
                StructureLaw::HydrostaticRectangularGate(gate) => gate.maximum_opening_m(),
                _ => return Err(identity("actuator requires an actual hydrostatic gate").at(slot)),
            };
            if [
                definition.opening_rate_m_per_s,
                definition.closing_rate_m_per_s,
                definition.accuracy.maximum_absolute_opening_error_m,
                definition.accuracy.maximum_relative_opening_error,
                definition.accuracy.maximum_arrival_window_s,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err(invalid(
                    "actuator rates and explicit accuracy bounds must be finite and nonnegative",
                )
                .at(slot));
            }
            if [
                &definition.rate_provenance,
                &definition.ideal_response_applicability,
            ]
            .iter()
            .any(|s| s.trim().is_empty() || s.chars().any(char::is_control))
            {
                return Err(invalid("actuator rates and ideal response require explicit provenance and applicability").at(slot));
            }
            if [initial.opening_m, initial.target_opening_m]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0 || *v > limit)
            {
                return Err(invalid(
                    "initial actuator opening and target must lie inside actual gate range",
                )
                .at(slot));
            }
            slots.push(slot);
            maximum.push(limit);
        }
        let tensor = |values: &[f64]| structures.tensor(values);
        let values = |f: fn(&GateActuatorDefinition) -> f64| {
            tensor(&definitions.iter().map(f).collect::<Vec<_>>())
        };
        let zero = tensor(&vec![0.0; definitions.len()])?;
        let one = tensor(&vec![1.0; definitions.len()])?;
        let power = crate::cpu::structures::power_metadata::power_of_two;
        let result = Self {
            maximum: tensor(&maximum)?,
            opening_rate: values(|d| d.opening_rate_m_per_s)?,
            closing_rate: values(|d| d.closing_rate_m_per_s)?,
            opening_power: tensor(
                &definitions
                    .iter()
                    .map(|d| {
                        if power(d.opening_rate_m_per_s) && d.opening_rate_m_per_s > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>(),
            )?,
            closing_power: tensor(
                &definitions
                    .iter()
                    .map(|d| {
                        if power(d.closing_rate_m_per_s) && d.closing_rate_m_per_s > 0.0 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .collect::<Vec<_>>(),
            )?,
            absolute: values(|d| d.accuracy.maximum_absolute_opening_error_m)?,
            relative: values(|d| d.accuracy.maximum_relative_opening_error)?,
            window: values(|d| d.accuracy.maximum_arrival_window_s)?,
            structures,
            definitions,
            initial_states,
            schedule,
            slots,
            zero,
            one,
        };
        Ok(result)
    }
    pub(crate) fn tensor(&self, values: &[f64]) -> Result<CpuTensor, StructureError> {
        self.structures.tensor(values)
    }
    pub(crate) fn constant(&self, value: f64) -> Result<CpuTensor, StructureError> {
        if value.to_bits() == 0.0f64.to_bits() {
            return Ok(self.zero.clone());
        }
        if value.to_bits() == 1.0f64.to_bits() {
            return Ok(self.one.clone());
        }
        self.tensor(&vec![value; self.definitions.len()])
    }
    pub(crate) fn reject(&self, mask: &CpuTensor, message: &str) -> Result<(), StructureError> {
        if self.structures.any(mask)? {
            let flags = mask.try_to_vec::<f64>()?;
            let lane = flags.iter().position(|v| *v != 0.0).ok_or_else(|| {
                StructureError::numerical("actuator rejection mask contradicts reduction")
            })?;
            let slot = self.slots.get(lane).copied().ok_or_else(|| {
                StructureError::numerical("actuator rejection mask has an invalid lane count")
            })?;
            return Err(StructureError::numerical(message).at(slot));
        }
        Ok(())
    }
}
