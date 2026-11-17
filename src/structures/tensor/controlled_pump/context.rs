//! Immutable original sampled pump literals without independent response memory.

use super::TensorControlledPumpSelection;
use crate::structures::tensor::control::TensorControlContext;
use crate::structures::{
    ControlCommand, OperatingControlBatch, PumpResponseDefinition, StructureError,
    StructureInputErrorKind, StructureLaw,
};
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct TensorControlledPumpContext<R: Runtime<DType = DType>> {
    pub controls: Arc<TensorControlContext<R>>,
    pub definitions: Vec<PumpResponseDefinition>,
    low_running: Tensor<R>,
    high_running: Tensor<R>,
    low_speed: Tensor<R>,
    high_speed: Tensor<R>,
    low_power: Tensor<R>,
    high_power: Tensor<R>,
}
fn literal(command: ControlCommand) -> Result<(f64, f64, f64), StructureError> {
    match command {
        ControlCommand::PumpStoppedIsolated => Ok((0.0, 1.0, 1.0)),
        ControlCommand::PumpRunning { speed_ratio } => Ok((
            1.0,
            speed_ratio,
            if crate::structures::tensor::power_metadata::power_of_two(speed_ratio) {
                1.0
            } else {
                0.0
            },
        )),
        ControlCommand::GateTarget { .. } => Err(StructureError::input(
            StructureInputErrorKind::Definition,
            "controlled pump response requires pump-only original commands",
        )),
    }
}
impl<R: Runtime<DType = DType>> TensorControlledPumpContext<R> {
    pub(crate) fn new(
        controls: &OperatingControlBatch<R>,
        definitions: Vec<PumpResponseDefinition>,
    ) -> Result<Self, StructureError> {
        if definitions.len() != controls.definitions().len() || definitions.is_empty() {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "controlled pump response requires one complete ordered definition per original controller",
            ));
        }
        let mut low = Vec::with_capacity(definitions.len());
        let mut high = Vec::with_capacity(definitions.len());
        for (lane, (response, control)) in
            definitions.iter().zip(controls.definitions()).enumerate()
        {
            let slot = controls.context.slots.get(lane).copied().ok_or_else(|| {
                StructureError::input(
                    StructureInputErrorKind::Definition,
                    "checked pump controller lacks its actual original structure slot",
                )
            })?;
            if response.structure_id != control.structure_id {
                return Err(StructureError::input(StructureInputErrorKind::Identity,"controlled pump response definitions must match every original controlled pump in order").at(slot));
            }
            if !controls
                .elements()
                .get(slot)
                .is_some_and(|element| matches!(element.law, StructureLaw::Pump(_)))
            {
                return Err(StructureError::input(
                    StructureInputErrorKind::Definition,
                    "controlled pump response requires an actual pump-only controlled subset",
                )
                .at(slot));
            }
            crate::structures::definition::text(
                &response.response_provenance,
                "pump response provenance",
            )
            .map_err(|error| error.at(slot))?;
            crate::structures::definition::text(
                &response.ideal_response_applicability,
                "ideal instantaneous pump response applicability",
            )
            .map_err(|error| error.at(slot))?;
            low.push(literal(control.low_command).map_err(|error| error.at(slot))?);
            high.push(literal(control.high_command).map_err(|error| error.at(slot))?);
        }
        let tensor = |values: &[(f64, f64, f64)], field: fn(&(f64, f64, f64)) -> f64| {
            controls
                .context
                .structures
                .tensor(&values.iter().map(field).collect::<Vec<_>>())
        };
        Ok(Self {
            controls: Arc::clone(&controls.context),
            definitions,
            low_running: tensor(&low, |v| v.0)?,
            high_running: tensor(&high, |v| v.0)?,
            low_speed: tensor(&low, |v| v.1)?,
            high_speed: tensor(&high, |v| v.1)?,
            low_power: tensor(&low, |v| v.2)?,
            high_power: tensor(&high, |v| v.2)?,
        })
    }
    pub(crate) fn select(
        &self,
        high: &Tensor<R>,
    ) -> Result<TensorControlledPumpSelection<R>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        let client = &self.controls.structures.client;
        Ok(TensorControlledPumpSelection {
            running: client.where_cond(high, &self.high_running, &self.low_running)?,
            speed: client.where_cond(high, &self.high_speed, &self.low_speed)?,
            power_of_two: client.where_cond(high, &self.high_power, &self.low_power)?,
            high: high.clone(),
        })
    }
}
