//! Owning atomic sampled-controller and actual gate-response candidates.

use super::{ControlledGateEvent, ControlledGateTrial};
use crate::structures::tensor::controlled_gate::TensorControlledGateContext;
use crate::structures::{
    ControlSample, GateActuatorBatch, OperatingControlBatch, StructureError,
    StructureInputErrorKind,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Complete accepted sampled commands and separately modelled actual gate response.
#[derive(Clone, Debug)]
pub struct ControlledGateBatch<R: Runtime<DType = DType>> {
    context: Arc<TensorControlledGateContext<R>>,
    pub(crate) controls: OperatingControlBatch<R>,
    pub(crate) actuators: GateActuatorBatch<R>,
}

impl<R: Runtime<DType = DType>> ControlledGateBatch<R> {
    /// Bind fresh initial children with one sampled controller per supplied actual actuator.
    pub fn new(
        controls: OperatingControlBatch<R>,
        actuators: GateActuatorBatch<R>,
    ) -> Result<Self, StructureError> {
        let context = Arc::new(TensorControlledGateContext::new(&controls, &actuators)?);
        Ok(Self {
            context,
            controls,
            actuators,
        })
    }

    /// Accepted command continuation, exposed immutably for explicit projections.
    pub fn controls(&self) -> &OperatingControlBatch<R> {
        &self.controls
    }

    /// Accepted actual response, exposed immutably for operating-source frames and projections.
    pub fn actuators(&self) -> &GateActuatorBatch<R> {
        &self.actuators
    }

    /// Common accepted time in seconds.
    pub fn time(&self) -> f64 {
        self.controls.time()
    }

    /// Common accepted positive-duration interval count.
    pub fn accepted_steps(&self) -> usize {
        self.controls.accepted_steps()
    }

    /// Next decision, actuator arrival boundary, or common terminal boundary.
    pub fn next_event(&self) -> Result<Option<ControlledGateEvent>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>,
    {
        let control = self.controls.next_event();
        let actuator = self.actuators.next_event()?;
        let time_s = match (control, actuator) {
            (Some(control), Some(actuator)) => control.time_s.min(actuator.time_s),
            (Some(control), None) => control.time_s,
            (None, Some(actuator)) => actuator.time_s,
            (None, None) => return Ok(None),
        };
        let control = control.filter(|event| event.time_s == time_s);
        let actuator = actuator.filter(|event| event.time_s == time_s);
        Ok(Some(ControlledGateEvent {
            time_s,
            decision_index: control.and_then(|event| event.decision_index),
            arrival_lower_boundary: actuator.is_some_and(|event| event.arrival_lower_boundary),
            arrival_upper_boundary: actuator.is_some_and(|event| event.arrival_upper_boundary),
            terminal: control.is_some_and(|event| event.terminal)
                || actuator.is_some_and(|event| event.terminal),
        }))
    }

    /// Trial both children atomically, applying each endpoint command after old-target response.
    pub fn trial_to(
        &self,
        end_s: f64,
        decision_samples: &[ControlSample],
    ) -> Result<ControlledGateTrial<R>, StructureError>
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
        let duration = end_s - self.time();
        if !end_s.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "controlled gate interval=[{},{}],duration={duration}: supply a finite positive original duration",
                    self.time(),
                    end_s
                ),
            ));
        }
        let event = self.next_event()?.ok_or_else(|| {
            StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "controlled gate trial exceeds terminal support: retain the original common horizon",
            )
        })?;
        if end_s > event.time_s {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "controlled gate interval crosses its next event: split at the common event",
            ));
        }
        let controls = self.controls.trial_to(end_s, decision_samples)?.into_next();
        let update = if end_s == event.time_s && event.decision_index.is_some() {
            Some(self.context.update(&controls, &self.actuators, end_s)?)
        } else {
            None
        };
        let actuators = self
            .actuators
            .trial_to_with_update(end_s, update)?
            .into_next();
        Ok(ControlledGateTrial {
            next: Self {
                context: Arc::clone(&self.context),
                controls,
                actuators,
            },
        })
    }
}
