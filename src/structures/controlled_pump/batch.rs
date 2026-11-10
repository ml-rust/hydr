//! Owning sampled control continuation with explicitly assumed instantaneous pump response.

use super::{ControlledPumpOutput, ControlledPumpTrial, PumpResponseDefinition};
use crate::cpu::structures::controlled_pump::{
    CpuControlledPumpContext, CpuControlledPumpSelection,
};
use crate::structures::{
    ControlEvent, ControlSample, OperatingControlBatch, StructureError, StructureOperation,
};
use std::sync::Arc;

/// Complete accepted controls and an explicit memoryless actual pump response model.
#[derive(Clone, Debug)]
pub struct ControlledPumpBatch {
    pub(crate) context: Arc<CpuControlledPumpContext>,
    pub(crate) controls: OperatingControlBatch,
}
impl ControlledPumpBatch {
    /// Adopt any checked pump-only controller continuation with complete explicit response evidence.
    ///
    /// Advanced controls remain eligible because this response has no independent continuation memory.
    pub fn new(
        controls: OperatingControlBatch,
        definitions: Vec<PumpResponseDefinition>,
    ) -> Result<Self, StructureError> {
        let context = Arc::new(CpuControlledPumpContext::new(&controls, definitions)?);
        Ok(Self { context, controls })
    }
    /// Accepted command continuation, exposed immutably for explicit control projections.
    pub fn controls(&self) -> &OperatingControlBatch {
        &self.controls
    }
    /// Complete original ordered response assumptions and provenance.
    pub fn definitions(&self) -> &[PumpResponseDefinition] {
        &self.context.definitions
    }
    /// Accepted original control time in seconds.
    pub fn time(&self) -> f64 {
        self.controls.time()
    }
    /// Original accepted control interval count. No independent response counter exists.
    pub fn accepted_steps(&self) -> usize {
        self.controls.accepted_steps()
    }
    /// Next original sampled decision or terminal support boundary.
    pub fn next_event(&self) -> Option<ControlEvent> {
        self.controls.next_event()
    }
    /// Trial sampled controls without changing accepted operation or qualification memory.
    ///
    /// The candidate command governs the response on the endpoint's right side. No water transfer follows.
    pub fn trial_to(
        &self,
        end_s: f64,
        samples: &[ControlSample],
    ) -> Result<ControlledPumpTrial, StructureError> {
        let controls = self.controls.trial_to(end_s, samples)?.into_next();
        Ok(ControlledPumpTrial {
            next: Self {
                context: Arc::clone(&self.context),
                controls,
            },
        })
    }
    pub(crate) fn selected(&self) -> Result<CpuControlledPumpSelection, StructureError> {
        self.context.select(&self.controls.state.high)
    }
    /// Explicitly project actual operation, original controller identity, and response assumptions.
    pub fn outputs(&self) -> Result<Vec<ControlledPumpOutput>, StructureError> {
        super::projection::outputs(self)
    }
    /// Explicitly project original-order operation packets without claiming hydraulic or transfer acceptance.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError> {
        Ok(self
            .outputs()?
            .into_iter()
            .map(|output| StructureOperation {
                structure_id: output.structure_id,
                operating_point: output.operating_point,
            })
            .collect())
    }
}
