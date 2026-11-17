//! Owning sampled control continuation with explicitly assumed instantaneous pump response.

use super::{ControlledPumpOutput, ControlledPumpTrial, PumpResponseDefinition};
use crate::structures::tensor::controlled_pump::{
    TensorControlledPumpContext, TensorControlledPumpSelection,
};
use crate::structures::{
    ControlEvent, ControlSample, OperatingControlBatch, StructureError, StructureOperation,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Complete accepted controls and an explicit memoryless actual pump response model.
#[derive(Clone, Debug)]
pub struct ControlledPumpBatch<R: Runtime<DType = DType>> {
    pub(crate) context: Arc<TensorControlledPumpContext<R>>,
    pub(crate) controls: OperatingControlBatch<R>,
}
impl<R: Runtime<DType = DType>> ControlledPumpBatch<R> {
    /// Adopt any checked pump-only controller continuation with complete explicit response evidence.
    ///
    /// Advanced controls remain eligible because this response has no independent continuation memory.
    pub fn new(
        controls: OperatingControlBatch<R>,
        definitions: Vec<PumpResponseDefinition>,
    ) -> Result<Self, StructureError> {
        let context = Arc::new(TensorControlledPumpContext::new(&controls, definitions)?);
        Ok(Self { context, controls })
    }
    /// Accepted command continuation, exposed immutably for explicit control projections.
    pub fn controls(&self) -> &OperatingControlBatch<R> {
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
    ) -> Result<ControlledPumpTrial<R>, StructureError>
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
        let controls = self.controls.trial_to(end_s, samples)?.into_next();
        Ok(ControlledPumpTrial {
            next: Self {
                context: Arc::clone(&self.context),
                controls,
            },
        })
    }
    pub(crate) fn selected(&self) -> Result<TensorControlledPumpSelection<R>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        self.context.select(&self.controls.state.high)
    }
    /// Explicitly project actual operation, original controller identity, and response assumptions.
    pub fn outputs(&self) -> Result<Vec<ControlledPumpOutput>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
        super::projection::outputs(self)
    }
    /// Explicitly project original-order operation packets without claiming hydraulic or transfer acceptance.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError>
    where
        R::Client: numr::ops::ConditionalOps<R>,
    {
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
