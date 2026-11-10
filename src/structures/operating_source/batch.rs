//! Stateless exclusive source registries and point-only constitutive proposals.

use super::{
    ActuatorSourceFrame, OperatingSourceBinding, OperatingSourceProposal, OperatingSourceSupport,
};
use crate::cpu::structures::{
    CpuStructureInput, DerivedGateOperations,
    operating_source::{CpuOperatingSourceContext, assemble},
};
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureHeads, StructureInputErrorKind,
    StructureOperation, StructureProposal,
};
use std::sync::Arc;

/// Immutable exclusive source registry. Every point requires original manual packets and fresh actuator frames.
#[derive(Clone, Debug)]
pub struct OperatingSourceBatch {
    pub(crate) context: Arc<CpuOperatingSourceContext>,
}
impl StructureBatch {
    /// Bind exactly one explicit writer to every original element without capturing accepted actuator state.
    pub fn with_operating_sources(
        &self,
        sources: Vec<OperatingSourceBinding<'_>>,
    ) -> Result<OperatingSourceBatch, StructureError> {
        Ok(OperatingSourceBatch {
            context: Arc::new(super::binding::build(Arc::clone(&self.context), sources)?),
        })
    }
}
impl OperatingSourceBatch {
    /// Complete immutable actual structure definitions in original order.
    pub fn elements(&self) -> &[StructureElement] {
        &self.context.structures.elements
    }
    /// Closed bounded-source intersection. None declares no source interval or future manual hold.
    pub fn support(&self) -> Option<OperatingSourceSupport> {
        self.context.support
    }
    pub(super) fn check_time(&self, time_s: f64) -> Result<(), StructureError> {
        if !time_s.is_finite() {
            return Err(StructureError::input(
                StructureInputErrorKind::NonfiniteInput,
                "operating source point requires finite original query time",
            ));
        }
        if self
            .support()
            .is_some_and(|support| time_s < support.start_s || time_s > support.end_s)
        {
            return Err(StructureError::input(
                StructureInputErrorKind::UnsupportedRegime,
                "operating source query lies outside common closed support",
            ));
        }
        Ok(())
    }
    /// Evaluate hydraulic laws at admitted centers while retaining mandatory source uncertainty receipts.
    ///
    /// Manual packets follow the complete original manual-owned subsequence. Frames follow registered actuated-group order.
    /// This method advances no response, accepts no transfer, and supplies no ideal-discharge enclosure or integrated volume.
    pub fn propose_at(
        &self,
        time_s: f64,
        heads: &[StructureHeads],
        manual_operations: &[StructureOperation],
        actuator_frames: &[ActuatorSourceFrame<'_>],
    ) -> Result<OperatingSourceProposal, StructureError> {
        self.check_time(time_s)?;
        let context = &self.context.structures;
        let heads_input = CpuStructureInput::heads(context, heads)?;
        if manual_operations.len() != self.context.manual_slots.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "supply exactly every manual-owned operation in original complete subsequence order",
            ));
        }
        let point = assemble(&self.context, time_s, actuator_frames)?;
        let input = CpuStructureInput::with_derived_heads(
            context,
            heads_input,
            manual_operations,
            DerivedGateOperations {
                slots: &self.context.derived_slots,
                mask: &self.context.derived_mask,
                opening: &point.opening,
                isolated: &point.closed,
            },
        )?;
        let result = crate::numerics::structures::proposal(context, &input)?;
        let hydraulic = StructureProposal::from_derived(
            Arc::clone(context),
            heads.to_vec(),
            manual_operations.to_vec(),
            self.context.derived_slots.clone(),
            point.opening.clone(),
            result,
        );
        Ok(OperatingSourceProposal::new(
            Arc::clone(&self.context),
            time_s,
            hydraulic,
            point,
        ))
    }
}
