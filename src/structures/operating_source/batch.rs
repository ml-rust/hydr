//! Stateless exclusive source registries and point-only constitutive proposals.

use super::{
    OperatingSourceBinding, OperatingSourceFrame, OperatingSourceProposal, OperatingSourceSupport,
};
use crate::structures::{
    StructureBatch, StructureElement, StructureError, StructureHeads, StructureInputErrorKind,
    StructureOperation, StructureProposal,
};
use crate::tensor::structures::{
    DerivedGateOperations, DerivedPumpOperationData, DerivedPumpOperations, TensorStructureInput,
    operating_source::{TensorOperatingSourceContext, assemble},
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Immutable exclusive registry requiring original manual packets and fresh dynamic frames.
#[derive(Clone, Debug)]
pub struct OperatingSourceBatch<R: Runtime<DType = DType>> {
    pub(crate) context: Arc<TensorOperatingSourceContext<R>>,
}
impl<R: Runtime<DType = DType>> StructureBatch<R> {
    /// Bind one explicit writer per original element without capturing dynamic continuation.
    pub fn with_operating_sources(
        &self,
        sources: Vec<OperatingSourceBinding<'_, R>>,
    ) -> Result<OperatingSourceBatch<R>, StructureError>
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
        Ok(OperatingSourceBatch {
            context: Arc::new(super::binding::build(Arc::clone(&self.context), sources)?),
        })
    }
}
impl<R: Runtime<DType = DType>> OperatingSourceBatch<R> {
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
    /// Manual packets follow the original manual-owned subsequence. Frames follow dynamic-group registration order.
    /// This method advances no response, accepts no transfer, and supplies no ideal-discharge enclosure or integrated volume.
    pub fn propose_at(
        &self,
        time_s: f64,
        heads: &[StructureHeads],
        manual_operations: &[StructureOperation],
        frames: &[OperatingSourceFrame<'_, R>],
    ) -> Result<OperatingSourceProposal<R>, StructureError>
    where
        R::Client: numr::ops::CumulativeOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::Interp1dAlgorithms<R>
            + solvr::interpolate::LinearInterp1dDiagnostics<R>,
    {
        self.check_time(time_s)?;
        let context = &self.context.structures;
        let heads_input = TensorStructureInput::heads(context, heads)?;
        if manual_operations.len() != self.context.manual_slots.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "supply exactly every manual-owned operation in original complete subsequence order",
            ));
        }
        let point = assemble(&self.context, time_s, frames)?;
        let input = TensorStructureInput::with_derived_heads(
            context,
            heads_input,
            manual_operations,
            Some(DerivedGateOperations {
                slots: &self.context.derived_slots,
                mask: &self.context.derived_mask,
                opening: &point.opening,
                isolated: &point.closed,
            }),
            (!self.context.pump_slots.is_empty()).then_some(DerivedPumpOperations {
                slots: &self.context.pump_slots,
                mask: &self.context.pump_mask,
                running: &point.pump_running,
                isolated: &point.pump_isolated,
                speed: &point.pump_speed,
                power_of_two: &point.pump_power_of_two,
            }),
        )?;
        let result = crate::numerics::structures::proposal(context, &input)?;
        let hydraulic = StructureProposal::from_derived(
            Arc::clone(context),
            heads.to_vec(),
            manual_operations.to_vec(),
            self.context.derived_slots.clone(),
            point.opening.clone(),
            if self.context.pump_slots.is_empty() {
                None
            } else {
                Some(DerivedPumpOperationData {
                    slots: self.context.pump_slots.clone(),
                    running: point.pump_running.clone(),
                    speed: point.pump_speed.clone(),
                })
            },
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
