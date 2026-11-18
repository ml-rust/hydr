//! Laws under an interval operation, and interval evidence projected per structure.

use super::{OperatingSourceBatch, OperatingSourceInterval};
use crate::structures::tensor::{
    actuator::record::TensorRecord,
    operating_source::{TensorOperatingSourceKind, TensorOperatingSourceValue},
};
use crate::structures::{
    ControlLatch, GateActuatorPosition, PumpResponseModel, StructureError, StructureHeads,
    StructureInputErrorKind, StructureOperatingPoint, StructureOperation, StructureProposal,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> OperatingSourceInterval<R> {
    /// Interval start in seconds.
    #[doc(hidden)]
    pub fn start_s(&self) -> f64 {
        self.inner.start_s
    }
    /// Interval end in seconds.
    #[doc(hidden)]
    pub fn end_s(&self) -> f64 {
        self.inner.end_s
    }
    /// Evaluate every law at `heads` under the interval operation.
    ///
    /// `source` must be the registry that built this interval.
    /// Manual packets follow the original manual-owned subsequence.
    #[doc(hidden)]
    pub fn propose(
        &self,
        source: &OperatingSourceBatch<R>,
        heads: &[StructureHeads],
        manual_operations: &[StructureOperation],
    ) -> Result<StructureProposal<R>, StructureError>
    where
        R::Client: numr::ops::CumulativeOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::Interp1dAlgorithms<R>,
    {
        if !Arc::ptr_eq(&source.context, &self.inner.context) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                "interval operation belongs to another operating-source registry",
            ));
        }
        let heads_input = crate::structures::tensor::TensorStructureInput::heads(
            &source.context.structures,
            heads,
        )?;
        if manual_operations.len() != source.context.manual_slots.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "supply exactly every manual-owned operation in original complete subsequence order",
            ));
        }
        source.hydraulic(
            heads,
            heads_input,
            manual_operations,
            &self.inner.interval.point,
        )
    }
}

/// Interval evidence for one structure.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct OperatingSourceIntervalReceipt {
    /// Registered source identity.
    pub source_id: String,
    /// Original complete structure index.
    pub element: usize,
    /// Original complete structure identity.
    pub structure_id: String,
    /// Interval start in seconds.
    pub start_s: f64,
    /// Interval end in seconds.
    pub end_s: f64,
    /// Source-specific endpoint evidence and interval operation.
    pub detail: OperatingSourceIntervalDetail,
}

/// Source-specific interval evidence.
///
/// For a gate the mean of the ideal opening lies within `absolute_error_m + midpoint_rounding_m`
/// of `opening_m` when the path is linear. An actuator inside an unresolved arrival window can
/// arrive within the interval. Its path stays monotone, so the mean lies in
/// `[lower_opening_m, upper_opening_m]`.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub enum OperatingSourceIntervalDetail {
    /// The supplied manual packet. The registry holds no interval evidence for it.
    Manual,
    /// Prescribed trajectory read at both ends.
    PrescribedGate {
        /// Opening at the start in metres.
        start_opening_m: f64,
        /// Opening at the end in metres.
        end_opening_m: f64,
        /// Interval opening the law takes, in metres.
        opening_m: f64,
        /// Larger endpoint interpolation error in metres.
        absolute_error_m: f64,
        /// Exact distance from `opening_m` to the mean of the endpoint openings, in metres.
        midpoint_rounding_m: f64,
        /// Closed by definition at both ends: exact closure over the interval.
        closed: bool,
    },
    /// Actuator position at the accepted start and at the old-target end.
    ActuatedGate {
        /// Original actuator identity.
        actuator_id: String,
        /// Accepted position at the start.
        start: GateActuatorPosition,
        /// Old-target position at the end, before any endpoint target packet.
        end: GateActuatorPosition,
        /// Interval opening the law takes, in metres.
        opening_m: f64,
        /// Smaller endpoint lower bound in metres.
        lower_opening_m: f64,
        /// Larger endpoint upper bound in metres.
        upper_opening_m: f64,
        /// Larger endpoint absolute error in metres.
        absolute_error_m: f64,
        /// Exact distance from `opening_m` to the mean of the endpoint centers, in metres.
        midpoint_rounding_m: f64,
        /// Canonical closure at both ends: exact closure over the interval.
        closed: bool,
    },
    /// Accepted right-side pump operation at the start, held over the interval.
    ControlledPump {
        /// Original controller identity.
        controller_id: String,
        /// Accepted latch at the start.
        latch: ControlLatch,
        /// Declared response model.
        response_model: PumpResponseModel,
        /// Held operation.
        operating_point: StructureOperatingPoint,
    },
}

impl<R: Runtime<DType = DType>> OperatingSourceInterval<R> {
    /// Project interval evidence for every structure in original order.
    #[doc(hidden)]
    pub fn receipts(&self) -> Result<Vec<OperatingSourceIntervalReceipt>, StructureError> {
        let data = &self.inner;
        let context = &data.context;
        let n = context.structures.elements.len();
        let read = |tensor| crate::structures::projection::read(tensor, n);
        let start_opening = read(&data.start.opening)?;
        let end_opening = read(&data.end.opening)?;
        let opening = read(&data.interval.point.opening)?;
        let error = read(&data.interval.point.absolute_error)?;
        let rounding = read(&data.interval.rounding)?;
        let closed = read(&data.interval.point.closed)?;
        let mut details = vec![None; n];
        for ((group, start), end) in context
            .groups
            .iter()
            .zip(&data.start.values)
            .zip(&data.end.values)
        {
            match (&group.kind, start, end) {
                (
                    TensorOperatingSourceKind::Manual,
                    TensorOperatingSourceValue::Manual,
                    TensorOperatingSourceValue::Manual,
                ) => {
                    for &slot in &group.slots {
                        details[slot] = Some(OperatingSourceIntervalDetail::Manual);
                    }
                }
                (
                    TensorOperatingSourceKind::Prescribed(_),
                    TensorOperatingSourceValue::Prescribed,
                    TensorOperatingSourceValue::Prescribed,
                ) => {
                    for &slot in &group.slots {
                        details[slot] = Some(OperatingSourceIntervalDetail::PrescribedGate {
                            start_opening_m: start_opening[slot],
                            end_opening_m: end_opening[slot],
                            opening_m: opening[slot],
                            absolute_error_m: error[slot],
                            midpoint_rounding_m: rounding[slot],
                            closed: closed[slot] == 1.0,
                        });
                    }
                }
                (
                    TensorOperatingSourceKind::Actuated(actuators),
                    TensorOperatingSourceValue::Actuated(start),
                    TensorOperatingSourceValue::Actuated(end),
                ) => {
                    let start = start.position.project()?;
                    let end = end.position.project()?;
                    for (lane, &slot) in group.slots.iter().enumerate() {
                        let start = crate::structures::project_actuator_position(&start, lane)?;
                        let end = crate::structures::project_actuator_position(&end, lane)?;
                        details[slot] = Some(OperatingSourceIntervalDetail::ActuatedGate {
                            actuator_id: actuators.definitions[lane].actuator_id.clone(),
                            start,
                            end,
                            opening_m: opening[slot],
                            lower_opening_m: start.lower_opening_m.min(end.lower_opening_m),
                            upper_opening_m: start.upper_opening_m.max(end.upper_opening_m),
                            absolute_error_m: error[slot],
                            midpoint_rounding_m: rounding[slot],
                            closed: closed[slot] == 1.0,
                        });
                    }
                }
                (
                    TensorOperatingSourceKind::ControlledPump(pumps),
                    TensorOperatingSourceValue::ControlledPump(start),
                    TensorOperatingSourceValue::ControlledPump(_),
                ) => {
                    let count = group.slots.len();
                    let running = crate::structures::projection::read(&start.running, count)?;
                    let speed = crate::structures::projection::read(&start.speed, count)?;
                    let high = crate::structures::projection::read(&start.high, count)?;
                    for (lane, &slot) in group.slots.iter().enumerate() {
                        details[slot] = Some(OperatingSourceIntervalDetail::ControlledPump {
                            controller_id: pumps.controls.definitions[lane].controller_id.clone(),
                            latch: if high[lane] == 0.0 {
                                ControlLatch::Low
                            } else {
                                ControlLatch::High
                            },
                            response_model: pumps.definitions[lane].model,
                            operating_point: if running[lane] == 0.0 {
                                StructureOperatingPoint::PumpStoppedIsolated
                            } else {
                                StructureOperatingPoint::PumpRunning {
                                    speed_ratio: speed[lane],
                                }
                            },
                        });
                    }
                }
                _ => {
                    return Err(StructureError::numerical(
                        "interval endpoint differs from its registered source kind",
                    ));
                }
            }
        }
        details
            .into_iter()
            .enumerate()
            .map(|(element, detail)| {
                let group = context
                    .owner
                    .get(element)
                    .and_then(|&owner| context.groups.get(owner))
                    .ok_or_else(|| {
                        StructureError::numerical("interval receipt lacks its registered owner")
                    })?;
                Ok(OperatingSourceIntervalReceipt {
                    source_id: group.id.clone(),
                    element,
                    structure_id: context.structures.elements[element].id.clone(),
                    start_s: data.start_s,
                    end_s: data.end_s,
                    detail: detail.ok_or_else(|| {
                        StructureError::numerical("interval receipt lacks its source evidence")
                    })?,
                })
            })
            .collect()
    }
}
