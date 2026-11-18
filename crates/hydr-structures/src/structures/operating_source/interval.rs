//! Interval operations: the operation each law takes over one event-free interval.
//!
//! A coordinator ends every interval at `next_event_after`, so no prescribed knot, actuator event,
//! or control decision lies strictly inside it.
//!
//! | Owner | Interval operation |
//! | --- | --- |
//! | Manual | The supplied packet |
//! | Prescribed or actuated gate | Midpoint of the start and end openings |
//! | Gate closed at both ends | Exact closure, with the unknown-head waiver |
//! | Controlled pump | The accepted right-side operation at the start, held |
//!
//! A gate opening moves linearly inside an event-free interval, so its mean equals the midpoint.

use super::{OperatingSourceBatch, OperatingSourceFrame};
use crate::structures::tensor::operating_source::{
    TensorOperatingSourceContext, TensorOperatingSourceInterval, TensorOperatingSourceKind,
    TensorOperatingSourcePoint, assemble, combine,
};
use crate::structures::{
    ControlledPumpPreparedInterval, GateActuatorPreparedInterval, StructureError,
    StructureInputErrorKind,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

/// Prepared dynamic response for one interval, in dynamic-group registration order.
#[doc(hidden)]
#[derive(Debug)]
pub enum OperatingSourceIntervalFrame<'a, R: Runtime<DType = DType>> {
    /// Old-target actuator response from `GateActuatorBatch::prepare_to`.
    ActuatedGate {
        /// Registered source identity.
        source_id: &'a str,
        /// Prepared interval. Its start batch and end position supply both endpoints.
        actuator: &'a GateActuatorPreparedInterval<R>,
    },
    /// Held pump operation from `ControlledPumpBatch::prepare_to`.
    ControlledPump {
        /// Registered source identity.
        source_id: &'a str,
        /// Prepared interval. Its start batch supplies the held operation.
        pump: &'a ControlledPumpPreparedInterval<R>,
    },
}
impl<R: Runtime<DType = DType>> Copy for OperatingSourceIntervalFrame<'_, R> {}
impl<R: Runtime<DType = DType>> Clone for OperatingSourceIntervalFrame<'_, R> {
    fn clone(&self) -> Self {
        *self
    }
}

#[derive(Debug)]
pub(super) struct IntervalData<R: Runtime<DType = DType>> {
    pub context: Arc<TensorOperatingSourceContext<R>>,
    pub start_s: f64,
    pub end_s: f64,
    pub start: TensorOperatingSourcePoint<R>,
    pub end: TensorOperatingSourcePoint<R>,
    pub interval: TensorOperatingSourceInterval<R>,
}

/// Immutable interval operation for every registered structure.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct OperatingSourceInterval<R: Runtime<DType = DType>> {
    pub(super) inner: Arc<IntervalData<R>>,
}

impl<R: Runtime<DType = DType>> OperatingSourceBatch<R> {
    /// Build the interval operation from the start and end of one event-free interval.
    ///
    /// Supply one prepared frame per dynamic group, in registration order, each prepared over
    /// `[start_s, end_s]`. A prescribed knot strictly inside the interval rejects.
    #[doc(hidden)]
    pub fn interval_operation(
        &self,
        start_s: f64,
        end_s: f64,
        frames: &[OperatingSourceIntervalFrame<'_, R>],
    ) -> Result<OperatingSourceInterval<R>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::LinearInterp1dDiagnostics<R>,
    {
        let duration = end_s - start_s;
        if !start_s.is_finite() || !end_s.is_finite() || !duration.is_finite() || duration <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!(
                    "operating interval=[{start_s},{end_s}],duration={duration}: supply a finite positive duration"
                ),
            ));
        }
        self.check_time(start_s)?;
        self.check_time(end_s)?;
        for group in &self.context.groups {
            if let TensorOperatingSourceKind::Prescribed(source) = &group.kind {
                let first = source.event_times.partition_point(|t| *t <= start_s);
                if source.event_times.get(first).is_some_and(|t| *t < end_s) {
                    return Err(StructureError::input(
                        StructureInputErrorKind::UnsupportedRegime,
                        format!(
                            "source={:?},interval=[{start_s},{end_s}]: interval contains a prescribed knot: end it at next_event_after",
                            group.id
                        ),
                    ));
                }
            }
        }
        let mut starts = Vec::with_capacity(frames.len());
        let mut ends = Vec::with_capacity(frames.len());
        for frame in frames {
            let (id, start, end, start_frame, end_frame) = match *frame {
                OperatingSourceIntervalFrame::ActuatedGate {
                    source_id,
                    actuator,
                } => (
                    source_id,
                    actuator.start_s(),
                    actuator.end_s(),
                    OperatingSourceFrame::ActuatedGate {
                        source_id,
                        actuator: actuator.start(),
                    },
                    OperatingSourceFrame::ActuatedGate {
                        source_id,
                        actuator: &actuator.frame,
                    },
                ),
                OperatingSourceIntervalFrame::ControlledPump { source_id, pump } => (
                    source_id,
                    pump.start_s(),
                    pump.end_s(),
                    OperatingSourceFrame::ControlledPump {
                        source_id,
                        pump: pump.start(),
                    },
                    OperatingSourceFrame::ControlledPump {
                        source_id,
                        pump: &pump.frame,
                    },
                ),
            };
            if start.to_bits() != start_s.to_bits() || end.to_bits() != end_s.to_bits() {
                return Err(StructureError::input(
                    StructureInputErrorKind::UnsupportedRegime,
                    format!(
                        "source={id:?},prepared=[{start},{end}],interval=[{start_s},{end_s}]: prepare every dynamic frame over the same interval"
                    ),
                ));
            }
            starts.push(start_frame);
            ends.push(end_frame);
        }
        let start = assemble(&self.context, start_s, &starts)?;
        let end = assemble(&self.context, end_s, &ends)?;
        let interval = combine(&self.context, &start, &end)?;
        Ok(OperatingSourceInterval {
            inner: Arc::new(IntervalData {
                context: Arc::clone(&self.context),
                start_s,
                end_s,
                start,
                end,
                interval,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::OperatingSourceIntervalFrame as Frame;
    use crate::structures::interval_fixture::*;
    use crate::structures::*;

    struct Mixed {
        context: StructureBatch<Cpu>,
        path: GateTrajectoryBatch<Cpu>,
        gate: ControlledGateBatch<Cpu>,
        pump: ControlledPumpBatch<Cpu>,
        registry: OperatingSourceBatch<Cpu>,
    }

    fn mixed(knots: &[(f64, f64)]) -> TestResult<Mixed> {
        let context = context()?;
        let path = trajectory(&context, "g0", knots)?;
        let gate = controlled_gate(&context)?;
        let pump = controlled_pump(&context)?;
        let registry = context.with_operating_sources(vec![
            OperatingSourceBinding::Prescribed {
                source_id: "path".into(),
                trajectory: &path,
            },
            OperatingSourceBinding::Actuated {
                source_id: "actual".into(),
                actuator: gate.actuators(),
            },
            OperatingSourceBinding::ControlledPump {
                source_id: "pumps".into(),
                pump: &pump,
            },
            OperatingSourceBinding::Manual {
                source_id: "manual".into(),
                structure_ids: vec!["g3".into()],
            },
        ])?;
        Ok(Mixed {
            context,
            path,
            gate,
            pump,
            registry,
        })
    }

    impl Mixed {
        /// Accept the first decision with every relay switched high.
        fn advance_to_first_decision(&mut self) -> TestResult {
            self.gate = self
                .gate
                .trial_to(1.0, &samples("g1", 1.0, 3.0))?
                .into_next();
            self.pump = self
                .pump
                .trial_to(1.0, &samples("p2", 1.0, 3.0))?
                .into_next();
            Ok(())
        }
        fn interval(&self, end_s: f64) -> TestResult<OperatingSourceInterval<Cpu>> {
            let gate = self.gate.prepare_to(end_s)?;
            let pump = self.pump.prepare_to(end_s)?;
            Ok(self.registry.interval_operation(
                self.gate.time(),
                end_s,
                &[
                    Frame::ActuatedGate {
                        source_id: "actual",
                        actuator: gate.actuators(),
                    },
                    Frame::ControlledPump {
                        source_id: "pumps",
                        pump: &pump,
                    },
                ],
            )?)
        }
    }

    fn manual(opening_m: f64) -> Vec<StructureOperation> {
        vec![StructureOperation {
            structure_id: "g3".into(),
            operating_point: StructureOperatingPoint::GateOpening { opening_m },
        }]
    }

    fn gate_detail(receipt: &OperatingSourceIntervalReceipt) -> TestResult<(f64, f64, bool)> {
        match &receipt.detail {
            OperatingSourceIntervalDetail::PrescribedGate {
                opening_m,
                midpoint_rounding_m,
                closed,
                ..
            }
            | OperatingSourceIntervalDetail::ActuatedGate {
                opening_m,
                midpoint_rounding_m,
                closed,
                ..
            } => Ok((*opening_m, *midpoint_rounding_m, *closed)),
            other => Err(format!("expected gate detail, found {other:?}").into()),
        }
    }

    fn held(receipt: &OperatingSourceIntervalReceipt) -> TestResult<StructureOperatingPoint> {
        match &receipt.detail {
            OperatingSourceIntervalDetail::ControlledPump {
                operating_point, ..
            } => Ok(*operating_point),
            other => Err(format!("expected pump detail, found {other:?}").into()),
        }
    }

    #[test]
    fn midpoint_of_a_linear_path_equals_its_mean_bit_for_bit() -> TestResult {
        let mut fixture = mixed(&[(0.0, 0.25), (4.0, 0.75)])?;
        fixture.advance_to_first_decision()?;
        let receipts = fixture.interval(2.0)?.receipts()?;
        let prescribed = fixture.path.evaluate(1.5)?.outputs()?.remove(0).opening_m;
        assert_eq!(
            gate_detail(&receipts[0])?,
            (prescribed, 0.0, false),
            "prescribed midpoint equals the path at mid-interval"
        );
        assert_eq!(prescribed.to_bits(), 0.4375f64.to_bits());
        let actual = fixture
            .gate
            .trial_to(1.5, &[])?
            .actuators()
            .outputs()?
            .remove(0)
            .position
            .opening_m;
        assert_eq!(gate_detail(&receipts[1])?, (actual, 0.0, false));
        assert_eq!(actual.to_bits(), 0.125f64.to_bits());
        assert_eq!(
            held(&receipts[2])?,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
        );
        assert_eq!(receipts[3].detail, OperatingSourceIntervalDetail::Manual);
        Ok(())
    }

    #[test]
    fn an_inexact_midpoint_reports_its_exact_rounding() -> TestResult {
        let fixture = mixed(&[(0.0, 0.1), (4.0, 0.7)])?;
        let a = fixture.path.evaluate(0.0)?.outputs()?.remove(0).opening_m;
        let b = fixture.path.evaluate(1.0)?.outputs()?.remove(0).opening_m;
        let sum = a + b;
        let carried = sum - a;
        let residual = (a - (sum - carried)) + (b - carried);
        let (opening, rounding, closed) = gate_detail(&fixture.interval(1.0)?.receipts()?[0])?;
        assert_eq!(opening.to_bits(), (sum * 0.5).to_bits());
        assert_eq!(rounding.to_bits(), (residual * 0.5).abs().to_bits());
        assert!(!closed);
        Ok(())
    }

    #[test]
    fn the_pump_operation_is_held_from_the_start_of_the_interval() -> TestResult {
        let fixture = mixed(&[(0.0, 0.25), (4.0, 0.75)])?;
        let interval = fixture.interval(1.0)?;
        let switched = fixture
            .pump
            .trial_to(1.0, &samples("p2", 1.0, 3.0))?
            .operations()?;
        assert_eq!(
            switched[0].operating_point,
            StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 }
        );
        assert_eq!(
            held(&interval.receipts()?[2])?,
            StructureOperatingPoint::PumpStoppedIsolated
        );
        Ok(())
    }

    #[test]
    fn closed_at_both_ends_is_exact_closure_with_the_unknown_head_waiver() -> TestResult {
        let mut fixture = mixed(&[(0.0, 0.0), (4.0, 0.0)])?;
        let heads = heads(&fixture.context, HeadSupport::Unknown);
        let interval = fixture.interval(1.0)?;
        let receipts = interval.receipts()?;
        assert_eq!(gate_detail(&receipts[0])?, (0.0, 0.0, true));
        assert_eq!(gate_detail(&receipts[1])?, (0.0, 0.0, true));
        let outputs = interval
            .propose(&fixture.registry, &heads, &manual(0.0))?
            .outputs()?;
        for slot in [0, 1, 3] {
            assert_eq!(outputs[slot].regime, StructureRegime::GateClosed);
            assert_eq!(outputs[slot].discharge_m3_per_s.to_bits(), 0.0f64.to_bits());
        }
        fixture.advance_to_first_decision()?;
        let opening = fixture.interval(2.0)?;
        assert!(!gate_detail(&opening.receipts()?[1])?.2);
        let error = opening
            .propose(&fixture.registry, &heads, &manual(0.0))
            .err()
            .ok_or("closure at one end grants no unknown-head waiver")?;
        assert!(error.input_kind().is_some());
        Ok(())
    }

    #[test]
    fn the_interval_law_equals_the_point_law_under_the_same_operation() -> TestResult {
        let mut fixture = mixed(&[(0.0, 0.25), (4.0, 0.75)])?;
        fixture.advance_to_first_decision()?;
        let heads = heads(&fixture.context, HeadSupport::Wet { stage_m: 0.0 });
        let interval = fixture.interval(2.0)?;
        let receipts = interval.receipts()?;
        let operations = [
            (
                "g0",
                StructureOperatingPoint::GateOpening {
                    opening_m: gate_detail(&receipts[0])?.0,
                },
            ),
            (
                "g1",
                StructureOperatingPoint::GateOpening {
                    opening_m: gate_detail(&receipts[1])?.0,
                },
            ),
            ("p2", held(&receipts[2])?),
            (
                "g3",
                StructureOperatingPoint::GateOpening { opening_m: 0.5 },
            ),
        ];
        let explicit = fixture
            .context
            .with_operating_sources(vec![OperatingSourceBinding::Manual {
                source_id: "all".into(),
                structure_ids: operations.iter().map(|(id, _)| (*id).into()).collect(),
            }])?
            .propose_at(
                0.0,
                &heads,
                &operations
                    .iter()
                    .map(|(id, point)| StructureOperation {
                        structure_id: (*id).into(),
                        operating_point: *point,
                    })
                    .collect::<Vec<_>>(),
                &[],
            )?
            .outputs()?
            .into_iter()
            .map(|output| output.hydraulic)
            .collect::<Vec<_>>();
        let interval = interval
            .propose(&fixture.registry, &heads, &manual(0.5))?
            .outputs()?;
        assert_eq!(bits(&interval), bits(&explicit));
        Ok(())
    }

    #[test]
    fn an_interval_containing_an_event_or_another_span_rejects() -> TestResult {
        let fixture = mixed(&[(0.0, 0.25), (0.5, 0.5), (4.0, 0.75)])?;
        let error = fixture.interval(1.0).err().ok_or("interior knot rejects")?;
        assert!(error.to_string().contains("prescribed knot"));
        let aligned = mixed(&[(0.0, 0.25), (4.0, 0.75)])?;
        let gate = aligned.gate.prepare_to(1.0)?;
        let pump = aligned.pump.prepare_to(1.0)?;
        let frames = [
            Frame::ActuatedGate {
                source_id: "actual",
                actuator: gate.actuators(),
            },
            Frame::ControlledPump {
                source_id: "pumps",
                pump: &pump,
            },
        ];
        let error = aligned
            .registry
            .interval_operation(0.0, 0.5, &frames)
            .err()
            .ok_or("another span rejects")?;
        assert_eq!(
            error.input_kind(),
            Some(StructureInputErrorKind::UnsupportedRegime)
        );
        let reversed = [frames[1], frames[0]];
        let error = aligned
            .registry
            .interval_operation(0.0, 1.0, &reversed)
            .err()
            .ok_or("frames out of registration order reject")?;
        assert_eq!(error.input_kind(), Some(StructureInputErrorKind::Identity));
        Ok(())
    }
}
