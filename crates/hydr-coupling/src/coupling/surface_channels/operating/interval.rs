//! Event-aligned ends and per-attempt operations.
//!
//! The interval operation depends on no iterate. Each attempt prepares every child once, and
//! every evaluation of the attempt takes the same link laws.

use super::{ACTUATED, OperatingChildren, PUMPED, child_error};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::coupling::surface_channels::structures::LinkOperations;
use crate::structures::{
    ControlledGatePreparedInterval, ControlledPumpPreparedInterval, OperatingSourceBatch,
    OperatingSourceFrame, OperatingSourceInterval, OperatingSourceIntervalDetail,
    OperatingSourceIntervalFrame, StructureError, StructureOperatingPoint,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::LinearInterp1dDiagnostics;
use std::sync::Arc;

/// Link laws and prepared children of one attempt.
pub(in super::super) struct AttemptOperation<R: Runtime<DType = DType>> {
    /// Attempt end, in seconds.
    pub end: f64,
    /// Link laws under the attempt's operations.
    pub operations: Arc<LinkOperations<R>>,
    /// Prepared children, or `None` without operating children.
    pub prepared: Option<PreparedOperation<R>>,
    /// Geometry receipts applied at the start of this interval.
    pub geometry: Vec<crate::coupling::surface_channels::SurfaceChannelGeometryReceipt>,
}

/// Children prepared to the attempt end without samples, and their interval operation.
pub(in super::super) struct PreparedOperation<R: Runtime<DType = DType>> {
    pub(super) gate: Option<ControlledGatePreparedInterval<R>>,
    pub(super) pump: Option<ControlledPumpPreparedInterval<R>>,
    pub(super) interval: OperatingSourceInterval<R>,
}

/// Interval operation of one accepted interval.
pub(in super::super) struct StepOperation<R: Runtime<DType = DType>> {
    /// Operation receipts of every structure from the operating sources.
    pub interval: OperatingSourceInterval<R>,
    /// Operation of every link in table order.
    pub points: Vec<StructureOperatingPoint>,
}

/// The interval operation of `registry` over `[start_s, end_s]` from prepared `frames`.
///
/// Admission keeps it as a function pointer, so a trial calls it without the trajectory
/// interpolation bound.
pub(super) fn interval_operation<R: Runtime<DType = DType>>(
    registry: &OperatingSourceBatch<R>,
    start_s: f64,
    end_s: f64,
    frames: &[OperatingSourceIntervalFrame<'_, R>],
) -> Result<OperatingSourceInterval<R>, StructureError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + LinearInterp1dDiagnostics<R>,
{
    registry.interval_operation(start_s, end_s, frames)
}

impl<R: Runtime<DType = DType>> AttemptOperation<R> {
    /// The declared link laws of `context` over an attempt ending at `end`.
    pub fn declared(context: &Context<R>, end: f64) -> Self {
        Self {
            end,
            operations: Arc::clone(&context.operations),
            prepared: None,
            geometry: Vec::new(),
        }
    }
}

impl<R: Runtime<DType = DType>> OperatingChildren<R>
where
    R::Client: Ops<R>,
{
    /// Fresh frames of the accepted dynamic children, in registration order.
    fn frames(&self) -> Vec<OperatingSourceFrame<'_, R>> {
        let mut frames = Vec::new();
        if let Some(gate) = &self.gate {
            frames.push(OperatingSourceFrame::ActuatedGate {
                source_id: ACTUATED,
                actuator: gate.actuators(),
            });
        }
        if let Some(pump) = &self.pump {
            frames.push(OperatingSourceFrame::ControlledPump {
                source_id: PUMPED,
                pump,
            });
        }
        frames
    }

    /// Time of the first operating event strictly after the accepted time `start`, in seconds.
    ///
    /// Each actuator arrival window counts as one event at its upper bound. The window
    /// encloses one arrival instant, so an end at either bound reaches the same event. An end
    /// inside a window reaches it too, and the next interval may then cross its upper bound.
    /// No interval holds a whole window, a decision, a knot, or a terminal bound strictly
    /// inside it. An interval ends strictly inside a window only at another event or a caller
    /// limit, so the bounds of one window never split off an interval of their own.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] when no event follows: the common operating
    /// support ends at `start`.
    pub fn next_event_after(&self, start: f64) -> Result<f64, SurfaceChannelError> {
        let registry = self
            .binding
            .registry
            .next_coordinated_event_after(start, &self.frames())
            .map_err(child_error("operating sources"))?
            .map(|event| event.time_s);
        let gate = match &self.gate {
            Some(gate) => gate
                .next_coordinated_event()
                .map_err(child_error("controlled-gate"))?
                .map(|event| event.time_s),
            None => None,
        };
        let pump = self
            .pump
            .as_ref()
            .and_then(|pump| pump.next_event())
            .map(|event| event.time_s);
        [registry, gate, pump]
            .into_iter()
            .flatten()
            .reduce(f64::min)
            .ok_or_else(|| {
                SurfaceChannelError::input(format!(
                    "operation-ownership: time={start}: the operating children hold no later event, so their common support ends: end the run at the support end"
                ))
            })
    }

    /// Prepares every child from `start` to `end` and builds the attempt's link laws.
    ///
    /// A gate takes the midpoint opening of its interval, or exact closure when closed at both
    /// ends. A pump holds its accepted operation. A manual structure keeps its declared operation.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] naming the child for every child rejection.
    pub fn prepare(
        &self,
        context: &Context<R>,
        start: f64,
        end: f64,
    ) -> Result<AttemptOperation<R>, SurfaceChannelError> {
        let gate = self
            .gate
            .as_ref()
            .map(|gate| gate.prepare_coordinated_to(end))
            .transpose()
            .map_err(child_error("controlled-gate"))?;
        let pump = self
            .pump
            .as_ref()
            .map(|pump| pump.prepare_to(end))
            .transpose()
            .map_err(child_error("controlled-pump"))?;
        let mut frames = Vec::new();
        if let Some(prepared) = &gate {
            frames.push(OperatingSourceIntervalFrame::ActuatedGate {
                source_id: ACTUATED,
                actuator: prepared.actuators(),
            });
        }
        if let Some(prepared) = &pump {
            frames.push(OperatingSourceIntervalFrame::ControlledPump {
                source_id: PUMPED,
                pump: prepared,
            });
        }
        let binding = &self.binding;
        let interval = (binding.interval)(&binding.registry, start, end, &frames)
            .map_err(child_error("operating sources"))?;
        let mut points = context.operations.points.clone();
        for receipt in interval
            .receipts()
            .map_err(child_error("operating sources"))?
        {
            let point = match receipt.detail {
                OperatingSourceIntervalDetail::Manual => continue,
                OperatingSourceIntervalDetail::PrescribedGate { opening_m, .. }
                | OperatingSourceIntervalDetail::ActuatedGate { opening_m, .. } => {
                    StructureOperatingPoint::GateOpening { opening_m }
                }
                OperatingSourceIntervalDetail::ControlledPump {
                    operating_point, ..
                } => operating_point,
            };
            let slot = binding
                .links
                .get(&receipt.structure_id)
                .and_then(|&link| points.get_mut(link))
                .ok_or_else(|| {
                    SurfaceChannelError::Numerical(format!(
                        "operating structure={:?} lacks its link",
                        receipt.structure_id
                    ))
                })?;
            *slot = point;
        }
        let operations = context.operations.operated(
            &context.link_elements,
            points,
            (context.head_absolute_m, context.head_relative),
        )?;
        Ok(AttemptOperation {
            end,
            operations: Arc::new(operations),
            prepared: Some(PreparedOperation {
                gate,
                pump,
                interval,
            }),
            geometry: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelOperating;
    use super::super::scenario::{build, controlled_pump, signal, trajectory};
    use crate::coupling::surface_channels::pools::scenario::{
        PoolForcing, advance, check as pool_check, declaration, inflow_binding, inflow_input, link,
        pond, pool, stages, surface,
    };
    use crate::coupling::surface_channels::structures::SurfaceChannelOperationOwner;
    use crate::coupling::surface_channels::structures::scenario::{
        Batch, Tanks, TestResult, check, client, gate_law, pump_law, step, structure,
    };
    use crate::structures::{
        ControlLatch, DirectionPolicy, OperatingSourceIntervalDetail, StructureBatch,
        StructureOperatingPoint,
    };

    /// Receipt detail and table operation of link `id` over the last accepted interval.
    fn operation(
        batch: &Batch,
        id: &str,
    ) -> TestResult<(OperatingSourceIntervalDetail, StructureOperatingPoint, f64)> {
        let step = batch.last.as_ref().ok_or("an accepted interval")?;
        let operation = step.operation.as_ref().ok_or("an interval operation")?;
        let receipt = operation
            .interval
            .receipts()?
            .into_iter()
            .find(|receipt| receipt.structure_id == id)
            .ok_or("the structure receipt")?;
        let link = batch
            .context
            .link_ids
            .iter()
            .position(|link| link == id)
            .ok_or("the link")?;
        let signed = step.signed_m3.try_to_vec::<f64>()?[link];
        Ok((receipt.detail, operation.points[link], signed))
    }

    /// A prescribed gate ramps open between two tanks: closed to 1 s, linear to 0.2 m at 3 s,
    /// then held to the end of its support at 4.5 s. Every request asks for 0.75 s.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - No accepted interval holds a knot strictly inside. Knots at 1 s and 3 s end intervals.
    /// - Each interval's endpoint openings equal the path at its start and end, bit for bit.
    /// - Each interval's opening equals `(start + end) * 0.5` bit for bit, and the law takes it.
    /// - Intervals closed at both ends take exact closure and move `0.0` bit for bit.
    /// - Water moves surface to reach once the gate opens. Every interval passes its mass bound,
    ///   both pairings, and the composite balance.
    #[test]
    fn a_prescribed_gate_takes_the_midpoint_of_its_path() -> TestResult {
        let knots = [(0.0, 0.0), (1.0, 0.0), (3.0, 0.2), (4.5, 0.2)];
        let mut gate = structure(
            "gate",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            [0.0, 0.0],
        );
        gate.operation = SurfaceChannelOperationOwner::Prescribed;
        let assignment = trajectory("gate", &knots)?;
        let path = StructureBatch::<numr::runtime::cpu::CpuRuntime>::new(
            client(),
            vec![gate.element.clone()],
        )?
        .with_gate_trajectory_subset(vec![assignment.clone()])?;
        let opening =
            |time: f64| -> TestResult<f64> { Ok(path.evaluate(time)?.outputs()?[0].opening_m) };
        let operating = SurfaceChannelOperating {
            gates: None,
            pumps: None,
            trajectories: vec![assignment],
            signals: Vec::new(),
        };
        let mut batch = build(
            Tanks::at(1.0, 0.3),
            declaration(Vec::new(), false, Vec::new()),
            &[],
            vec![gate],
            operating,
        )?;
        let mut ends = Vec::new();
        let mut moved = false;
        while batch.time() < 4.5 {
            let start = batch.time();
            batch = step(&batch, 0.75, &[])?;
            check(&batch)?;
            let end = batch.time();
            ends.push(end);
            assert!(
                knots.iter().all(|&(knot, _)| !(start < knot && knot < end)),
                "[{start},{end}] holds a knot"
            );
            let (detail, point, signed) = operation(&batch, "gate")?;
            let OperatingSourceIntervalDetail::PrescribedGate {
                start_opening_m,
                end_opening_m,
                opening_m,
                closed,
                ..
            } = detail
            else {
                return Err(format!("expected a prescribed gate, found {detail:?}").into());
            };
            let (a, b) = (opening(start)?, opening(end)?);
            assert_eq!(start_opening_m.to_bits(), a.to_bits());
            assert_eq!(end_opening_m.to_bits(), b.to_bits());
            let expected = if closed { 0.0 } else { (a + b) * 0.5 };
            assert_eq!(opening_m.to_bits(), expected.to_bits(), "[{start},{end}]");
            assert_eq!(point, StructureOperatingPoint::GateOpening { opening_m });
            assert_eq!(closed, a == 0.0 && b == 0.0);
            if closed {
                assert_eq!(signed.to_bits(), 0.0f64.to_bits(), "[{start},{end}]");
            }
            moved |= signed > 0.0;
        }
        assert!(ends.contains(&1.0) && ends.contains(&3.0), "{ends:?}");
        assert!(moved);
        Ok(())
    }

    /// A pump lifts a sump with a steady inflow onto a dry surface cell through a free outlet.
    /// Its relay decides every 2 s: it starts at or above 0.6 m and stops at or below 0.4 m.
    /// Every request asks for 1 s.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every decision time ends an interval.
    /// - Each decision latches by the relay rule applied to the sump stage the candidate reached
    ///   at that decision.
    /// - Each interval holds the operation of the latch accepted at its start, in its receipt and
    ///   in its law. So each command switches at its decision time exactly.
    /// - The pump starts at least once and stops at least once.
    /// - A stopped interval moves `0.0` bit for bit. A running interval moves sump water.
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    #[test]
    fn a_controlled_pump_switches_at_its_decision_times() -> TestResult {
        let mut pump = link(
            "pump",
            pump_law(0.1, Some(1.0))?,
            DirectionPolicy::AtoBOnly,
            StructureOperatingPoint::PumpStoppedIsolated,
            (pool("sump"), 0.0),
            (surface(), 0.0),
        );
        pump.operation = SurfaceChannelOperationOwner::ControlledPump;
        let structures = vec![pump];
        let decisions: Vec<f64> = (1..=20).map(|k| 2.0 * f64::from(k)).collect();
        let child = controlled_pump("pump", [0.4, 0.6], (0.0, &decisions, 40.0));
        let operating = SurfaceChannelOperating {
            gates: None,
            pumps: Some(child),
            trajectories: Vec::new(),
            signals: vec![signal("pump", pool("sump"))],
        };
        let pools = declaration(
            vec![pond("sump", 0.0, 4.0, 2.0)?],
            false,
            vec![inflow_binding("feed", "sump")],
        );
        let mut batch = build(
            Tanks::at(0.0, 0.2),
            pools,
            &[0.44 * 4.0],
            structures,
            operating,
        )?;
        let forcing = PoolForcing {
            sources: None,
            inflows: vec![inflow_input("feed", 0.06)],
        };
        let latch = |batch: &Batch| -> TestResult<ControlLatch> {
            let operating = batch.operating.as_ref().ok_or("operating children")?;
            let pump = operating.pump().ok_or("the pump child")?;
            Ok(pump.controls().states()?[0].latch)
        };
        let (mut started, mut stopped) = (false, false);
        let mut ends = Vec::new();
        while batch.time() < 32.0 {
            let start = batch.time();
            let before = latch(&batch)?;
            batch = advance(&batch, 1.0, &forcing)?;
            let end = batch.time();
            ends.push(end);
            pool_check(&batch, &[0.06 * (end - start)])?;
            let (detail, point, signed) = operation(&batch, "pump")?;
            let held = match before {
                ControlLatch::Low => StructureOperatingPoint::PumpStoppedIsolated,
                ControlLatch::High => StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            };
            let OperatingSourceIntervalDetail::ControlledPump {
                operating_point, ..
            } = detail
            else {
                return Err(format!("expected a controlled pump, found {detail:?}").into());
            };
            assert_eq!((operating_point, point), (held, held), "[{start},{end}]");
            match held {
                StructureOperatingPoint::PumpStoppedIsolated => {
                    assert_eq!(signed.to_bits(), 0.0f64.to_bits(), "[{start},{end}]");
                }
                _ => assert!(signed > 0.0, "[{start},{end}] {signed}"),
            }
            let after = latch(&batch)?;
            if decisions.contains(&end) {
                let stage = stages(&batch)?[0];
                let expected = if stage >= 0.6 {
                    ControlLatch::High
                } else if stage <= 0.4 {
                    ControlLatch::Low
                } else {
                    before
                };
                assert_eq!(after, expected, "decision at {end}, stage={stage}");
                started |= before == ControlLatch::Low && after == ControlLatch::High;
                stopped |= before == ControlLatch::High && after == ControlLatch::Low;
            } else {
                assert_eq!(after, before, "no decision at {end}");
            }
        }
        assert!(
            decisions
                .iter()
                .filter(|&&t| t <= 32.0)
                .all(|t| ends.contains(t)),
            "{ends:?}"
        );
        assert!(started && stopped, "started={started} stopped={stopped}");
        Ok(())
    }
}
