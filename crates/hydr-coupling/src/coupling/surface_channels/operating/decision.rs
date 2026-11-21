//! Decision completion: signal samples from the converged candidate, child trials, and the check
//! that each child kept its prepared left state.

use super::interval::{PreparedOperation, StepOperation};
use super::{BoundSignal, OperatingChildren, child_error};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::Ops;
use crate::structures::{ControlSample, ControlledGateTrial, GateActuatorOutput};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Candidate end stages and end wet masks at every donor group, each `[S + C + P]`.
pub(in super::super) struct CandidateNodes<'a, R: Runtime<DType = DType>> {
    pub stage: &'a Tensor<R>,
    pub wet: &'a Tensor<R>,
}

/// Host stages and wet masks of the candidate, read once on the first decision.
struct Read<'a, R: Runtime<DType = DType>> {
    nodes: Option<CandidateNodes<'a, R>>,
    host: Option<(Vec<f64>, Vec<f64>)>,
}

impl<R: Runtime<DType = DType>> Read<'_, R> {
    fn host(&mut self) -> Result<&(Vec<f64>, Vec<f64>), SurfaceChannelError> {
        if self.host.is_none() {
            let nodes = self.nodes.as_ref().ok_or_else(|| {
                SurfaceChannelError::Numerical(
                    "a decision interval lacks its candidate node stages".into(),
                )
            })?;
            self.host = Some((
                nodes.stage.try_to_vec::<f64>()?,
                nodes.wet.try_to_vec::<f64>()?,
            ));
        }
        self.host.as_ref().ok_or_else(|| {
            SurfaceChannelError::Numerical("candidate node stages are unread".into())
        })
    }

    /// One sample per signal of `signals` at decision time `time_s`.
    ///
    /// Observation and availability time equal the decision time. A dry node supplies no sample.
    fn samples(
        &mut self,
        signals: &[BoundSignal],
        time_s: f64,
    ) -> Result<Vec<ControlSample>, SurfaceChannelError> {
        let (stage, wet) = self.host()?;
        signals
            .iter()
            .map(|signal| {
                let missing = || {
                    SurfaceChannelError::Numerical(format!(
                        "signal node={:?} lacks its candidate stage",
                        signal.node
                    ))
                };
                let stage_m = *stage.get(signal.group).ok_or_else(missing)?;
                if *wet.get(signal.group).ok_or_else(missing)? == 0.0 {
                    return Err(SurfaceChannelError::input(format!(
                        "control signal controller={:?} signal={:?} node={:?} is dry at decision time={time_s}: bind the signal to a node that stays wet at its decisions",
                        signal.controller_id, signal.signal_id, signal.node
                    )));
                }
                Ok(ControlSample {
                    controller_id: signal.controller_id.clone(),
                    signal_id: signal.signal_id.clone(),
                    support_id: signal.support_id.clone(),
                    datum: signal.datum.clone(),
                    observed_at_s: time_s,
                    available_at_s: time_s,
                    stage_m,
                })
            })
            .collect()
    }
}

/// Rejects a gate trial whose left positions differ from the prepared ones in any bit.
fn check_left<R: Runtime<DType = DType>>(
    left: &[GateActuatorOutput],
    trial: &ControlledGateTrial<R>,
) -> Result<(), SurfaceChannelError> {
    let step = trial
        .actuators()
        .last_step()
        .map_err(child_error("controlled-gate"))?
        .ok_or_else(|| {
            SurfaceChannelError::Numerical("a controlled-gate trial lacks its step".into())
        })?;
    let bits = |p: &crate::structures::GateActuatorPosition| {
        [
            p.opening_m.to_bits(),
            p.lower_opening_m.to_bits(),
            p.upper_opening_m.to_bits(),
            p.absolute_error_m.to_bits(),
        ]
    };
    let same = left.len() == step.receipts.len()
        && left.iter().zip(&step.receipts).all(|(prepared, receipt)| {
            prepared.actuator_id == receipt.actuator_id
                && bits(&prepared.position) == bits(&receipt.left_position)
        });
    if same {
        Ok(())
    } else {
        Err(SurfaceChannelError::Numerical(
            "controlled-gate left positions differ from the prepared interval operation".into(),
        ))
    }
}

impl<R: Runtime<DType = DType>> OperatingChildren<R>
where
    R::Client: Ops<R>,
{
    /// Completes every child of `prepared` at `end` from the converged candidate `nodes`.
    ///
    /// A decision end samples every signal of its child. Other ends take no samples.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] for a dry signal node at a decision, naming the
    /// signal and node, and for every child rejection, naming the child.
    pub fn complete(
        &self,
        prepared: PreparedOperation<R>,
        end: f64,
        nodes: Option<CandidateNodes<'_, R>>,
        points: &[crate::structures::StructureOperatingPoint],
    ) -> Result<(Self, StepOperation<R>), SurfaceChannelError> {
        let PreparedOperation {
            gate,
            pump,
            interval,
        } = prepared;
        let mut read = Read { nodes, host: None };
        let gate = match (&self.gate, gate) {
            (Some(batch), Some(prepared)) => {
                let decision = prepared
                    .event()
                    .is_some_and(|event| event.decision_index.is_some());
                let samples = if decision {
                    read.samples(&self.binding.gate_signals, end)?
                } else {
                    Vec::new()
                };
                let left = prepared
                    .actuators()
                    .left_outputs()
                    .map_err(child_error("controlled-gate"))?;
                let trial = batch
                    .complete(prepared, &samples)
                    .map_err(child_error("controlled-gate"))?;
                check_left(&left, &trial)?;
                Some(trial.into_next())
            }
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "a prepared interval differs from the controlled-gate child".into(),
                ));
            }
        };
        let pump = match (&self.pump, pump) {
            (Some(batch), Some(prepared)) => {
                let decision = prepared
                    .event()
                    .is_some_and(|event| event.decision_index.is_some());
                let samples = if decision {
                    read.samples(&self.binding.pump_signals, end)?
                } else {
                    Vec::new()
                };
                let trial = batch
                    .complete(prepared, &samples)
                    .map_err(child_error("controlled-pump"))?;
                Some(trial.into_next())
            }
            (None, None) => None,
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "a prepared interval differs from the controlled-pump child".into(),
                ));
            }
        };
        let off = |time: f64| time.to_bits() != end.to_bits();
        if gate.as_ref().is_some_and(|g| off(g.time()))
            || pump.as_ref().is_some_and(|p| off(p.time()))
        {
            return Err(SurfaceChannelError::Numerical(format!(
                "an operating child disagrees on the candidate end={end}"
            )));
        }
        Ok((
            Self {
                binding: Arc::clone(&self.binding),
                gate,
                pump,
            },
            StepOperation {
                interval,
                points: points.to_vec(),
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelOperating;
    use super::super::scenario::{bits, build, children, controlled_gate, signal};
    use crate::coupling::surface_channels::SurfaceChannelError;
    use crate::coupling::surface_channels::pools::scenario::{
        PoolForcing, advance, check, declaration, inflow_binding, inflow_input, link, pond, pool,
        stages, surface,
    };
    use crate::coupling::surface_channels::structures::scenario::{
        Batch, Tanks, TestResult, gate_law,
    };
    use crate::coupling::surface_channels::structures::{
        SurfaceChannelNode, SurfaceChannelOperationOwner,
    };
    use crate::structures::{
        ControlLatch, DirectionPolicy, OperatingSourceIntervalDetail, StructureOperatingPoint,
    };

    /// Decision times of the pond relay, in s.
    const DECISIONS: [f64; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    /// Relay thresholds `[lower, upper]` on the pond stage, in m.
    const THRESHOLDS: [f64; 2] = [0.45, 0.6];
    /// Pond inflow, in m³/s.
    const INFLOW: f64 = 0.12;

    /// A 4 m² pond filled at `INFLOW` from 0.315 m, draining through a controlled outlet gate
    /// onto the dry surface cell. The relay signal samples `node`.
    fn pond_outlet(node: SurfaceChannelNode) -> Result<Batch, Box<dyn std::error::Error>> {
        let mut outlet = link(
            "outlet",
            gate_law(0.0)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            (pool("pond"), 0.0),
            (surface(), 0.0),
        );
        outlet.operation = SurfaceChannelOperationOwner::Actuated;
        let structures = vec![outlet];
        let gate = controlled_gate("outlet", THRESHOLDS, (0.3, 0.1), (0.0, &DECISIONS, 20.0));
        let operating = SurfaceChannelOperating {
            gates: Some(gate),
            pumps: None,
            trajectories: Vec::new(),
            signals: vec![signal("outlet", node)],
        };
        let pools = declaration(
            vec![pond("pond", 0.0, 4.0, 2.0)?],
            false,
            vec![inflow_binding("feed", "pond")],
        );
        Ok(build(
            Tanks::at(0.0, 0.2),
            pools,
            &[0.315 * 4.0],
            structures,
            operating,
        )?)
    }

    fn forcing() -> PoolForcing {
        PoolForcing {
            sources: None,
            inflows: vec![inflow_input("feed", INFLOW)],
        }
    }

    /// Latch and target opening of the accepted outlet child.
    fn relay(batch: &Batch) -> TestResult<(ControlLatch, f64)> {
        let operating = batch.operating.as_ref().ok_or("operating children")?;
        let gate = operating.gate().ok_or("the gate child")?;
        Ok((
            gate.controls().states()?[0].latch,
            gate.actuators().outputs()?[0].target_opening_m,
        ))
    }

    /// The pond outlet opens on its relay once the pond stage reaches the upper threshold.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every decision time ends an interval. Between decisions the latch holds.
    /// - Every decision stage keeps at least 0.01 m from each threshold.
    /// - The latch switches high at the first decision with stage at or above 0.6 m, and stays
    ///   low before it. The actuator target becomes 0.3 m at that decision.
    /// - Up to that decision every interval is closed at both ends and moves `0.0` bit for bit.
    /// - After it, each interval takes the midpoint of its actuator path, bit for bit, opens
    ///   above zero, and moves pond water onto the surface.
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    #[test]
    fn an_actuated_outlet_opens_at_the_first_decision_above_its_threshold() -> TestResult {
        let mut batch = pond_outlet(pool("pond"))?;
        let mut switched = None;
        let mut ends = Vec::new();
        while batch.time() < 12.0 {
            let start = batch.time();
            batch = advance(&batch, 1.0, &forcing())?;
            let end = batch.time();
            ends.push(end);
            check(&batch, &[INFLOW * (end - start)])?;
            let step = batch.last.as_ref().ok_or("an accepted interval")?;
            let signed = step.signed_m3.try_to_vec::<f64>()?[0];
            let operation = step.operation.as_ref().ok_or("an interval operation")?;
            let receipt = operation.interval.receipts()?.remove(0);
            let OperatingSourceIntervalDetail::ActuatedGate {
                start: left,
                end: right,
                opening_m,
                closed,
                ..
            } = receipt.detail
            else {
                return Err(
                    format!("expected an actuated gate, found {:?}", receipt.detail).into(),
                );
            };
            assert_eq!(
                operation.points[0],
                StructureOperatingPoint::GateOpening { opening_m }
            );
            if switched.is_none() {
                assert!(closed, "[{start},{end}]");
                assert_eq!(signed.to_bits(), 0.0f64.to_bits(), "[{start},{end}]");
            } else {
                let midpoint = (left.opening_m + right.opening_m) * 0.5;
                assert_eq!(opening_m.to_bits(), midpoint.to_bits(), "[{start},{end}]");
                assert!(!closed && opening_m > 0.0, "[{start},{end}]");
                assert!(signed > 0.0, "[{start},{end}] {signed}");
            }
            let (latch, target) = relay(&batch)?;
            if !DECISIONS.contains(&end) {
                let expected = if switched.is_some() {
                    ControlLatch::High
                } else {
                    ControlLatch::Low
                };
                assert_eq!(latch, expected, "no decision at {end}");
                continue;
            }
            let stage = stages(&batch)?[0];
            for threshold in THRESHOLDS {
                assert!((stage - threshold).abs() >= 0.01, "stage={stage} at {end}");
            }
            match switched {
                None if stage >= THRESHOLDS[1] => {
                    assert_eq!((latch, target), (ControlLatch::High, 0.3), "at {end}");
                    switched = Some(end);
                }
                None => assert_eq!((latch, target), (ControlLatch::Low, 0.0), "at {end}"),
                Some(_) => assert_eq!(latch, ControlLatch::High, "at {end}"),
            }
        }
        assert!(DECISIONS.iter().all(|t| ends.contains(t)), "{ends:?}");
        let switched = switched.ok_or("the outlet opens")?;
        assert!(switched < 12.0, "{switched}");
        Ok(())
    }

    /// The relay signal samples the dry surface cell.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The first decision rejects without a retry, naming the signal and the node.
    /// - The rejected trial leaves the coordinator and its operating child unchanged bit for bit.
    #[test]
    fn a_dry_signal_node_rejects_its_decision_and_changes_nothing() -> TestResult {
        let mut batch = pond_outlet(surface())?;
        let error = loop {
            let before = (children(&batch)?, batch.time(), batch.accepted_steps());
            match batch.trial_to(1.0, &forcing().apply(step_forcing(&batch, 1.0))) {
                Ok(trial) => {
                    assert!(trial.time() < 1.0, "the decision at 1 s rejects");
                    batch = trial.into_next();
                }
                Err(error) => {
                    let after = (children(&batch)?, batch.time(), batch.accepted_steps());
                    assert_eq!(after, before);
                    break error;
                }
            }
        };
        let SurfaceChannelError::Input(message) = &error else {
            return Err(format!("expected an input rejection, found {error}").into());
        };
        assert!(!error.is_retryable());
        assert!(
            message.contains("signal=\"signal-outlet\"")
                && message.contains("node=\"surface-cell:0\"")
                && message.contains("dry"),
            "{message}"
        );
        Ok(())
    }

    /// Caller reach forcing of the tanks over `[batch.time(), batch.time() + dt]`.
    fn step_forcing(
        batch: &Batch,
        dt: f64,
    ) -> crate::coupling::surface_channels::SurfaceChannelForcingInterval {
        use crate::channel::ChannelBoundary;
        use crate::coupling::surface_channels::structures::scenario::{
            REACH, inflow as reach_inflow, withdrawal,
        };
        use crate::coupling::surface_channels::{
            SurfaceChannelForcingInterval, SurfaceChannelReachForcing,
        };
        let start = batch.time();
        SurfaceChannelForcingInterval {
            start_s: start,
            end_s: start + dt,
            reaches: vec![SurfaceChannelReachForcing {
                reach_id: REACH.into(),
                left: Some(ChannelBoundary::Wall),
                right: Some(ChannelBoundary::Wall),
                lateral_inputs: vec![reach_inflow(0.0), withdrawal(0.0)],
            }],
            pool_sources: None,
            pool_inflows: Vec::new(),
            runoff: Vec::new(),
        }
    }

    /// Trials from one accepted state that reject or that the caller drops.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - A dropped decision trial leaves every operating child bit for bit unchanged.
    /// - A trial whose pool forcing rejects leaves them unchanged too.
    /// - The same trial taken again yields the same operating children and receipt, bit for bit.
    #[test]
    fn rejected_and_dropped_trials_leave_the_operating_children_unchanged() -> TestResult {
        let mut batch = pond_outlet(pool("pond"))?;
        for _ in 0..9 {
            batch = advance(&batch, 1.0, &forcing())?;
        }
        let before = children(&batch)?;
        let reach = step_forcing(&batch, 1.0);
        let end = batch.time() + 1.0;
        let first = batch.trial_to(end, &forcing().apply(reach.clone()))?;
        let accepted = children(&first.next)?;
        let receipt = bits(&first.diagnostics()?);
        assert_ne!(accepted, before);
        drop(first);
        assert_eq!(children(&batch)?, before);
        assert!(
            batch
                .trial_to(end, &PoolForcing::none().apply(reach.clone()))
                .is_err()
        );
        assert_eq!(children(&batch)?, before);
        let again = batch.trial_to(end, &forcing().apply(reach))?;
        assert_eq!(children(&again.next)?, accepted);
        assert_eq!(bits(&again.diagnostics()?), receipt);
        Ok(())
    }
}
