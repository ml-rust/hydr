//! Operating-child fixtures: relays, actuators, trajectories, and coordinators that bind them.
//!
//! The coordinator builds every child on its structure links in ascending identity.

use super::{
    SurfaceChannelActuators, SurfaceChannelControlledGates, SurfaceChannelControlledPumps,
    SurfaceChannelControls, SurfaceChannelOperating, SurfaceChannelSignal,
};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::pools::SurfaceChannelPools;
use crate::coupling::surface_channels::pools::scenario::with_nodes;
use crate::coupling::surface_channels::structures::SurfaceChannelNode;
use crate::coupling::surface_channels::structures::SurfaceChannelStructure;
use crate::coupling::surface_channels::structures::scenario::{
    Batch, DATUM, Tanks, TestResult, client,
};
use crate::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlStageSignal,
    GateActuatorAccuracy, GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule,
    GateMotionLimits, GateOpeningKnot, GateOpeningTrajectory, GateTrajectoryAccuracy,
    GateTrajectoryAssignment, OperatingControlDefinition, PumpResponseDefinition,
    PumpResponseModel, SampledControlTiming,
};

/// Debug text: equal text means equal bits for every finite field.
pub fn bits<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
}

/// Controller identity of structure `id`.
pub fn controller(id: &str) -> String {
    format!("relay-{id}")
}

/// Signal identity of structure `id`'s controller.
pub fn signal_id(id: &str) -> String {
    format!("signal-{id}")
}

/// The signal of structure `id`'s controller bound to `node`.
pub fn signal(id: &str, node: SurfaceChannelNode) -> SurfaceChannelSignal {
    SurfaceChannelSignal {
        controller_id: controller(id),
        signal_id: signal_id(id),
        node,
    }
}

/// Two-threshold relay on structure `id`, starting low at `start`, with decisions at `times`.
///
/// The stage at a decision latches high at or above `upper` and low at or below `lower`.
pub fn relay(
    id: &str,
    [lower, upper]: [f64; 2],
    [low, high]: [ControlCommand; 2],
    (start, times, end): (f64, &[f64], f64),
) -> SurfaceChannelControls {
    SurfaceChannelControls {
        definitions: vec![OperatingControlDefinition {
            controller_id: controller(id),
            structure_id: id.into(),
            signal: ControlStageSignal {
                signal_id: signal_id(id),
                support_id: format!("support-{id}"),
                datum: DATUM.into(),
            },
            lower_stage_m: lower,
            upper_stage_m: upper,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: low,
            high_command: high,
            provenance: "Synthetic sampled relay".into(),
        }],
        initial_states: vec![ControlInitialState {
            controller_id: controller(id),
            latch: ControlLatch::Low,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        schedule: ControlDecisionSchedule {
            initial_time_s: start,
            decision_times_s: times.to_vec(),
            end_s: end,
        },
        timing: SampledControlTiming {
            maximum_decision_gap_s: end - start,
            maximum_timer_lateness_s: end - start,
            sampled_applicability: "Synthetic sampled policy".into(),
        },
    }
}

/// Actuator on gate `id` at `rate` m/s each way, closed at `start`, with targets at `times`.
pub fn actuator(
    id: &str,
    rate: f64,
    (start, times, end): (f64, &[f64], f64),
) -> SurfaceChannelActuators {
    SurfaceChannelActuators {
        definitions: vec![GateActuatorDefinition {
            actuator_id: format!("actuator-{id}"),
            structure_id: id.into(),
            opening_rate_m_per_s: rate,
            closing_rate_m_per_s: rate,
            accuracy: GateActuatorAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
                maximum_arrival_window_s: 1e-10,
            },
            rate_provenance: "Synthetic rates".into(),
            ideal_response_applicability: "Ideal rate response".into(),
        }],
        initial_states: vec![GateActuatorInitialState {
            actuator_id: format!("actuator-{id}"),
            opening_m: 0.0,
            target_opening_m: 0.0,
        }],
        schedule: GateActuatorSchedule {
            initial_time_s: start,
            target_times_s: times.to_vec(),
            end_s: end,
        },
    }
}

/// Controlled gate `id`: closed when low, `target` m open when high, starting closed and low.
pub fn controlled_gate(
    id: &str,
    thresholds: [f64; 2],
    (target, rate): (f64, f64),
    schedule: (f64, &[f64], f64),
) -> SurfaceChannelControlledGates {
    let commands = [
        ControlCommand::GateTarget { opening_m: 0.0 },
        ControlCommand::GateTarget { opening_m: target },
    ];
    SurfaceChannelControlledGates {
        controls: relay(id, thresholds, commands, schedule),
        actuators: actuator(id, rate, schedule),
    }
}

/// Controlled pump `id`: stopped when low, running at reference speed when high.
pub fn controlled_pump(
    id: &str,
    thresholds: [f64; 2],
    schedule: (f64, &[f64], f64),
) -> SurfaceChannelControlledPumps {
    let commands = [
        ControlCommand::PumpStoppedIsolated,
        ControlCommand::PumpRunning { speed_ratio: 1.0 },
    ];
    SurfaceChannelControlledPumps {
        controls: relay(id, thresholds, commands, schedule),
        responses: vec![PumpResponseDefinition {
            structure_id: id.into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "Synthetic instantaneous operation".into(),
            ideal_response_applicability: "No startup delay".into(),
        }],
    }
}

/// Linear opening path of gate `id` through `(time, opening)` knots.
pub fn trajectory(id: &str, knots: &[(f64, f64)]) -> TestResult<GateTrajectoryAssignment> {
    let trajectory = GateOpeningTrajectory::new(
        knots
            .iter()
            .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
            .collect(),
        GateMotionLimits {
            maximum_opening_speed_m_per_s: f64::MAX,
            maximum_closing_speed_m_per_s: f64::MAX,
        },
        GateTrajectoryAccuracy {
            maximum_absolute_opening_error_m: 1e-12,
            maximum_relative_opening_error: 1e-12,
        },
        "Synthetic linear trajectory".into(),
    )?;
    Ok(GateTrajectoryAssignment {
        structure_id: id.into(),
        trajectory,
    })
}

/// Operating children with only `signals` bound.
pub fn operating(signals: Vec<SurfaceChannelSignal>) -> SurfaceChannelOperating {
    SurfaceChannelOperating {
        gates: None,
        pumps: None,
        trajectories: Vec::new(),
        signals,
    }
}

/// The coordinator of `tanks` and `pools` at `volumes`, joined by `structures`, with `operating`.
pub fn build(
    tanks: Tanks,
    pools: SurfaceChannelPools,
    volumes: &[f64],
    structures: Vec<SurfaceChannelStructure>,
    operating: SurfaceChannelOperating,
) -> Result<Batch, SurfaceChannelError> {
    let parts = tanks
        .parts()
        .map_err(|error| SurfaceChannelError::Input(error.to_string()))?;
    let (mut setup, initial) = with_nodes(parts, pools, volumes, structures);
    setup.operating = Some(operating);
    Batch::new(client(), setup, initial)
}

/// Debug text of every accepted operating child: time, counters, states, outputs, last step.
pub fn children(batch: &Batch) -> TestResult<String> {
    let operating = batch.operating.as_ref().ok_or("operating children")?;
    let mut text = String::new();
    if let Some(gate) = operating.gate() {
        text.push_str(&bits(&(
            gate.time().to_bits(),
            gate.accepted_steps(),
            gate.controls().states()?,
            gate.controls().last_step()?,
            gate.actuators().states()?,
            gate.actuators().outputs()?,
            gate.actuators().last_step()?,
        )));
    }
    if let Some(pump) = operating.pump() {
        text.push_str(&bits(&(
            pump.time().to_bits(),
            pump.accepted_steps(),
            pump.controls().states()?,
            pump.controls().last_step()?,
            pump.outputs()?,
        )));
    }
    Ok(text)
}
