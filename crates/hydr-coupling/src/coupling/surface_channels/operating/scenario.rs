//! Operating-child fixtures: relays, actuators, trajectories, and coordinators that bind them.
//!
//! Every child sits on the coordinator's structure links in ascending identity.

use super::{SurfaceChannelOperating, SurfaceChannelSignal};
use crate::coupling::surface_channels::SurfaceChannelDefinition;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::pools::SurfaceChannelPools;
use crate::coupling::surface_channels::structures::SurfaceChannelNode;
use crate::coupling::surface_channels::structures::SurfaceChannelStructure;
use crate::coupling::surface_channels::structures::scenario::{
    Batch, DATUM, Tanks, TestResult, client,
};
use crate::storage::LevelPoolState;
use crate::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlStageSignal,
    ControlledGateBatch, ControlledPumpBatch, GateActuatorAccuracy, GateActuatorBatch,
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule, GateMotionLimits,
    GateOpeningKnot, GateOpeningTrajectory, GateTrajectoryAccuracy, GateTrajectoryAssignment,
    GateTrajectoryBatch, OperatingControlBatch, OperatingControlDefinition, PumpResponseDefinition,
    PumpResponseModel, SampledControlTiming, StructureBatch,
};
use numr::runtime::cpu::CpuRuntime;

pub type Cpu = CpuRuntime;

/// Debug text: equal text means equal bits for every finite field.
pub fn bits<T: std::fmt::Debug>(value: &T) -> String {
    format!("{value:?}")
}

/// Structure batch of `structures` in ascending identity, the context of every child.
pub fn context(structures: &[SurfaceChannelStructure]) -> TestResult<StructureBatch<Cpu>> {
    let mut elements: Vec<_> = structures.iter().map(|s| s.element.clone()).collect();
    elements.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(StructureBatch::<Cpu>::new(client(), elements)?)
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
    context: &StructureBatch<Cpu>,
    id: &str,
    [lower, upper]: [f64; 2],
    [low, high]: [ControlCommand; 2],
    (start, times, end): (f64, &[f64], f64),
) -> TestResult<OperatingControlBatch<Cpu>> {
    Ok(context.with_operating_controls(
        vec![OperatingControlDefinition {
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
        vec![ControlInitialState {
            controller_id: controller(id),
            latch: ControlLatch::Low,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        ControlDecisionSchedule {
            initial_time_s: start,
            decision_times_s: times.to_vec(),
            end_s: end,
        },
        SampledControlTiming {
            maximum_decision_gap_s: end - start,
            maximum_timer_lateness_s: end - start,
            sampled_applicability: "Synthetic sampled policy".into(),
        },
    )?)
}

/// Actuator on gate `id` at `rate` m/s each way, closed at `start`, with targets at `times`.
pub fn actuator(
    context: &StructureBatch<Cpu>,
    id: &str,
    rate: f64,
    (start, times, end): (f64, &[f64], f64),
) -> TestResult<GateActuatorBatch<Cpu>> {
    Ok(context.with_gate_actuators(
        vec![GateActuatorDefinition {
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
        vec![GateActuatorInitialState {
            actuator_id: format!("actuator-{id}"),
            opening_m: 0.0,
            target_opening_m: 0.0,
        }],
        GateActuatorSchedule {
            initial_time_s: start,
            target_times_s: times.to_vec(),
            end_s: end,
        },
    )?)
}

/// Controlled gate `id`: closed when low, `target` m open when high, starting closed and low.
pub fn controlled_gate(
    context: &StructureBatch<Cpu>,
    id: &str,
    thresholds: [f64; 2],
    (target, rate): (f64, f64),
    schedule: (f64, &[f64], f64),
) -> TestResult<ControlledGateBatch<Cpu>> {
    let commands = [
        ControlCommand::GateTarget { opening_m: 0.0 },
        ControlCommand::GateTarget { opening_m: target },
    ];
    Ok(ControlledGateBatch::new(
        relay(context, id, thresholds, commands, schedule)?,
        actuator(context, id, rate, schedule)?,
    )?)
}

/// Controlled pump `id`: stopped when low, running at reference speed when high.
pub fn controlled_pump(
    context: &StructureBatch<Cpu>,
    id: &str,
    thresholds: [f64; 2],
    schedule: (f64, &[f64], f64),
) -> TestResult<ControlledPumpBatch<Cpu>> {
    let commands = [
        ControlCommand::PumpStoppedIsolated,
        ControlCommand::PumpRunning { speed_ratio: 1.0 },
    ];
    Ok(ControlledPumpBatch::new(
        relay(context, id, thresholds, commands, schedule)?,
        vec![PumpResponseDefinition {
            structure_id: id.into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "Synthetic instantaneous operation".into(),
            ideal_response_applicability: "No startup delay".into(),
        }],
    )?)
}

/// Linear opening path of gate `id` through `(time, opening)` knots.
pub fn trajectory(
    context: &StructureBatch<Cpu>,
    id: &str,
    knots: &[(f64, f64)],
) -> TestResult<GateTrajectoryBatch<Cpu>> {
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
    Ok(
        context.with_gate_trajectory_subset(vec![GateTrajectoryAssignment {
            structure_id: id.into(),
            trajectory,
        }])?,
    )
}

/// Operating children with only `signals` bound.
pub fn operating(signals: Vec<SurfaceChannelSignal>) -> SurfaceChannelOperating<Cpu> {
    SurfaceChannelOperating {
        gate: None,
        pump: None,
        trajectory: None,
        signals,
    }
}

/// The coordinator of `tanks` and `pools` at `volumes`, joined by `structures`, with `operating`.
pub fn build(
    tanks: Tanks,
    pools: SurfaceChannelPools,
    volumes: &[f64],
    structures: Vec<SurfaceChannelStructure>,
    operating: SurfaceChannelOperating<Cpu>,
) -> Result<Batch, SurfaceChannelError> {
    let (setup, initial) = tanks
        .parts()
        .map_err(|error| SurfaceChannelError::Input(error.to_string()))?;
    let definition = SurfaceChannelDefinition::with_nodes(setup, pools, structures)?;
    let states: Vec<LevelPoolState> = volumes
        .iter()
        .map(|&volume_m3| LevelPoolState { volume_m3 })
        .collect();
    Batch::with_operating(client(), definition, initial, &states, operating)
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
