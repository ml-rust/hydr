//! Operating declarations of a setup and the fresh children the coordinator builds from them.

use super::{OperatingBatches, child_error};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::structures::SurfaceChannelNode;
use crate::structures::{
    ControlDecisionSchedule, ControlInitialState, ControlledGateBatch, ControlledPumpBatch,
    GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule,
    GateTrajectoryAssignment, OperatingControlDefinition, PumpResponseDefinition,
    SampledControlTiming, StructureBatch, StructureElement,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;

/// One controller signal bound to the node whose stage it samples.
///
/// At each decision the signal samples the node's stage in the converged candidate. Observation
/// and availability time equal the decision time. A dry node at a decision rejects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurfaceChannelSignal {
    /// Identity of the controller in its operating child.
    pub controller_id: String,
    /// Identity of the controller's stage signal.
    pub signal_id: String,
    /// Node whose stage the signal samples: a pool, or a surface or channel cell a link joins.
    pub node: SurfaceChannelNode,
}

/// Original model of one sampled controller child.
///
/// The arguments of [`StructureBatch::with_operating_controls`]. The coordinator binds them to its
/// structure links in ascending identity.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelControls {
    /// Ordered controller definitions.
    pub definitions: Vec<OperatingControlDefinition>,
    /// Initial memory of every controller.
    pub initial_states: Vec<ControlInitialState>,
    /// Decision calendar. Its initial time equals the common initial time.
    pub schedule: ControlDecisionSchedule,
    /// Sampled applicability and timing policy.
    pub timing: SampledControlTiming,
}

/// Original model of one gate actuator child.
///
/// The arguments of [`StructureBatch::with_gate_actuators`].
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelActuators {
    /// Ordered actuator definitions, one per actuated gate.
    pub definitions: Vec<GateActuatorDefinition>,
    /// Initial actual openings and targets.
    pub initial_states: Vec<GateActuatorInitialState>,
    /// Target calendar. Its initial time equals the common initial time.
    pub schedule: GateActuatorSchedule,
}

/// Sampled controllers and the actuators of every `Actuated` gate.
///
/// The coordinator builds one [`ControlledGateBatch`] from them.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelControlledGates {
    /// One controller per actuated gate.
    pub controls: SurfaceChannelControls,
    /// One actuator per actuated gate.
    pub actuators: SurfaceChannelActuators,
}

/// Sampled controllers and the response models of every `ControlledPump` pump.
///
/// The coordinator builds one [`ControlledPumpBatch`] from them.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelControlledPumps {
    /// One controller per controlled pump.
    pub controls: SurfaceChannelControls,
    /// One response model per controlled pump.
    pub responses: Vec<PumpResponseDefinition>,
}

/// Operating children of a coordinator, declared as data, with their signal bindings.
///
/// Construction builds each child fresh at the common initial time on the setup's structure
/// links in ascending identity. Each dynamically owned structure belongs to exactly one child:
/// `Prescribed` gates to `trajectories`, `Actuated` gates to `gates`, `ControlledPump` pumps to
/// `pumps`. Every controller signal of `gates` and `pumps` has exactly one binding in `signals`.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelOperating {
    /// Controlled gates, or `None` without an `Actuated` owner.
    pub gates: Option<SurfaceChannelControlledGates>,
    /// Controlled pumps, or `None` without a `ControlledPump` owner.
    pub pumps: Option<SurfaceChannelControlledPumps>,
    /// One linear opening path per `Prescribed` gate. Empty without one.
    pub trajectories: Vec<GateTrajectoryAssignment>,
    /// One binding per controller signal of `gates` and `pumps`.
    pub signals: Vec<SurfaceChannelSignal>,
}

/// Builds fresh children of `declared` on `elements`, the setup's structure links in ascending
/// identity, on `client`.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming the child for every child rejection.
pub(in super::super) fn build_children<R: Runtime<DType = DType>>(
    client: R::Client,
    elements: Vec<StructureElement>,
    declared: &SurfaceChannelOperating,
) -> Result<OperatingBatches<R>, SurfaceChannelError>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let structures =
        StructureBatch::<R>::new(client, elements).map_err(child_error("structures"))?;
    let controls = |child: &'static str, declared: &SurfaceChannelControls| {
        structures
            .with_operating_controls(
                declared.definitions.clone(),
                declared.initial_states.clone(),
                declared.schedule.clone(),
                declared.timing.clone(),
            )
            .map_err(child_error(child))
    };
    let gate = match &declared.gates {
        Some(gates) => {
            let child = "controlled-gate";
            let actuators = structures
                .with_gate_actuators(
                    gates.actuators.definitions.clone(),
                    gates.actuators.initial_states.clone(),
                    gates.actuators.schedule.clone(),
                )
                .map_err(child_error(child))?;
            Some(
                ControlledGateBatch::new(controls(child, &gates.controls)?, actuators)
                    .map_err(child_error(child))?,
            )
        }
        None => None,
    };
    let pump = match &declared.pumps {
        Some(pumps) => {
            let child = "controlled-pump";
            Some(
                ControlledPumpBatch::new(
                    controls(child, &pumps.controls)?,
                    pumps.responses.clone(),
                )
                .map_err(child_error(child))?,
            )
        }
        None => None,
    };
    let trajectory = if declared.trajectories.is_empty() {
        None
    } else {
        Some(
            structures
                .with_gate_trajectory_subset(declared.trajectories.clone())
                .map_err(child_error("trajectory"))?,
        )
    };
    Ok(OperatingBatches {
        gate,
        pump,
        trajectory,
    })
}
