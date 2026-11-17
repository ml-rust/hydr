//! Typed instantaneous structure laws and held-head proposals without storage or donor acceptance.

mod actuator;
mod applicability;
mod batch;
mod control;
pub(crate) mod coupling;
pub(crate) mod definition;
mod error;
mod gate;
mod heads;
mod held;
pub(crate) mod identity;
mod numerics;
mod operation;
mod operation_projection;
mod output;
mod parameters;
mod projection;
mod proposal;
mod pump;
pub(crate) mod tensor;

pub use actuator::{
    GateActuatorAccuracy, GateActuatorAnchor, GateActuatorArrivalCertificate, GateActuatorBatch,
    GateActuatorBinaryWitness, GateActuatorCompletionReceipt, GateActuatorDefinition,
    GateActuatorDirection, GateActuatorEndpointReceipt, GateActuatorEvent,
    GateActuatorFmaStageReceipt, GateActuatorForwardReceipt, GateActuatorInitialState,
    GateActuatorMotionReceipt, GateActuatorOutput, GateActuatorPhase, GateActuatorPosition,
    GateActuatorSchedule, GateActuatorSnapshot, GateActuatorSnapshotData, GateActuatorState,
    GateActuatorStepDiagnostics, GateActuatorStepReceipt, GateActuatorSupport,
    GateActuatorTargetChangeReceipt, GateActuatorTimeExpansion, GateActuatorTrial,
    GateTargetUpdate,
};
pub use applicability::{OrificeApplicability, WeirApplicability};
pub use batch::StructureBatch;
pub use definition::{DirectionPolicy, FlowDirection, StructureElement, StructureLaw};
pub use error::{StructureError, StructureInputErrorKind};
pub(crate) use gate::GateTrajectoryEvaluationData;
pub use gate::{
    GateApplicability, GateHydraulicOperands, GateMotionLimits, GateOpeningKnot,
    GateOpeningTrajectory, GateParameters, GateTrajectoryAccuracy, GateTrajectoryAssignment,
    GateTrajectoryBatch, GateTrajectoryEvaluation, GateTrajectoryEvent, GateTrajectoryInterval,
    GateTrajectoryOutput, GateTrajectorySupport,
};
pub use heads::{EndpointHead, HeadConvention, HeadSupport, StructureHeads};
pub use held::HeldHeadProposal;
pub use operation::{StructureOperatingPoint, StructureOperation};
pub use output::{
    HeldHeadInterval, HeldHeadVolumeOutput, StructureHydraulicOperands, StructureLawKind,
    StructureOutput, StructureRegime, StructureStatus,
};
pub use parameters::{SubmergedOrificeParameters, SuppressedRectangularWeirParameters};
pub use proposal::StructureProposal;
pub use pump::{
    PumpApplicability, PumpCurve, PumpCurveKnot, PumpHydraulicOperands, PumpParameters,
    PumpSpeedPolicy,
};

pub use control::{
    ControlCommand, ControlCommandOutput, ControlDecisionReceipt, ControlDecisionSchedule,
    ControlEvent, ControlInitialState, ControlLatch, ControlPending, ControlSample,
    ControlStageSignal, ControlState, ControlStepDiagnostics, ControlTimeExpansion,
    ControlTimerWitness, ControlTransitionReceipt, OperatingControlBatch,
    OperatingControlDefinition, OperatingControlSnapshot, OperatingControlSnapshotData,
    OperatingControlTrial, SampledControlTiming,
};

mod operating_source;
pub use operating_source::{
    OperatingSourceBatch, OperatingSourceBinding, OperatingSourceDetail, OperatingSourceEvent,
    OperatingSourceEventKind, OperatingSourceEventOrigin, OperatingSourceFrame,
    OperatingSourceOutput, OperatingSourceProposal, OperatingSourceReceipt, OperatingSourceSupport,
};

pub(crate) use actuator::{
    project_actuator_phase, project_actuator_position, same_actuator_context,
};

mod controlled_gate;
pub use controlled_gate::{
    ControlledGateBatch, ControlledGateEvent, ControlledGateSnapshot, ControlledGateSnapshotData,
    ControlledGateTrial,
};

mod controlled_pump;
pub(crate) use control::same_control_context;
pub(crate) use controlled_pump::same_controlled_pump_context;
pub use controlled_pump::{
    ControlledPumpBatch, ControlledPumpOutput, ControlledPumpSnapshot, ControlledPumpSnapshotData,
    ControlledPumpTrial, PumpResponseDefinition, PumpResponseModel,
};
