//! Typed instantaneous structure laws and held-head proposals without storage or donor acceptance.

mod actuator;
mod applicability;
mod batch;
pub(crate) mod blockage;
mod control;
pub(crate) mod coupling;
mod culvert;
pub(crate) mod definition;
mod error;
mod gate;
mod heads;
mod held;
pub(crate) mod identity;
#[cfg(test)]
pub(crate) mod interval_fixture;
mod numerics;
mod operation;
mod operation_projection;
mod output;
mod parameters;
mod projection;
mod proposal;
mod pump;
pub(crate) mod tensor;

#[doc(hidden)]
pub use actuator::GateActuatorPreparedInterval;
pub use actuator::{
    GATE_ACTUATOR_CONTRACT_VERSION, GATE_ACTUATOR_SNAPSHOT_SCHEMA, GateActuatorAccuracy,
    GateActuatorAnchor, GateActuatorArrivalCertificate, GateActuatorBatch,
    GateActuatorBinaryWitness, GateActuatorCompletionReceipt, GateActuatorDefinition,
    GateActuatorDirection, GateActuatorEndpointReceipt, GateActuatorEvent,
    GateActuatorFmaStageReceipt, GateActuatorForwardReceipt, GateActuatorInitialState,
    GateActuatorMotionReceipt, GateActuatorOutput, GateActuatorPhase, GateActuatorPosition,
    GateActuatorSchedule, GateActuatorSnapshot, GateActuatorSnapshotData, GateActuatorState,
    GateActuatorStepDiagnostics, GateActuatorStepReceipt, GateActuatorSupport,
    GateActuatorTargetChangeReceipt, GateActuatorTimeExpansion, GateActuatorTrial,
    GateTargetUpdate, gate_actuator_capabilities,
};
pub use applicability::{CulvertApplicability, OrificeApplicability, WeirApplicability};
pub use batch::StructureBatch;
pub use blockage::{
    BlockageFraction, BlockageGeometry, StructureBlockage, StructureCondition,
    StructureConditionKind, StructureGeometry, StructureGeometryReport,
};
pub(crate) use culvert::CulvertSide;
pub use culvert::{
    BarrelDepthStatus, BarrelState, CriticalDepthStatus, CulvertAlignment, CulvertBarrel,
    CulvertBarrelBatch, CulvertBarrelDepth, CulvertBarrelDepthRow, CulvertBarrelGeometry,
    CulvertBarrelGeometryRow, CulvertBarrelShape, CulvertEnd, CulvertEnds, CulvertFloor,
    CulvertInlet, CulvertInletBatch, CulvertInletForm, CulvertInletHeadwater,
    CulvertInletHeadwaterRow, CulvertInletRow, CulvertOutletBatch, CulvertOutletHeadwater,
    CulvertOutletHeadwaterRow, CulvertOutletJump, CulvertOutletJumpRow, CulvertOutletRow,
    CulvertParameters, InletRegime, JumpStatus, NormalDepthStatus, OutletRegime,
    culvert_barrel_capabilities, culvert_capabilities, culvert_inlet_capabilities,
    culvert_outlet_capabilities,
};
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
pub use numerics::culvert::{BarrelClient, CulvertClient};
pub use operation::{StructureOperatingPoint, StructureOperation};
pub use output::{
    HeldHeadInterval, HeldHeadVolumeOutput, StructureHydraulicOperands, StructureLawKind,
    StructureOutput, StructureRegime, StructureStatus,
};
pub use parameters::{SubmergedOrificeParameters, SuppressedRectangularWeirParameters};
pub use proposal::StructureProposal;
pub use pump::{
    PumpApplicability, PumpCurve, PumpCurveKnot, PumpFreeOutlet, PumpHydraulicOperands,
    PumpParameters, PumpSpeedPolicy,
};

pub use control::{
    ControlCommand, ControlCommandOutput, ControlDecisionReceipt, ControlDecisionSchedule,
    ControlEvent, ControlInitialState, ControlLatch, ControlPending, ControlSample,
    ControlStageSignal, ControlState, ControlStepDiagnostics, ControlTimeExpansion,
    ControlTimerWitness, ControlTransitionReceipt, OPERATING_CONTROL_CONTRACT_VERSION,
    OPERATING_CONTROL_SNAPSHOT_SCHEMA, OperatingControlBatch, OperatingControlDefinition,
    OperatingControlSnapshot, OperatingControlSnapshotData, OperatingControlTrial,
    SampledControlTiming, operating_control_capabilities,
};

mod operating_source;
pub use operating_source::{
    OperatingSourceBatch, OperatingSourceBinding, OperatingSourceDetail, OperatingSourceEvent,
    OperatingSourceEventKind, OperatingSourceEventOrigin, OperatingSourceFrame,
    OperatingSourceOutput, OperatingSourceProposal, OperatingSourceReceipt, OperatingSourceSupport,
};
#[doc(hidden)]
pub use operating_source::{
    OperatingSourceInterval, OperatingSourceIntervalDetail, OperatingSourceIntervalFrame,
    OperatingSourceIntervalReceipt,
};

pub(crate) use actuator::{
    project_actuator_phase, project_actuator_position, same_actuator_context,
};

mod controlled_gate;
#[doc(hidden)]
pub use controlled_gate::ControlledGatePreparedInterval;
pub use controlled_gate::{
    CONTROLLED_GATE_CONTRACT_VERSION, CONTROLLED_GATE_SNAPSHOT_SCHEMA, ControlledGateBatch,
    ControlledGateEvent, ControlledGateSnapshot, ControlledGateSnapshotData, ControlledGateTrial,
    controlled_gate_capabilities,
};

mod controlled_pump;
pub(crate) use control::same_control_context;
#[doc(hidden)]
pub use controlled_pump::ControlledPumpPreparedInterval;
pub(crate) use controlled_pump::same_controlled_pump_context;
pub use controlled_pump::{
    CONTROLLED_PUMP_CONTRACT_VERSION, CONTROLLED_PUMP_SNAPSHOT_SCHEMA, ControlledPumpBatch,
    ControlledPumpOutput, ControlledPumpSnapshot, ControlledPumpSnapshotData, ControlledPumpTrial,
    PumpResponseDefinition, PumpResponseModel, controlled_pump_capabilities,
};
