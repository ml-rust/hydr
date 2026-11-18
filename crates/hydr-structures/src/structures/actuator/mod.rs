//! Explicit ideal finite-rate gate response with checked numerical continuation.

mod capabilities;
mod definition;
mod event;
mod package;
mod position;
mod receipts;
mod state;
pub use capabilities::gate_actuator_capabilities;
pub use definition::{
    GateActuatorAccuracy, GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule,
    GateTargetUpdate,
};
pub use event::GateActuatorEvent;
pub use package::{
    GATE_ACTUATOR_CONTRACT_VERSION, GATE_ACTUATOR_SNAPSHOT_SCHEMA, GateActuatorSnapshotData,
};
pub use position::{
    GateActuatorAnchor, GateActuatorOutput, GateActuatorPhase, GateActuatorPosition,
    GateActuatorSupport, GateActuatorTimeExpansion,
};
pub use receipts::{
    GateActuatorArrivalCertificate, GateActuatorBinaryWitness, GateActuatorCompletionReceipt,
    GateActuatorDirection, GateActuatorEndpointReceipt, GateActuatorFmaStageReceipt,
    GateActuatorForwardReceipt, GateActuatorMotionReceipt,
};
pub use state::{
    GateActuatorState, GateActuatorStepDiagnostics, GateActuatorStepReceipt,
    GateActuatorTargetChangeReceipt,
};

mod batch;
mod interval;
mod trial;
pub use batch::GateActuatorBatch;
#[doc(hidden)]
pub use interval::GateActuatorPreparedInterval;
pub use trial::GateActuatorTrial;

mod projection;
mod projection_values;
mod snapshot;
pub use snapshot::GateActuatorSnapshot;

mod compatibility;
pub(crate) use compatibility::same_context as same_actuator_context;
pub(crate) use projection_values::{
    phase as project_actuator_phase, position as project_actuator_position,
};
