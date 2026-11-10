//! Explicit ideal finite-rate gate response with checked numerical continuation.

mod definition;
mod event;
mod package;
mod position;
mod receipts;
mod state;
pub use definition::{
    GateActuatorAccuracy, GateActuatorDefinition, GateActuatorInitialState, GateActuatorSchedule,
    GateTargetUpdate,
};
pub use event::GateActuatorEvent;
pub use package::GateActuatorSnapshotData;
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
mod trial;
pub use batch::GateActuatorBatch;
pub use trial::GateActuatorTrial;

mod projection;
mod projection_values;
mod snapshot;
pub use snapshot::GateActuatorSnapshot;
