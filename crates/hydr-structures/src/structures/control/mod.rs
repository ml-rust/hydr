//! Sampled operating commands with explicit qualification, transactional trials, and complete continuation.

mod batch;
mod capabilities;
mod command;
mod definition;
mod projection;
mod projection_values;
mod receipts;
mod schedule;
mod signal;
mod snapshot;
mod state;
mod trial;

pub use batch::OperatingControlBatch;
pub use capabilities::operating_control_capabilities;
pub use command::{ControlCommand, ControlCommandOutput, ControlLatch};
pub use definition::OperatingControlDefinition;
pub use receipts::{
    ControlDecisionReceipt, ControlStepDiagnostics, ControlTimeExpansion, ControlTimerWitness,
    ControlTransitionReceipt,
};
pub use schedule::{ControlDecisionSchedule, ControlEvent, SampledControlTiming};
pub use signal::{ControlSample, ControlStageSignal};
pub use snapshot::{
    OPERATING_CONTROL_CONTRACT_VERSION, OPERATING_CONTROL_SNAPSHOT_SCHEMA,
    OperatingControlSnapshot, OperatingControlSnapshotData,
};
pub use state::{ControlInitialState, ControlPending, ControlState};
pub use trial::OperatingControlTrial;

mod compatibility;
pub(crate) use compatibility::same_context as same_control_context;
