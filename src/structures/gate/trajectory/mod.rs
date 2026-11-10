//! Immutable prescribed gate openings with explicit motion and accuracy policies.

mod accuracy;
mod batch;
mod binding;
mod definition;
mod evaluation;
mod events;

pub use accuracy::{GateMotionLimits, GateTrajectoryAccuracy};
pub use batch::GateTrajectoryBatch;
pub use binding::GateTrajectoryAssignment;
pub use definition::{GateOpeningKnot, GateOpeningTrajectory};
pub(crate) use evaluation::GateTrajectoryEvaluationData;
pub use evaluation::{GateTrajectoryEvaluation, GateTrajectoryOutput};
pub use events::{GateTrajectoryEvent, GateTrajectoryInterval, GateTrajectorySupport};
