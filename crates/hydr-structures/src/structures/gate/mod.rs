//! Checked hydrostatic rectangular apertures and explicit opening support.

mod applicability;
mod operands;
mod parameters;
mod projection;
mod trajectory;

pub use applicability::GateApplicability;
pub use operands::GateHydraulicOperands;
pub use parameters::GateParameters;
pub(crate) use projection::GateOperandProjection;

pub(crate) use trajectory::GateTrajectoryEvaluationData;
pub use trajectory::{
    GateMotionLimits, GateOpeningKnot, GateOpeningTrajectory, GateTrajectoryAccuracy,
    GateTrajectoryAssignment, GateTrajectoryBatch, GateTrajectoryEvaluation, GateTrajectoryEvent,
    GateTrajectoryInterval, GateTrajectoryOutput, GateTrajectorySupport,
};
