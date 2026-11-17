//! Checked contributing-support ownership and immutable real-consumer input packets.

mod admission;
mod compile;
mod definition;
mod error;
mod interval;
mod partition;
mod selection;
mod subcatchment;
mod surface;

pub use definition::{
    BoundaryBinding, CellRainAssignment, Contribution, ContributionKind, PartitionAssertion,
    SupportAtom, SupportPlan, SupportPlanInput, SurfaceCellSupport, WholeReplacement,
};
pub use error::{SupportError, SupportErrorKind};
pub use interval::{
    AdmittedInterval, BoundBoundaryInput, RainSuppressionReceipt, SubcatchmentContributionInput,
    SupportIntervalInput, SurfaceObservationInput,
};
