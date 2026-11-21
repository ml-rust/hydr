//! Checked contributing-support ownership and immutable real-consumer input packets.

mod admission;
mod compile;
mod definition;
mod error;
#[cfg(test)]
mod fixture;
mod interval;
mod partition;
mod rates;
mod selection;
mod subcatchment;
mod surface;

pub use definition::{
    BoundaryBinding, CellRainAssignment, Contribution, ContributionKind, PartitionAssertion,
    PoolRainSupport, SupportAtom, SupportPlan, SupportPlanInput, SurfaceCellSupport,
    WholeReplacement,
};
pub use error::{SupportError, SupportErrorKind};
pub use interval::{
    AdmittedInterval, BoundBoundaryInput, RainSuppressionReceipt, SubcatchmentContributionInput,
    SupportIntervalInput, SurfaceObservationInput,
};
pub use rates::AdmittedRates;
