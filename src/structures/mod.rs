//! Typed instantaneous structure laws and held-head proposals without storage or donor acceptance.

mod applicability;
mod batch;
pub(crate) mod definition;
mod error;
mod heads;
mod held;
mod output;
mod parameters;
mod projection;
mod proposal;

pub use applicability::{OrificeApplicability, WeirApplicability};
pub use batch::StructureBatch;
pub use definition::{DirectionPolicy, FlowDirection, StructureElement, StructureLaw};
pub use error::{StructureError, StructureInputErrorKind};
pub use heads::{EndpointHead, HeadConvention, HeadSupport, StructureHeads};
pub use held::HeldHeadProposal;
pub use output::{
    HeldHeadInterval, HeldHeadVolumeOutput, StructureHydraulicOperands, StructureLawKind,
    StructureOutput, StructureRegime, StructureStatus,
};
pub use parameters::{SubmergedOrificeParameters, SuppressedRectangularWeirParameters};
pub use proposal::StructureProposal;
