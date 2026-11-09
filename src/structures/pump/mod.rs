//! Checked reference pump curves, explicit operating packets, and evaluated operands.

mod applicability;
mod curve;
mod operands;
mod operating;
mod parameters;

pub use applicability::{PumpApplicability, PumpSpeedPolicy};
pub use curve::{PumpCurve, PumpCurveKnot};
pub use operands::PumpHydraulicOperands;
pub use operating::{StructureOperatingPoint, StructureOperation};
pub use parameters::PumpParameters;

mod projection;
pub(crate) use projection::PumpOperandProjection;
