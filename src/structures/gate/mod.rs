//! Checked hydrostatic rectangular apertures and explicit opening support.

mod applicability;
mod operands;
mod parameters;
mod projection;

pub use applicability::GateApplicability;
pub use operands::GateHydraulicOperands;
pub use parameters::GateParameters;
pub(crate) use projection::GateOperandProjection;
