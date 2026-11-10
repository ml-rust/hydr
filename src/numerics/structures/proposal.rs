//! Shared complete-law point evaluation in passive, pump, then gate order.

use crate::cpu::structures::{CpuStructureContext, CpuStructureInput, CpuStructureResult};
use crate::structures::StructureError;

pub(crate) fn evaluate(
    context: &CpuStructureContext,
    input: &CpuStructureInput,
) -> Result<CpuStructureResult, StructureError> {
    let passive = super::evaluate(context, input)?;
    let result = super::pump::evaluate(context, input, passive)?;
    super::gate::evaluate(context, input, result)
}
