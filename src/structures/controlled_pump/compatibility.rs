//! Complete immutable control and response-model identity without state projection.

use crate::cpu::structures::controlled_pump::CpuControlledPumpContext;
use std::sync::Arc;

pub(crate) fn same_context(
    a: &Arc<CpuControlledPumpContext>,
    b: &Arc<CpuControlledPumpContext>,
) -> bool {
    Arc::ptr_eq(a, b)
        || (a.definitions == b.definitions
            && crate::structures::same_control_context(&a.controls, &b.controls))
}
