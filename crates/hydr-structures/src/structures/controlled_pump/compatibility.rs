//! Complete immutable control and response-model identity without state projection.

use crate::structures::tensor::controlled_pump::TensorControlledPumpContext;
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::sync::Arc;

pub(crate) fn same_context<R: Runtime<DType = DType>>(
    a: &Arc<TensorControlledPumpContext<R>>,
    b: &Arc<TensorControlledPumpContext<R>>,
) -> bool {
    Arc::ptr_eq(a, b)
        || (a.definitions == b.definitions
            && crate::structures::same_control_context(&a.controls, &b.controls))
}
