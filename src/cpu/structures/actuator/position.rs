//! Persistent batched finite-rate position tensors.

use super::record::record;
use crate::cpu::CpuTensor;

record!(
    CpuActuatorPosition,
    CpuActuatorPositionValues {
        center: CpuTensor,
        lower: CpuTensor,
        upper: CpuTensor,
        error: CpuTensor,
    }
);
record!(
    CpuActuatorAnchor,
    CpuActuatorAnchorValues {
        epoch: CpuTensor,
        position: CpuActuatorPosition,
    }
);
record!(
    CpuActuatorArrival,
    CpuActuatorArrivalValues {
        kind: CpuTensor,
        lower: CpuTensor,
        upper: CpuTensor,
        width_high: CpuTensor,
        width_low: CpuTensor,
    }
);
