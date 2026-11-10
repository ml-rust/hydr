//! Persistent batched finite-rate forward tensors.

use super::record::record;
use super::*;
use crate::cpu::CpuTensor;

record!(
    CpuActuatorFma,
    CpuActuatorFmaValues {
        valid: CpuTensor,
        rate: CpuTensor,
        time: CpuTensor,
        addend: CpuTensor,
        result: CpuTensor,
        error: CpuTensor,
        zero_grid: CpuTensor,
        binary_valid: CpuTensor,
        binary_product: CpuTensor,
        binary_high: CpuTensor,
        binary_low: CpuTensor,
    }
);
record!(
    CpuActuatorMotion,
    CpuActuatorMotionValues {
        valid: CpuTensor,
        direction: CpuTensor,
        rate: CpuTensor,
        primary: CpuActuatorFma,
        low: CpuActuatorFma,
        error: CpuTensor,
        raw_lower_valid: CpuTensor,
        raw_lower: CpuTensor,
        raw_upper_valid: CpuTensor,
        raw_upper: CpuTensor,
    }
);
record!(
    CpuActuatorEndpoint,
    CpuActuatorEndpointValues {
        anchor: CpuTensor,
        motion: CpuActuatorMotion,
        lower: CpuTensor,
        upper: CpuTensor,
    }
);
record!(
    CpuActuatorForward,
    CpuActuatorForwardValues {
        valid: CpuTensor,
        anchor: CpuActuatorAnchor,
        query: CpuTensor,
        target: CpuTensor,
        elapsed_high: CpuTensor,
        elapsed_low: CpuTensor,
        lower_endpoint: CpuActuatorEndpoint,
        upper_endpoint: CpuActuatorEndpoint,
        center_motion: CpuActuatorMotion,
        position: CpuActuatorPosition,
    }
);
