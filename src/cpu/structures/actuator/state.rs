//! Persistent batched state with heap-owned immutable bounded receipts.

use super::record::record;
use super::*;
use crate::cpu::CpuTensor;
use std::sync::Arc;

record!(
    CpuActuatorCompletion,
    CpuActuatorCompletionValues {
        valid: CpuTensor,
        episode: CpuTensor,
        generation: CpuTensor,
        confirmed: CpuTensor,
        anchor: CpuActuatorAnchor,
        target: CpuTensor,
        arrival: CpuActuatorArrival,
        effective_upper: CpuTensor,
        width_high: CpuTensor,
        width_low: CpuTensor,
        forward: Arc<CpuActuatorForward>,
    }
);
record!(
    CpuActuatorUpdate,
    CpuActuatorUpdateValues {
        valid: CpuTensor,
        target: CpuTensor,
        issued: CpuTensor,
        available: CpuTensor,
    }
);
record!(
    CpuActuatorChange,
    CpuActuatorChangeValues {
        valid: CpuTensor,
        index: CpuTensor,
        time: CpuTensor,
        previous_target: CpuTensor,
        previous_anchor: CpuActuatorAnchor,
        previous_completion: CpuActuatorCompletion,
        update: CpuActuatorUpdate,
        left: CpuActuatorPosition,
        new_anchor: CpuActuatorAnchor,
        generation: CpuTensor,
        new_episode_valid: CpuTensor,
        new_episode: CpuTensor,
    }
);
record!(
    CpuActuatorState,
    CpuActuatorStateValues {
        anchor: CpuActuatorAnchor,
        target: CpuTensor,
        position: CpuActuatorPosition,
        phase: CpuTensor,
        arrival: CpuActuatorArrival,
        generation: CpuTensor,
        episodes: CpuTensor,
        arrivals: CpuTensor,
        active_valid: CpuTensor,
        active_index: CpuTensor,
        latest_update: CpuActuatorUpdate,
        last_change: Arc<CpuActuatorChange>,
        last_completion: Arc<CpuActuatorCompletion>,
    }
);
