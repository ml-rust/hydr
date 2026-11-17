//! Exact complete receipt identity, including signed zero and optional evidence.

use super::super::*;

pub(super) trait ReceiptBits {
    fn same_bits(&self, other: &Self) -> bool;
}
impl ReceiptBits for f64 {
    fn same_bits(&self, other: &Self) -> bool {
        self.is_finite() && other.is_finite() && self.to_bits() == other.to_bits()
    }
}
macro_rules! plain { ($($ty:ty),* $(,)?)=>{$(impl ReceiptBits for $ty{fn same_bits(&self,other:&Self)->bool{self==other}})*}; }
plain!(
    usize,
    bool,
    String,
    GateActuatorPhase,
    GateActuatorDirection,
    GateActuatorSupport
);
impl<T: ReceiptBits> ReceiptBits for Option<T> {
    fn same_bits(&self, other: &Self) -> bool {
        match (self, other) {
            (Some(a), Some(b)) => a.same_bits(b),
            (None, None) => true,
            _ => false,
        }
    }
}
impl<T: ReceiptBits> ReceiptBits for [T] {
    fn same_bits(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().zip(other).all(|(a, b)| a.same_bits(b))
    }
}
impl<T: ReceiptBits> ReceiptBits for Vec<T> {
    fn same_bits(&self, other: &Self) -> bool {
        self.as_slice().same_bits(other.as_slice())
    }
}
macro_rules! receipt {($ty:ty{$($field:ident),* $(,)?})=>{impl ReceiptBits for $ty{fn same_bits(&self,other:&Self)->bool{true $(&& self.$field.same_bits(&other.$field))*}}};}
receipt!(GateActuatorAccuracy {
    maximum_absolute_opening_error_m,
    maximum_relative_opening_error,
    maximum_arrival_window_s
});
receipt!(GateActuatorDefinition {
    actuator_id,
    structure_id,
    opening_rate_m_per_s,
    closing_rate_m_per_s,
    accuracy,
    rate_provenance,
    ideal_response_applicability
});
receipt!(GateActuatorInitialState {
    actuator_id,
    opening_m,
    target_opening_m
});
receipt!(GateActuatorSchedule {
    initial_time_s,
    target_times_s,
    end_s
});
receipt!(GateTargetUpdate {
    actuator_id,
    structure_id,
    target_opening_m,
    issued_at_s,
    available_at_s
});
receipt!(GateActuatorEvent {
    time_s,
    target_index,
    arrival_lower_boundary,
    arrival_upper_boundary,
    terminal
});
receipt!(GateActuatorPosition {
    opening_m,
    lower_opening_m,
    upper_opening_m,
    absolute_error_m
});
receipt!(GateActuatorAnchor { time_s, position });
receipt!(GateActuatorTimeExpansion { high_s, low_s });
receipt!(GateActuatorOutput {
    element,
    actuator_id,
    structure_id,
    time_s,
    target_opening_m,
    position,
    phase,
    support,
    arrival_lower_s,
    arrival_upper_s
});
receipt!(GateActuatorArrivalCertificate {
    anchor,
    target_opening_m,
    lower_s,
    upper_s,
    width_s
});
receipt!(GateActuatorBinaryWitness {
    product_m,
    sum_high_m,
    sum_low_m
});
receipt!(GateActuatorFmaStageReceipt {
    signed_rate_m_per_s,
    time_component_s,
    addend_m,
    result_m,
    rounding_error_m,
    zero_grid_certificate,
    exact_binary_witness
});
receipt!(GateActuatorMotionReceipt {
    direction,
    rate_m_per_s,
    primary,
    elapsed_low,
    raw_error_m,
    raw_lower_m,
    raw_upper_m
});
receipt!(GateActuatorEndpointReceipt {
    anchor_opening_m,
    motion,
    lower_opening_m,
    upper_opening_m
});
receipt!(GateActuatorForwardReceipt {
    anchor,
    query_time_s,
    target_opening_m,
    elapsed_s,
    lower_endpoint,
    upper_endpoint,
    center_motion,
    position
});
receipt!(GateActuatorCompletionReceipt {
    episode_index,
    target_generation,
    confirmed_at_s,
    original_arrival,
    effective_upper_s,
    effective_width_s,
    forward
});
receipt!(GateActuatorTargetChangeReceipt {
    target_index,
    time_s,
    previous_target_opening_m,
    previous_anchor,
    previous_completion,
    update,
    left_position,
    new_anchor,
    target_generation,
    new_episode_index
});
receipt!(GateActuatorState {
    actuator_id,
    anchor,
    target_opening_m,
    position,
    phase,
    arrival,
    target_change_count,
    episode_count,
    arrival_count,
    active_episode_index,
    latest_update,
    last_target_change,
    last_completion
});
receipt!(GateActuatorStepReceipt {
    element,
    actuator_id,
    before,
    left_position,
    left_forward,
    update,
    completed_episode,
    target_changed,
    after
});
receipt!(GateActuatorStepDiagnostics {
    start_s,
    end_s,
    event,
    receipts
});
