//! Persistent authoritative volume and element-level lifetime counters.

use super::CpuPoolContext;
use crate::cpu::CpuTensor;
use crate::storage::{LevelPoolError, LevelPoolState};

#[derive(Clone, Debug)]
pub(crate) struct CpuPoolState {
    pub sources: Option<super::CpuPoolSourceState>,
    pub volume_m3: CpuTensor,
    pub initial_volume_m3: CpuTensor,
    pub cumulative_incoming_volume_m3: CpuTensor,
    pub cumulative_outgoing_volume_m3: CpuTensor,
    pub incoming_roundoff_bound_m3: CpuTensor,
    pub outgoing_roundoff_bound_m3: CpuTensor,
}
impl CpuPoolState {
    pub(crate) fn new(
        context: &CpuPoolContext,
        states: &[LevelPoolState],
    ) -> Result<Self, LevelPoolError> {
        if states.len() != context.elements.len() {
            return Err(LevelPoolError::input(format!(
                "state count={},element count={}: supply explicit state per pool",
                states.len(),
                context.elements.len()
            )));
        }
        for (index, (state, element)) in states.iter().zip(&context.elements).enumerate() {
            if !state.volume_m3.is_finite()
                || state.volume_m3 < 0.0
                || state.volume_m3 > element.curve.maximum_volume_m3()
            {
                return Err(LevelPoolError::input(format!(
                    "volume={},maximum={}: supply explicit finite closed-range storage",
                    state.volume_m3,
                    element.curve.maximum_volume_m3()
                ))
                .at(index));
            }
        }
        let volume = context.tensor(&states.iter().map(|s| s.volume_m3).collect::<Vec<_>>())?;
        Ok(Self {
            sources: context
                .source_policies
                .as_ref()
                .map(|_| super::CpuPoolSourceState::zero(context)),
            initial_volume_m3: volume.clone(),
            volume_m3: volume,
            cumulative_incoming_volume_m3: context.zero.clone(),
            cumulative_outgoing_volume_m3: context.zero.clone(),
            incoming_roundoff_bound_m3: context.zero.clone(),
            outgoing_roundoff_bound_m3: context.zero.clone(),
        })
    }
}
