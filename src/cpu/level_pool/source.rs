//! Persistent source substeps and complete original rate/counter witnesses.

use crate::cpu::CpuTensor;

#[derive(Clone, Debug)]
pub(crate) struct CpuPoolSourceState {
    pub rain: CpuTensor,
    pub evaporation: CpuTensor,
    pub bottom_loss: CpuTensor,
    pub bottom_loss_demand: CpuTensor,
    pub rain_bound: CpuTensor,
    pub evaporation_bound: CpuTensor,
    pub bottom_loss_bound: CpuTensor,
    pub bottom_loss_demand_bound: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuPoolSourceInput {
    pub rain: CpuTensor,
    pub evaporation: CpuTensor,
    pub bottom_loss: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuPoolCap {
    pub potential_volume_m3: CpuTensor,
    pub actual_volume_m3: CpuTensor,
    pub unmet_volume_m3: CpuTensor,
    pub unmet_witness_m3: CpuTensor,
    pub reconciliation_residual_m3: CpuTensor,
    pub reconciliation_consistency_m3: CpuTensor,
    pub reconciliation_bound_m3: CpuTensor,
    pub availability_limited: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuPoolCounter {
    pub before_m3: CpuTensor,
    pub after_m3: CpuTensor,
    pub witness_m3: CpuTensor,
    pub bound_before_m3: CpuTensor,
    pub bound_after_m3: CpuTensor,
}
#[derive(Clone, Debug)]
pub(crate) struct CpuPoolSourceStep {
    pub frozen_area_m2: CpuTensor,
    pub rain_rate_m_per_s: CpuTensor,
    pub potential_evaporation_rate_m_per_s: CpuTensor,
    pub bottom_loss_rate_m3_per_s: CpuTensor,
    pub rain_rate_volume_m3_per_s: CpuTensor,
    pub rain_volume_m3: CpuTensor,
    pub potential_evaporation_rate_volume_m3_per_s: CpuTensor,
    pub potential_evaporation_volume_m3: CpuTensor,
    pub after_rain_volume_m3: CpuTensor,
    pub after_bottom_loss_volume_m3: CpuTensor,
    pub post_source_volume_m3: CpuTensor,
    pub rain_witness_m3: CpuTensor,
    pub bottom_loss_witness_m3: CpuTensor,
    pub evaporation_witness_m3: CpuTensor,
    pub post_source_reconstructed_volume_m3: CpuTensor,
    pub post_source_inverse_residual_m3: CpuTensor,
    pub segment: CpuTensor,
    pub post_source_stage_m: CpuTensor,
    pub active: CpuTensor,
    pub bottom_loss_cap: CpuPoolCap,
    pub evaporation_cap: CpuPoolCap,
    pub rain_counter: CpuPoolCounter,
    pub evaporation_counter: CpuPoolCounter,
    pub bottom_loss_counter: CpuPoolCounter,
    pub bottom_loss_demand_counter: CpuPoolCounter,
}

impl CpuPoolSourceState {
    pub(crate) fn zero(context: &super::CpuPoolContext) -> Self {
        Self {
            rain: context.zero.clone(),
            evaporation: context.zero.clone(),
            bottom_loss: context.zero.clone(),
            bottom_loss_demand: context.zero.clone(),
            rain_bound: context.zero.clone(),
            evaporation_bound: context.zero.clone(),
            bottom_loss_bound: context.zero.clone(),
            bottom_loss_demand_bound: context.zero.clone(),
        }
    }
}

impl CpuPoolSourceInput {
    pub(crate) fn new(
        context: &super::CpuPoolContext,
        inputs: &[crate::storage::LevelPoolSources],
    ) -> Result<Self, crate::storage::LevelPoolError> {
        if context.source_policies.is_none() {
            return Err(crate::storage::LevelPoolError::input(
                "source-aware input requires immutable enabled source policies",
            ));
        }
        if inputs.len() != context.elements.len() {
            return Err(crate::storage::LevelPoolError::input(
                "supply exactly one source record per ordered element",
            ));
        }
        for (index, input) in inputs.iter().enumerate() {
            if [
                input.rain_rate_m_per_s,
                input.potential_evaporation_rate_m_per_s,
                input.bottom_loss_rate_m3_per_s,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err(crate::storage::LevelPoolError::input(format!(
                    "sources={input:?}: require finite nonnegative original rates"
                ))
                .at(index));
            }
        }
        Ok(Self {
            rain: context.tensor(
                &inputs
                    .iter()
                    .map(|i| i.rain_rate_m_per_s)
                    .collect::<Vec<_>>(),
            )?,
            evaporation: context.tensor(
                &inputs
                    .iter()
                    .map(|i| i.potential_evaporation_rate_m_per_s)
                    .collect::<Vec<_>>(),
            )?,
            bottom_loss: context.tensor(
                &inputs
                    .iter()
                    .map(|i| i.bottom_loss_rate_m3_per_s)
                    .collect::<Vec<_>>(),
            )?,
        })
    }
}
