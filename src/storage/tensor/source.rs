//! Persistent source substeps and complete original rate/counter witnesses.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct TensorPoolSourceState<R: Runtime<DType = DType>> {
    pub rain: Tensor<R>,
    pub evaporation: Tensor<R>,
    pub bottom_loss: Tensor<R>,
    pub bottom_loss_demand: Tensor<R>,
    pub rain_bound: Tensor<R>,
    pub evaporation_bound: Tensor<R>,
    pub bottom_loss_bound: Tensor<R>,
    pub bottom_loss_demand_bound: Tensor<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorPoolSourceInput<R: Runtime<DType = DType>> {
    pub rain: Tensor<R>,
    pub evaporation: Tensor<R>,
    pub bottom_loss: Tensor<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorPoolCap<R: Runtime<DType = DType>> {
    pub potential_volume_m3: Tensor<R>,
    pub actual_volume_m3: Tensor<R>,
    pub unmet_volume_m3: Tensor<R>,
    pub unmet_witness_m3: Tensor<R>,
    pub reconciliation_residual_m3: Tensor<R>,
    pub reconciliation_consistency_m3: Tensor<R>,
    pub reconciliation_bound_m3: Tensor<R>,
    pub availability_limited: Tensor<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorPoolCounter<R: Runtime<DType = DType>> {
    pub before_m3: Tensor<R>,
    pub after_m3: Tensor<R>,
    pub witness_m3: Tensor<R>,
    pub bound_before_m3: Tensor<R>,
    pub bound_after_m3: Tensor<R>,
}
#[derive(Clone, Debug)]
pub(crate) struct TensorPoolSourceStep<R: Runtime<DType = DType>> {
    pub frozen_area_m2: Tensor<R>,
    pub rain_rate_m_per_s: Tensor<R>,
    pub potential_evaporation_rate_m_per_s: Tensor<R>,
    pub bottom_loss_rate_m3_per_s: Tensor<R>,
    pub rain_rate_volume_m3_per_s: Tensor<R>,
    pub rain_volume_m3: Tensor<R>,
    pub potential_evaporation_rate_volume_m3_per_s: Tensor<R>,
    pub potential_evaporation_volume_m3: Tensor<R>,
    pub after_rain_volume_m3: Tensor<R>,
    pub after_bottom_loss_volume_m3: Tensor<R>,
    pub post_source_volume_m3: Tensor<R>,
    pub rain_witness_m3: Tensor<R>,
    pub bottom_loss_witness_m3: Tensor<R>,
    pub evaporation_witness_m3: Tensor<R>,
    pub post_source_reconstructed_volume_m3: Tensor<R>,
    pub post_source_inverse_residual_m3: Tensor<R>,
    pub segment: Tensor<R>,
    pub post_source_stage_m: Tensor<R>,
    pub active: Tensor<R>,
    pub bottom_loss_cap: TensorPoolCap<R>,
    pub evaporation_cap: TensorPoolCap<R>,
    pub rain_counter: TensorPoolCounter<R>,
    pub evaporation_counter: TensorPoolCounter<R>,
    pub bottom_loss_counter: TensorPoolCounter<R>,
    pub bottom_loss_demand_counter: TensorPoolCounter<R>,
}

impl<R> TensorPoolSourceState<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn zero(context: &super::TensorPoolContext<R>) -> Self {
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

impl<R> TensorPoolSourceInput<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn new(
        context: &super::TensorPoolContext<R>,
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
