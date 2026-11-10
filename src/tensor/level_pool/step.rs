//! Complete persistent arithmetic and stage witnesses for one candidate.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct TensorPoolStep<R: Runtime<DType = DType>> {
    pub source: Option<super::TensorPoolSourceStep<R>>,
    pub old_volume_m3: Tensor<R>,
    pub incoming_volume_m3: Tensor<R>,
    pub outgoing_volume_m3: Tensor<R>,
    pub remaining_volume_m3: Tensor<R>,
    pub volume_m3: Tensor<R>,
    pub outgoing_witness_m3: Tensor<R>,
    pub incoming_witness_m3: Tensor<R>,
    pub water_residual_m3: Tensor<R>,
    pub witness_consistency_m3: Tensor<R>,
    pub witness_consistency_bound_m3: Tensor<R>,
    pub physical_scale_m3: Tensor<R>,
    pub stage_m: Tensor<R>,
    pub reconstructed_volume_m3: Tensor<R>,
    pub inverse_residual_m3: Tensor<R>,
    pub cumulative_incoming_before_m3: Tensor<R>,
    pub cumulative_incoming_after_m3: Tensor<R>,
    pub cumulative_incoming_witness_m3: Tensor<R>,
    pub cumulative_outgoing_before_m3: Tensor<R>,
    pub cumulative_outgoing_after_m3: Tensor<R>,
    pub cumulative_outgoing_witness_m3: Tensor<R>,
    pub incoming_bound_before_m3: Tensor<R>,
    pub incoming_bound_after_m3: Tensor<R>,
    pub outgoing_bound_before_m3: Tensor<R>,
    pub outgoing_bound_after_m3: Tensor<R>,
}
