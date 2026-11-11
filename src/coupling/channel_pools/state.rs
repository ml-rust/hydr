//! Private common state and bounded native continuation receipts.

use crate::channel::{ChannelCouplingForcing, ChannelReach, ChannelSnapshot};
use crate::storage::{LevelPoolBatch, LevelPoolSnapshot};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct Accepted<R: Runtime<DType = DType>> {
    pub(super) reaches: Vec<ChannelReach<R>>,
    pub(super) pools: LevelPoolBatch<R>,
    pub(super) counters: ParentCounters<R>,
    pub(super) last: Option<Arc<StoredStep<R>>>,
}

#[derive(Clone)]
pub(super) struct ParentCounters<R: Runtime<DType = DType>> {
    pub(super) connection_in_m3: Tensor<R>,
    pub(super) connection_out_m3: Tensor<R>,
    pub(super) connection_in_bound_m3: Tensor<R>,
    pub(super) connection_out_bound_m3: Tensor<R>,
    pub(super) external_in_m3: Tensor<R>,
    pub(super) external_out_m3: Tensor<R>,
    pub(super) external_in_bound_m3: Tensor<R>,
    pub(super) external_out_bound_m3: Tensor<R>,
    /// Per-pool cumulative directional exchange assembly bounds, shape `[P]`.
    pub(super) pool_incoming_assembly_bound_m3: Tensor<R>,
    pub(super) pool_outgoing_assembly_bound_m3: Tensor<R>,
    /// Channel transfer, complete channel assembly, and pool state assembly, shape `[3]`.
    pub(super) historical_physical_charge_m3: Tensor<R>,
}

#[derive(Clone)]
pub(super) struct PreviousFrame<R: Runtime<DType = DType>> {
    pub(super) reaches: Vec<ChannelSnapshot<R>>,
    pub(super) pools: LevelPoolSnapshot<R>,
    pub(super) counters: ParentCounters<R>,
}

pub(super) struct BoundForcing<R: Runtime<DType = DType>> {
    pub(super) start_s: f64,
    pub(super) end_s: f64,
    pub(super) reaches: Vec<ChannelCouplingForcing<R>>,
    pub(super) requires_exact_span: bool,
}

pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub(super) previous: PreviousFrame<R>,
    pub(super) forcing: Arc<BoundForcing<R>>,
    pub(super) requested_end_s: f64,
    pub(super) iterations: usize,
    pub(super) retries: usize,
    pub(super) interval: NativeInterval<R>,
    pub(super) accounting: NativeIntervalAccounting<R>,
}

pub(super) struct NativeEvaluation<R: Runtime<DType = DType>> {
    pub(super) reaches: Vec<ChannelReach<R>>,
    pub(super) pools: LevelPoolBatch<R>,
    pub(super) interval: NativeInterval<R>,
}

#[derive(Clone)]
pub(super) struct NativeInterval<R: Runtime<DType = DType>> {
    pub(super) input_packet: Tensor<R>,
    pub(super) output_packet: Tensor<R>,
    pub(super) midpoint_pool_stages_m: Tensor<R>,
    pub(super) connection_volume_m3: Tensor<R>,
    /// Actual contribution to channel momentum, including endpoint orientation and pressure.
    pub(super) connection_momentum_m4_per_s: Tensor<R>,
    /// Outward channel volumes at unconnected endpoints, shape `[X]`.
    pub(super) external_volume_m3: Tensor<R>,
    pub(super) cut_available_lower_m3: Tensor<R>,
    pub(super) cut_outgoing_upper_m3: Tensor<R>,
    /// Current directional exchange aggregation bounds, shape `[P]`.
    pub(super) pool_incoming_bound_m3: Tensor<R>,
    pub(super) pool_outgoing_bound_m3: Tensor<R>,
    pub(super) head_residual_m: Tensor<R>,
    pub(super) transfer_residual_m3: Tensor<R>,
}

#[derive(Clone)]
pub(super) struct NativeIntervalAccounting<R: Runtime<DType = DType>> {
    pub(super) storage_before_m3: Tensor<R>,
    pub(super) storage_after_m3: Tensor<R>,
    pub(super) external_in_m3: Tensor<R>,
    pub(super) external_out_m3: Tensor<R>,
    pub(super) lateral_in_m3: Tensor<R>,
    pub(super) lateral_out_m3: Tensor<R>,
    pub(super) residual_m3: Tensor<R>,
    pub(super) roundoff_bound_m3: Tensor<R>,
    /// Current physical charges in the same three-lane order as parent history.
    pub(super) physical_charge_m3: Tensor<R>,
}

#[derive(Clone)]
pub(super) struct NativeWaterBalance<R: Runtime<DType = DType>> {
    pub(super) initial_storage_m3: Tensor<R>,
    pub(super) storage_m3: Tensor<R>,
    pub(super) external_in_m3: Tensor<R>,
    pub(super) external_out_m3: Tensor<R>,
    pub(super) lateral_in_m3: Tensor<R>,
    pub(super) lateral_out_m3: Tensor<R>,
    pub(super) residual_m3: Tensor<R>,
    pub(super) roundoff_bound_m3: Tensor<R>,
}
