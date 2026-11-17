//! Immutable physical bindings and cached native reduction topology.

use super::{ChannelPoolInitialState, ChannelPoolSetup};
use crate::numerics::water_transfers::TransferTree;
use hydr_storage::internal::{LevelPoolCut, LevelPoolCutPlan};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use std::collections::BTreeMap;

pub(super) struct Context<R: Runtime<DType = DType>> {
    pub(super) client: R::Client,
    pub(super) setup: ChannelPoolSetup,
    pub(super) initial: ChannelPoolInitialState,
    pub(super) topology: Topology<R>,
    pub(super) reductions: ReductionPlans<R>,
    pub(super) cut_plan: LevelPoolCutPlan<R>,
    pub(super) initial_storage_m3: Tensor<R>,
    pub(super) initial_storage_bound_m3: Tensor<R>,
}

pub(super) struct Topology<R: Runtime<DType = DType>> {
    pub(super) reach_by_id: BTreeMap<String, usize>,
    pub(super) endpoint_connections: Vec<[Option<usize>; 2]>,
    /// Unconnected `(reach, endpoint)` pairs in reach order, left before right.
    pub(super) external_endpoints: Vec<(usize, usize)>,
    pub(super) connection_endpoint_indices: Tensor<R>,
    pub(super) external_endpoint_indices: Tensor<R>,
    pub(super) connection_pool_indices: Tensor<R>,
    pub(super) connection_cut_indices: Tensor<R>,
    pub(super) entrance_loss_coefficients: Tensor<R>,
    pub(super) pool_origins_m: Tensor<R>,
    pub(super) cuts: Vec<LevelPoolCut>,
    pub(super) cut_aggregation: CutAggregation<R>,
    /// Maps connection-ordered nonnegative magnitudes into pool order.
    pub(super) pool_transfers: TransferTree<R>,
}

pub(super) struct CutAggregation<R: Runtime<DType = DType>> {
    pub(super) blocks: TransferTree<R>,
    pub(super) covers: TransferTree<R>,
}

pub(super) struct ReductionPlans<R: Runtime<DType = DType>> {
    /// One group over channel cells in reach order, then pool volumes.
    pub(super) storage: TransferTree<R>,
    /// Two groups over lateral bindings in reach order: inflow, withdrawal.
    pub(super) lateral: TransferTree<R>,
    /// One group over unconnected endpoint order; reuse for each direction and bound vector.
    pub(super) external_counters: TransferTree<R>,
    /// One group over pool order; reuse for directional counters and assembly bounds.
    pub(super) pool_counters: TransferTree<R>,
    /// Three groups: weighted channel face charges, channel cell assembly, pool state assembly.
    pub(super) physical_charges: TransferTree<R>,
}
