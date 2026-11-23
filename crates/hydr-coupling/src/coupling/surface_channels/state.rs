//! Device-resident link counters and the receipt of the last accepted common interval.

use super::SurfaceChannelForcingInterval;
use super::operating::{OperatingChildren, StepOperation};
use super::pool_ends::{PoolEndCounters, PoolEndInterval};
use super::pools::PoolChild;
use super::ports::{PortCounters, PortInterval};
use super::runoff::{RunoffCounters, RunoffStep};
use crate::channel::ChannelReach;
use crate::runoff::SubcatchmentBatch;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use std::sync::Arc;

/// Lifetime directional link volumes in link identity order, each `[L]`.
#[derive(Clone)]
pub(super) struct LinkCounters<R: Runtime<DType = DType>> {
    pub to_channel_m3: Tensor<R>,
    pub to_surface_m3: Tensor<R>,
    /// Upward-rounded sum of every addition error of both counters.
    pub roundoff_bound_m3: Tensor<R>,
    /// Lifetime port volumes in port identity order. Empty without ports.
    pub ports: PortCounters,
    /// Lifetime pool-end volumes in pool-end identity order. Empty without pool ends.
    pub pool_ends: PoolEndCounters,
    /// Lifetime runoff delivery counters, or `None` without runoff.
    pub runoff: Option<RunoffCounters<R>>,
}

/// Host scalars of one accepted interval's accounting.
#[derive(Clone, Copy, Debug)]
pub(super) struct StepTotals {
    pub storage_before_m3: f64,
    pub storage_after_m3: f64,
    pub surface_in_m3: f64,
    pub surface_out_m3: f64,
    pub channel_in_m3: f64,
    pub channel_out_m3: f64,
    pub paired_bound_m3: f64,
    pub mass_residual_m3: f64,
    pub mass_bound_m3: f64,
}

/// Accepted children and counters at the start of the last interval.
///
/// The children carry no coordinator receipt, so the frame never nests.
pub(super) struct PreviousFrame<R: Runtime<DType = DType>> {
    pub surface: Solver<R>,
    /// Reaches in setup order.
    pub reaches: Vec<ChannelReach<R>>,
    pub counters: LinkCounters<R>,
    /// Accepted pools, or `None` without pools.
    pub pools: Option<PoolChild<R>>,
    /// Accepted operating children, or `None` without them.
    pub operating: Option<OperatingChildren<R>>,
    /// Accepted runoff child, or `None` without runoff.
    pub runoff: Option<SubcatchmentBatch<R>>,
}

/// Receipt of the last accepted interval. Link tensors are `[L]` in identity order.
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub previous: PreviousFrame<R>,
    /// Caller channel and pool forcing exactly as the trial received it.
    pub forcing: SurfaceChannelForcingInterval,
    pub start_s: f64,
    pub end_s: f64,
    pub requested_end_s: f64,
    pub retries: usize,
    pub iterations: usize,
    /// Contraction per update of the accepted attempt. Zero below two updates.
    pub contraction: f64,
    pub head_residual_m: f64,
    pub transfer_residual_m3: f64,
    pub totals: StepTotals,
    pub signed_m3: Tensor<R>,
    pub weir_m3: Tensor<R>,
    pub level: Tensor<R>,
    pub factor: Tensor<R>,
    pub regime: Tensor<R>,
    pub midpoint_surface_m: Tensor<R>,
    pub midpoint_channel_m: Tensor<R>,
    /// Port receipts and water totals. Empty without ports.
    pub ports: PortInterval,
    /// Pool-end volumes and reservoir heads. Empty without pool ends.
    pub pool_ends: PoolEndInterval,
    /// Interval operation of the operating children, or `None` without them.
    pub operation: Option<Arc<StepOperation<R>>>,
    /// Runoff delivery of the interval, or `None` without runoff.
    pub runoff: Option<RunoffStep>,
    /// Structure geometry receipts applied on this interval. Empty without schedules.
    pub geometry: Vec<super::SurfaceChannelGeometryReceipt>,
}
