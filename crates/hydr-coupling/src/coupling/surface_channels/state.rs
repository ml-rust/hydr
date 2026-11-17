//! Device-resident link counters and the receipt of the last accepted common interval.

use super::SurfaceChannelForcingInterval;
use crate::channel::ChannelReach;
use crate::solver::Solver;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Lifetime directional link volumes in link identity order, each `[L]`.
#[derive(Clone)]
pub(super) struct LinkCounters<R: Runtime<DType = DType>> {
    pub to_channel_m3: Tensor<R>,
    pub to_surface_m3: Tensor<R>,
    /// Upward-rounded sum of every addition error of both counters.
    pub roundoff_bound_m3: Tensor<R>,
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
}

/// Receipt of the last accepted interval. Link tensors are `[L]` in identity order.
pub(super) struct StoredStep<R: Runtime<DType = DType>> {
    pub previous: PreviousFrame<R>,
    /// Caller channel forcing exactly as the trial received it.
    pub forcing: SurfaceChannelForcingInterval,
    pub start_s: f64,
    pub end_s: f64,
    pub requested_end_s: f64,
    pub retries: usize,
    pub iterations: usize,
    pub head_residual_m: f64,
    pub transfer_residual_m3: f64,
    pub totals: StepTotals,
    pub signed_m3: Tensor<R>,
    pub weir_m3: Tensor<R>,
    pub factor: Tensor<R>,
    pub regime: Tensor<R>,
    pub midpoint_surface_m: Tensor<R>,
    pub midpoint_channel_m: Tensor<R>,
}
