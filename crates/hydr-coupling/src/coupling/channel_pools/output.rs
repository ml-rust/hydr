use super::ChannelPoolForcingInterval;

/// Actual channel endpoint transfer paired once with the opposite pool exchange.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolTransfer {
    /// Exact setup connection identity.
    pub connection_id: String,
    /// Actual water volume, positive channel-to-pool, in cubic metres.
    pub signed_volume_m3: f64,
    /// Native midpoint pool head used by the retained channel trial, in metres in the common datum.
    pub midpoint_pool_elevation_m: f64,
    /// Connection contribution to channel longitudinal momentum, in metres to the fourth power per second.
    pub longitudinal_momentum_impulse_m4_per_s: f64,
}

/// Lifetime directional water counters for one connection, without a pool momentum counter.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolConnectionLedger {
    /// Exact setup connection identity.
    pub connection_id: String,
    /// Accumulated nonnegative channel-to-pool volume, in cubic metres.
    pub channel_to_pool_volume_m3: f64,
    /// Accumulated nonnegative pool-to-channel volume, in cubic metres.
    pub pool_to_channel_volume_m3: f64,
    /// Conservative arithmetic envelope for accumulated counters, in cubic metres.
    pub counter_roundoff_bound_m3: f64,
}

/// Original-water eligibility check for exports through attachments at or above one elevation.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolDonorCut {
    /// Exact setup pool identity.
    pub pool_id: String,
    /// Attachment bed cut, in metres in the common datum.
    pub bed_elevation_m: f64,
    /// Downward-bounded original water above this cut, excluding incoming water, in cubic metres.
    pub available_lower_m3: f64,
    /// Upward-bounded export demand through attachments at or above this cut, in cubic metres.
    pub outgoing_upper_m3: f64,
}

/// Retained numerical iterates and physical support flags in immutable setup order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolIterationData {
    /// Numerical end-stage iterates in pool order, in metres. Inspect `pool_wet` before physical interpretation.
    pub end_stages_m: Vec<f64>,
    /// Signed channel-to-pool volumes in connection order, in cubic metres.
    pub signed_volumes_m3: Vec<f64>,
    /// End-state pool wetness in pool order. Child outputs retain their physical dry semantics.
    pub pool_wet: Vec<bool>,
    /// Attachment support flags in connection order for the retained evaluation.
    pub attachment_wet: Vec<bool>,
}

/// Lifetime physical water accounting across channels and pools, excluding internal connection traffic.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolWaterBalance {
    /// Complete original represented storage, in cubic metres.
    pub initial_storage_m3: f64,
    /// Complete current represented storage, in cubic metres.
    pub storage_m3: f64,
    /// Accumulated actual inflow through unconnected channel endpoints, in cubic metres.
    pub external_boundary_in_m3: f64,
    /// Accumulated actual outflow through unconnected channel endpoints, in cubic metres.
    pub external_boundary_out_m3: f64,
    /// Accumulated actual lateral inflow, in cubic metres.
    pub lateral_in_m3: f64,
    /// Accumulated actual lateral withdrawal, in cubic metres.
    pub lateral_out_m3: f64,
    /// Current minus initial storage, minus external inflows, plus external outflows, in cubic metres.
    pub residual_m3: f64,
    /// Conservative arithmetic envelope for physical accounting, in cubic metres.
    pub roundoff_bound_m3: f64,
}

/// One accepted common interval with retained forcing, numerical iteration, and paired physical transfers.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolStepDiagnostics {
    /// Common accepted start time, in seconds.
    pub start_s: f64,
    /// Actual common accepted end time, in seconds.
    pub end_s: f64,
    /// Original requested endpoint before duration limits and recalculation, in seconds.
    pub requested_end_s: f64,
    /// Fixed-point iterations completed for the accepted attempt.
    pub iterations: usize,
    /// Complete interval recalculations before acceptance.
    pub retries: usize,
    /// Final absolute head iteration residual, in metres.
    pub head_residual_m: f64,
    /// Final absolute signed-transfer iteration residual, in cubic metres.
    pub transfer_residual_m3: f64,
    /// Complete represented storage at the common start, in cubic metres.
    pub storage_before_m3: f64,
    /// Complete represented storage at the common end, in cubic metres.
    pub storage_after_m3: f64,
    /// Actual unconnected endpoint inflow over this interval, in cubic metres.
    pub external_boundary_in_m3: f64,
    /// Actual unconnected endpoint outflow over this interval, in cubic metres.
    pub external_boundary_out_m3: f64,
    /// Actual lateral inflow over this interval, in cubic metres.
    pub lateral_in_m3: f64,
    /// Actual lateral withdrawal over this interval, in cubic metres.
    pub lateral_out_m3: f64,
    /// Storage change minus external inflows plus external outflows, in cubic metres.
    pub mass_residual_m3: f64,
    /// Conservative child and aggregation arithmetic envelope, in cubic metres.
    pub aggregation_roundoff_bound_m3: f64,
    /// Complete original external policies and lateral operands for this accepted calculation.
    pub forcing_interval: ChannelPoolForcingInterval,
    /// Retained numerical iterates and support flags.
    pub iteration: ChannelPoolIterationData,
    /// Actual paired channel endpoint transfers in setup connection order.
    pub transfers: Vec<ChannelPoolTransfer>,
    /// Original-water cut checks, without incoming donor credit or added conservation allowances.
    pub donor_cuts: Vec<ChannelPoolDonorCut>,
}
