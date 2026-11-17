//! Host projections of accepted common intervals, link counters, and the composite balance.

/// Weir regime of one link at its accepted midpoint heads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceChannelWeirRegime {
    /// No supported side stands above the crest. The link carries no water.
    BelowCrest,
    /// One side stands above the crest and the other side does not.
    Free,
    /// Both sides stand above the crest at different heads. Villemonte reduces the flow.
    Submerged,
    /// Both sides stand above the crest at equal heads. The link carries no water.
    EqualHeads,
}

/// Accepted exchange through one link over one common interval.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelLinkTransfer {
    /// Exact setup link identity.
    pub link_id: String,
    /// Accepted volume, positive surface-to-channel, in cubic metres. Both sides apply it.
    pub signed_volume_m3: f64,
    /// Weir discharge at the midpoint heads times the interval, in cubic metres.
    pub weir_volume_m3: f64,
    /// Levelling factor in `[0, 1]` applied to the weir volume.
    ///
    /// It is exactly `1.0` unless the levelling cap binds. A smaller factor stops the exchange
    /// where the donor and receiver end heads level, or where the receiver reaches the crest.
    pub level_factor: f64,
    /// Donor allocation factor in `[0, 1]` applied to the levelled weir volume.
    pub donor_factor: f64,
    /// Weir regime at the midpoint heads.
    pub regime: SurfaceChannelWeirRegime,
    /// Mean of start and end surface stage at the linked cell, in metres.
    pub midpoint_surface_stage_m: f64,
    /// Mean of start and end channel stage at the linked cell, in metres.
    pub midpoint_channel_stage_m: f64,
}

/// One accepted common interval with its iteration, accounting, and per-link exchange.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelStepDiagnostics {
    /// Common accepted start time, in seconds.
    pub start_s: f64,
    /// Common accepted end time, in seconds.
    pub end_s: f64,
    /// Caller endpoint before duration limits and recalculation, in seconds.
    pub requested_end_s: f64,
    /// Complete interval recalculations before acceptance.
    pub retries: usize,
    /// Fixed-point iterations of the accepted attempt.
    pub iterations: usize,
    /// Largest end-stage change of the last fixed-point update, in metres.
    pub head_residual_m: f64,
    /// Largest signed-volume change of the last fixed-point update, in cubic metres.
    pub transfer_residual_m3: f64,
    /// Surface plus channel storage at the start, in cubic metres.
    pub storage_before_m3: f64,
    /// Surface plus channel storage at the end, in cubic metres.
    pub storage_after_m3: f64,
    /// Exchange water the surface received, in cubic metres.
    pub surface_exchange_in_m3: f64,
    /// Exchange water the surface gave, in cubic metres.
    pub surface_exchange_out_m3: f64,
    /// Exchange water the channels received, in cubic metres.
    pub channel_exchange_in_m3: f64,
    /// Exchange water the channels gave, in cubic metres.
    pub channel_exchange_out_m3: f64,
    /// Bound on the gap between each side's net exchange and the summed link volumes, in m³.
    pub paired_bound_m3: f64,
    /// Storage change minus external water, exchange excluded, in cubic metres.
    pub mass_residual_m3: f64,
    /// Admitted magnitude of `mass_residual_m3`, in cubic metres.
    pub mass_bound_m3: f64,
    /// Per-link exchange in ascending link identity.
    pub links: Vec<SurfaceChannelLinkTransfer>,
}

/// Lifetime directional exchange through one link.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelLinkLedger {
    /// Exact setup link identity.
    pub link_id: String,
    /// Accumulated surface-to-channel volume, in cubic metres.
    pub surface_to_channel_m3: f64,
    /// Accumulated channel-to-surface volume, in cubic metres.
    pub channel_to_surface_m3: f64,
    /// Conservative rounding envelope of both counters, in cubic metres.
    pub roundoff_bound_m3: f64,
}

/// Lifetime water accounting of the surface and every reach as one system.
///
/// Exchange stays inside the system. It enters only the separate exchange totals.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelWaterBalance {
    /// Original surface plus channel storage, in cubic metres.
    pub initial_storage_m3: f64,
    /// Current surface plus channel storage, in cubic metres.
    pub storage_m3: f64,
    /// Rain, inward ports, channel endpoint inflow, and caller lateral inflow, in m³.
    pub external_in_m3: f64,
    /// Land withdrawal, infiltration, evaporation, outward ports, channel endpoint outflow,
    /// and caller lateral withdrawal, in cubic metres.
    pub external_out_m3: f64,
    /// Signed surface numerical water adjustments, in cubic metres.
    pub water_adjustment_m3: f64,
    /// Exchange water the surface received, in cubic metres.
    pub surface_exchange_in_m3: f64,
    /// Exchange water the surface gave, in cubic metres.
    pub surface_exchange_out_m3: f64,
    /// Exchange water the channels received, in cubic metres.
    pub channel_exchange_in_m3: f64,
    /// Exchange water the channels gave, in cubic metres.
    pub channel_exchange_out_m3: f64,
    /// Storage change minus external water and adjustments, in cubic metres.
    pub residual_m3: f64,
    /// Admitted magnitude of `residual_m3`: the configured mass budget, child residuals,
    /// and counter rounding, in cubic metres.
    pub bound_m3: f64,
}
