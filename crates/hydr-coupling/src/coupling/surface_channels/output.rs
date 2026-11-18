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

/// Flow regime of one reach-end port over one accepted interval.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceChannelPortRegime {
    /// Both waves straddle the port. The HLL flux mixes both states.
    Subcritical,
    /// Every wave runs into the surface. The flux is the reach state's own flux.
    SupercriticalToSurface,
    /// Every wave runs into the reach. The flux is the surface state's own flux.
    SupercriticalToReach,
    /// A rarefaction fan holds its sonic point at the port.
    Transcritical,
    /// One side holds no water above the higher bed.
    WetDryFront,
    /// Neither side holds water above the higher bed. The port moves nothing.
    Dry,
}

/// Accepted exchange through one reach-end port over one common interval.
///
/// The flux comes from the accepted-start states and stays fixed for the attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPortTransfer {
    /// Exact setup port identity.
    pub port_id: String,
    /// Reach volume, positive surface-to-channel, in cubic metres: minus the reach's
    /// outward rate times the interval.
    pub signed_volume_m3: f64,
    /// Normal momentum leaving the reach along its outward normal, pressure included, in m⁴/s.
    pub reach_momentum_impulse_m4_per_s: f64,
    /// Momentum entering the surface cells in grid x and y, in m⁴/s.
    pub surface_momentum_impulse_m4_per_s: [f64; 2],
    /// Momentum leaving the reach minus momentum entering the surface, grid x and y, in m⁴/s.
    ///
    /// It holds the bed-step reaction and any tangential momentum that leaves the surface.
    pub junction_reaction_m4_per_s: [f64; 2],
    /// Signal-speed bound the reach received, in m/s.
    pub signal_speed_m_per_s: f64,
    /// Port regime from the regimes of its faces.
    pub regime: SurfaceChannelPortRegime,
    /// The signed volume changes sign against the previous interval.
    pub reversed: bool,
    /// Surface port volume plus reach outward volume, in cubic metres.
    pub paired_gap_m3: f64,
    /// Admitted magnitude of `paired_gap_m3`, in cubic metres.
    pub paired_bound_m3: f64,
}

/// Lifetime directional exchange through one reach-end port.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelPortLedger {
    /// Exact setup port identity.
    pub port_id: String,
    /// Accumulated surface-to-channel reach volume, in cubic metres.
    pub surface_to_channel_m3: f64,
    /// Accumulated channel-to-surface reach volume, in cubic metres.
    pub channel_to_surface_m3: f64,
    /// Conservative rounding envelope of both counters, in cubic metres.
    pub roundoff_bound_m3: f64,
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
    /// Fixed-point contraction per update over the last two updates, `sqrt(r_k / r_{k-2})`.
    ///
    /// `r` is the largest change over its allowance. Zero below two updates or after a zero
    /// change.
    pub contraction: f64,
    /// Largest end-stage change of the last fixed-point update, in metres.
    pub head_residual_m: f64,
    /// Largest signed-volume change of the last fixed-point update, in cubic metres.
    pub transfer_residual_m3: f64,
    /// Surface plus channel storage at the start, in cubic metres.
    pub storage_before_m3: f64,
    /// Surface plus channel storage at the end, in cubic metres.
    pub storage_after_m3: f64,
    /// Link water the surface received, in cubic metres. Port water stays out.
    pub surface_exchange_in_m3: f64,
    /// Link water the surface gave, in cubic metres. Port water stays out.
    pub surface_exchange_out_m3: f64,
    /// Link water the channels received, in cubic metres.
    pub channel_exchange_in_m3: f64,
    /// Link water the channels gave, in cubic metres.
    pub channel_exchange_out_m3: f64,
    /// Bound on the gap between each side's net exchange and the summed link volumes, in m³.
    pub paired_bound_m3: f64,
    /// Storage change minus external water, exchange and port water excluded, in cubic metres.
    pub mass_residual_m3: f64,
    /// Admitted magnitude of `mass_residual_m3`, in cubic metres.
    pub mass_bound_m3: f64,
    /// Per-link exchange in ascending link identity.
    pub links: Vec<SurfaceChannelLinkTransfer>,
    /// Per-port exchange in ascending port identity. Empty without ports.
    pub ports: Vec<SurfaceChannelPortTransfer>,
    /// Sum of every port's `paired_bound_m3`, in cubic metres. It joins `mass_bound_m3`.
    pub port_paired_bound_m3: f64,
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
