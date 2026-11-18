use super::ChannelLateralInput;

/// Explicit host projection of one accepted channel cell.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelCellResult {
    /// Authoritative stored water volume in m³.
    pub volume_m3: f64,
    /// Density-normalized longitudinal momentum in m⁴/s.
    pub longitudinal_momentum_m4_per_s: f64,
    /// Water depth above the local bed in metres, including zero when dry.
    pub depth_m: f64,
    /// Water elevation in the declared datum, in metres; absent when dry.
    pub stage_m: Option<f64>,
    /// Longitudinal depth-averaged velocity in m/s; absent when dry.
    pub velocity_m_per_s: Option<f64>,
    /// Longitudinal discharge in m³/s, positive in the reach's increasing direction.
    pub discharge_m3_per_s: f64,
    /// True exactly when the cell contains no represented water.
    pub dry: bool,
}

/// Cumulative external-water accounting without numerical water adjustments.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelWaterLedger {
    /// Original accepted storage in m³.
    pub initial_storage_m3: f64,
    /// Current accepted storage in m³.
    pub storage_m3: f64,
    /// Cumulative incoming endpoint volume in m³.
    pub boundary_in_m3: f64,
    /// Cumulative outgoing endpoint volume in m³.
    pub boundary_out_m3: f64,
    /// Cumulative incoming lateral volume in m³, derived from boundary counters.
    pub lateral_in_m3: f64,
    /// Cumulative withdrawn lateral volume in m³, derived from boundary counters.
    pub lateral_out_m3: f64,
    /// Conservative cumulative lateral-counter rounding bound in m³.
    pub lateral_counter_abs_bound_m3: f64,
    /// Conservative cumulative lateral assembly bound in m³.
    pub lateral_assembly_abs_bound_m3: f64,
    /// Storage change minus net external inflow in m³.
    pub residual_m3: f64,
    /// Cumulative cellwise charge for detected numerical flux-integral underflow, in m³.
    /// This bound consumes the unchanged water tolerance and excludes total water-solution error.
    pub numerical_transfer_underflow_bound_m3: f64,
    /// Cumulative cellwise radius discarded when surveyed two-part states select their centers, in m³.
    /// Charged apart from the transfer underflow bound. Zero on rectangular and triangular geometry.
    pub storage_selection_abs_bound_m3: f64,
}

/// Explicit host projection of the last accepted interval's bounded receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelStepDiagnostics {
    /// Scalar endpoint traces in left/right order. Wall, Exterior and FluxPort entries remain absent.
    pub boundary_traces: [Option<super::ChannelBoundaryTrace>; 2],
    /// Final Riemann flux brackets, absent for rectangular HLL intervals.
    pub riemann: Option<ChannelRiemannDiagnostics>,
    /// Radius-selection momentum bounds, absent for analytic geometry.
    pub friction_geometry: Option<ChannelFrictionGeometryDiagnostics>,
    /// Accepted interval start in seconds.
    pub start_s: f64,
    /// Accepted interval end in seconds.
    pub end_s: f64,
    /// Number of complete trial recalculations before acceptance.
    pub retries: usize,
    /// Effective minimum section-aware Courant interval bound in seconds, including positive infinity.
    pub cfl_limit_s: f64,
    /// Courant interval bound from the accepted-start state in seconds.
    pub accepted_start_cfl_limit_s: f64,
    /// Prefriction candidate Courant interval bound in seconds; absent without nonzero lateral exchange.
    pub candidate_cfl_limit_s: Option<f64>,
    /// All-outgoing donor interval bound in seconds, including positive infinity.
    pub depletion_limit_s: f64,
    /// Total storage at the accepted start in m³.
    pub storage_before_m3: f64,
    /// Total storage at the accepted end in m³.
    pub storage_after_m3: f64,
    /// Incoming endpoint volume during this interval in m³.
    pub boundary_in_m3: f64,
    /// Outgoing endpoint volume during this interval in m³.
    pub boundary_out_m3: f64,
    /// Incoming lateral volume during this interval in m³.
    pub lateral_in_m3: f64,
    /// Withdrawn lateral volume during this interval in m³.
    pub lateral_out_m3: f64,
    /// Conservative source assembly charge within the existing interval water tolerance, in m³.
    pub lateral_assembly_abs_bound_m3: f64,
    /// Conservative counter-addition charge within the existing lifetime water tolerance, in m³.
    pub lateral_counter_addition_abs_bound_m3: f64,
    /// Storage change minus net endpoint and lateral inflow in m³.
    pub residual_m3: f64,
    /// Largest absolute cell water residual in m³.
    pub max_local_residual_m3: f64,
    /// Largest absolute cell momentum recurrence residual in m⁴/s.
    pub max_local_momentum_residual_m4_per_s: f64,
    /// Largest cell bound on observed erased pressure/momentum products in the final interval, in m⁴/s.
    /// This excludes other rounding and truncation errors.
    pub max_momentum_underflow_bound_m4_per_s: f64,
    /// Shared face water volumes in m³, ordered left to right and positive toward increasing x.
    pub face_volume_m3: Vec<f64>,
    /// Low parts of the shared face water volumes in m³. Zero outside surveyed geometry.
    pub face_volume_low_m3: Vec<f64>,
    /// Radius discarded when each cell's two-part volume selects its center, in m³, one per cell.
    pub storage_selection_bound_m3: Vec<f64>,
    /// Radius discarded when each cell's final two-part area selects its center, in m², one per cell.
    /// A diagnostic, not a conservation charge. Zero outside surveyed geometry.
    pub recovery_selection_bound_m2: Vec<f64>,
    /// Radius discarded when each cell's final two-part discharge selects its center, in m³/s.
    /// A diagnostic, not a conservation charge. Zero outside surveyed geometry.
    pub discharge_recovery_selection_bound_m3_per_s: Vec<f64>,
    /// Detected numerical flux-integral underflow bounds in m³, one per ordered face.
    /// Surveyed bounds also hold the face product radius.
    pub face_transfer_underflow_bound_m3: Vec<f64>,
    /// Interval cellwise underflow charge in m³, counting interior face bounds twice.
    /// This bound consumes the unchanged water tolerance and excludes total water-solution error.
    pub transfer_underflow_bound_m3: f64,
    /// Shared homogeneous momentum transfers in m⁴/s, ordered left to right.
    pub face_momentum_m4_per_s: Vec<f64>,
    /// Low parts of the shared momentum transfers in m⁴/s. Zero outside surveyed geometry.
    pub face_momentum_low_m4_per_s: Vec<f64>,
    /// Left-side hydrostatic pressure corrections times duration in m⁴/s, one per face.
    pub left_pressure_impulse_m4_per_s: Vec<f64>,
    /// Right-side hydrostatic pressure corrections times duration in m⁴/s, one per face.
    pub right_pressure_impulse_m4_per_s: Vec<f64>,
    /// Signed Manning momentum changes in m⁴/s, one per ordered cell.
    pub friction_impulse_m4_per_s: Vec<f64>,
    /// Low parts of the Manning momentum changes in m⁴/s. Zero outside surveyed geometry.
    pub friction_impulse_low_m4_per_s: Vec<f64>,
    /// Cell bounds on observed erased pressure/momentum products in the final interval, in m⁴/s.
    /// Surveyed bounds also hold the face product radii and the momentum selection radii.
    /// These exclude other rounding and truncation errors.
    pub momentum_underflow_bound_m4_per_s: Vec<f64>,
    /// Conservative lateral momentum assembly bounds in m⁴/s, one per ordered cell.
    pub lateral_momentum_abs_bound_m4_per_s: Vec<f64>,
    /// Original accepted inputs and counter witnesses, one per declared lateral binding.
    pub lateral_receipts: Vec<ChannelLateralReceipt>,
}

/// Explicit cumulative accounting for one declared lateral boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelLateralLedger {
    /// Exact declared lateral boundary identity.
    pub boundary_id: String,
    /// Cumulative accepted volume magnitude in m³.
    pub cumulative_volume_m3: f64,
    /// Conservative cumulative addition bound in m³.
    pub roundoff_bound_m3: f64,
}

/// Complete caller-owned lateral continuation values in definition-binding order.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelLateralStateData {
    /// Authoritative cumulative volume magnitudes in m³, one per binding.
    pub cumulative_volume_m3: Vec<f64>,
    /// Conservative cumulative counter-addition bounds in m³, one per binding.
    pub counter_roundoff_bound_m3: Vec<f64>,
    /// Conservative cumulative water assembly charge in m³.
    pub cumulative_assembly_bound_m3: f64,
}

/// Original accepted lateral input, signed source change, and cumulative addition witnesses.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelLateralReceipt {
    /// Original supplied identity, amount variant, operand bits, and inflow velocity.
    pub input: ChannelLateralInput,
    /// Exact receiving or donating cell identity.
    pub cell_id: String,
    /// Applied longitudinal velocity in m/s; absent for zero withdrawals from dry donors.
    pub applied_longitudinal_velocity_m_per_s: Option<f64>,
    /// Accepted original integrated volume or represented rate-times-duration magnitude in m³.
    pub volume_m3: f64,
    /// Signed longitudinal momentum source change in m⁴/s, including the withdrawal's negative sign.
    pub momentum_m4_per_s: f64,
    /// Original cumulative accepted volume magnitude in m³.
    pub cumulative_volume_before_m3: f64,
    /// New cumulative accepted volume magnitude in m³.
    pub cumulative_volume_after_m3: f64,
    /// Signed TwoSum low component for this counter addition in m³.
    pub counter_addition_low_m3: f64,
    /// Original conservative cumulative counter bound in m³.
    pub counter_bound_before_m3: f64,
    /// New conservative cumulative counter bound in m³.
    pub counter_bound_after_m3: f64,
}

/// Explicit flux-bracket diagnostics, excluded from conservation error charges.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelRiemannDiagnostics {
    /// Water-flux bracket widths in m³/s, ordered by face.
    pub water_flux_width_m3_per_s: Vec<f64>,
    /// Momentum-flux bracket widths in m⁴/s², ordered by face.
    pub momentum_flux_width_m4_per_s2: Vec<f64>,
    /// Largest water-flux bracket width in m³/s.
    pub max_water_flux_width_m3_per_s: f64,
    /// Largest momentum-flux bracket width in m⁴/s².
    pub max_momentum_flux_width_m4_per_s2: f64,
    /// Maximum root iterations per solve batch, zero for analytical fluxes.
    pub root_iterations: usize,
    /// Complete surveyed solver work, absent for analytic geometry.
    pub surveyed_work: Option<super::surveyed::PrismaticRiemannWork>,
}

/// Exact-source sensitivity bounds from surveyed hydraulic-radius selection.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelFrictionGeometryDiagnostics {
    /// Momentum bounds in m⁴/s, ordered by cell.
    pub momentum_abs_bound_m4_per_s: Vec<f64>,
    /// Largest cell momentum bound in m⁴/s.
    pub max_momentum_abs_bound_m4_per_s: f64,
}
