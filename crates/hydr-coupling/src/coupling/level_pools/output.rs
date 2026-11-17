/// One accepted structure transfer and the proposal that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolTransfer {
    /// Exact ordered structure identity.
    pub structure_id: String,
    /// Original signed midpoint Q*dt proposal, in m³.
    pub proposed_volume_m3: f64,
    /// One authoritative signed accepted volume. Positive means A to B.
    pub signed_volume_m3: f64,
    /// Common factor applied to competing withdrawals from its donor.
    pub donor_limit_factor: f64,
    /// Physical midpoint stage, absent for a dry local attachment.
    pub midpoint_stage_a_m: Option<f64>,
    /// Physical midpoint stage, absent for a dry local attachment.
    pub midpoint_stage_b_m: Option<f64>,
}

/// Original candidate-estimate packet retained for complete checkpoint checks.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolIterationData {
    /// Estimated endpoint stages. Dry entries retain the declared interpolation limit only.
    pub end_stages_m: Vec<f64>,
    /// Previous accepted-volume estimates in structure order, in m³.
    pub signed_volumes_m3: Vec<f64>,
    /// Exact pool wet flags from authoritative candidate volumes.
    pub pool_wet: Vec<bool>,
    /// Exact midpoint support flags for ordered A attachments.
    pub attachment_a_wet: Vec<bool>,
    /// Exact midpoint support flags for ordered B attachments.
    pub attachment_b_wet: Vec<bool>,
}

/// Complete last accepted interval, projected only on explicit request.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledLevelPoolStepDiagnostics {
    /// Complete original boundary products and cumulative additions in binding order.
    pub external_inflows: Vec<super::CoupledPoolInflowReceipt>,
    /// Original covered local rates and external inputs. Explicit zero forcing remains present.
    pub forcing_interval: Option<super::CoupledLevelPoolForcingInterval>,
    /// Accepted interval start in seconds.
    pub start_s: f64,
    /// Actual accepted interval end in seconds.
    pub end_s: f64,
    /// Original requested endpoint, before timestep limits and retries.
    pub requested_end_s: f64,
    /// Number of fixed-point updates in the accepted solve.
    pub iterations: usize,
    /// Number of interval halvings before acceptance.
    pub retries: usize,
    /// Largest absolute endpoint-head iteration residual, in metres.
    pub head_residual_m: f64,
    /// Largest absolute signed-volume iteration residual, in m³.
    pub transfer_residual_m3: f64,
    /// Water residual after actual rain, external inflows, bottom loss, and evaporation, in m³.
    pub mass_residual_m3: f64,
    /// Conservative internal/external assembly and incoming-merge rounding bound, in m³.
    pub aggregation_roundoff_bound_m3: f64,
    /// Exact estimate packet used by the final reevaluated hydraulic map.
    pub iteration: CoupledLevelPoolIterationData,
    /// One accepted signed transfer per structure.
    pub transfers: Vec<CoupledLevelPoolTransfer>,
}
