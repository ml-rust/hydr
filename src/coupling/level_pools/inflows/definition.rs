//! Runtime-neutral boundary identities and complete continuation values.

pub use crate::forcing::WaterAmount as CoupledPoolInflowValue;

/// Caller evidence assigning one selected external input to one receiving pool.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoupledPoolInflowBinding {
    /// Unique hydraulic boundary identity.
    pub boundary_id: String,
    /// Exact receiving pool identity.
    pub pool_id: String,
    /// Unique identity of the already-selected supplied input.
    pub input_id: String,
    /// Physical input support identity. This establishes no geometric overlap proof.
    pub physical_support_id: String,
    /// Evidence connecting the external input to this receiving pool.
    pub boundary_assertion: String,
    /// Evidence excluding duplicate input, represented storage, and local rainfall ownership.
    pub ownership_assertion: String,
}

/// One declared external input for a covered rate or exact integrated span.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolInflow {
    /// Exact declared boundary identity.
    pub boundary_id: String,
    /// Exact expected supplied-input identity.
    pub input_id: String,
    /// Original rate or integrated volume. No conversion between variants occurs.
    pub value: CoupledPoolInflowValue,
}

/// Complete cumulative external-volume accounting for one boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolInflowLedger {
    /// Exact declared boundary identity.
    pub boundary_id: String,
    /// Exact expected supplied-input identity.
    pub input_id: String,
    /// Cumulative original represented external volume in m³.
    pub cumulative_volume_m3: f64,
    /// Conservative cumulative addition error bound in m³.
    pub roundoff_bound_m3: f64,
}

/// Original interval volume and cumulative addition witnesses for one boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct CoupledPoolInflowReceipt {
    /// Exact declared boundary identity.
    pub boundary_id: String,
    /// Exact expected supplied-input identity.
    pub input_id: String,
    /// Original supplied variant and operand, retaining its floating-point bits.
    pub input: CoupledPoolInflowValue,
    /// Actual accepted duration in seconds.
    pub dt_s: f64,
    /// Original integrated volume or represented rate-times-duration volume in m³.
    pub volume_m3: f64,
    /// Original accepted cumulative counter in m³.
    pub cumulative_before_m3: f64,
    /// Candidate cumulative counter in m³.
    pub cumulative_after_m3: f64,
    /// Signed TwoSum low component for the cumulative addition in m³.
    pub addition_witness_m3: f64,
    /// Original conservative cumulative addition bound in m³.
    pub bound_before_m3: f64,
    /// Candidate conservative cumulative addition bound in m³.
    pub bound_after_m3: f64,
}

impl CoupledPoolInflowLedger {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.boundary_id == other.boundary_id
            && self.input_id == other.input_id
            && self.cumulative_volume_m3.to_bits() == other.cumulative_volume_m3.to_bits()
            && self.roundoff_bound_m3.to_bits() == other.roundoff_bound_m3.to_bits()
    }
}
impl CoupledPoolInflowReceipt {
    pub(crate) fn same_bits(&self, other: &Self) -> bool {
        self.boundary_id == other.boundary_id
            && self.input_id == other.input_id
            && self.input.same_bits(&other.input)
            && self.dt_s.to_bits() == other.dt_s.to_bits()
            && self.volume_m3.to_bits() == other.volume_m3.to_bits()
            && self.cumulative_before_m3.to_bits() == other.cumulative_before_m3.to_bits()
            && self.cumulative_after_m3.to_bits() == other.cumulative_after_m3.to_bits()
            && self.addition_witness_m3.to_bits() == other.addition_witness_m3.to_bits()
            && self.bound_before_m3.to_bits() == other.bound_before_m3.to_bits()
            && self.bound_after_m3.to_bits() == other.bound_after_m3.to_bits()
    }
}
