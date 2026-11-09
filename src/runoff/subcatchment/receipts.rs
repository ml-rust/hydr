//! Complete source, interface, and represented drainage witnesses.

use super::{Subarea, SubcatchmentInput};
use crate::soil::SoilSourceReceipt;

/// One checked receiver rate bridge. Original interface volumes remain authoritative.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentBridgeReceipt {
    /// Stable merged incoming volume in m³.
    pub merged_volume_m3: f64,
    /// Outward volume-merge error envelope in m³.
    pub merge_bound_m3: f64,
    /// Received volume divided by actual area in metres.
    pub received_depth_m: f64,
    /// Received depth divided by the original interval in m/s.
    pub received_rate_m_per_s: f64,
    /// Constructed rain plus received rate in m/s.
    pub total_rate_m_per_s: f64,
    /// Constructed rate times the original interval in metres.
    pub input_depth_m: f64,
    /// Single pre-soil available depth in metres.
    pub available_depth_m: f64,
    /// Shared rain rate times the interval in metres.
    pub rain_depth_m: f64,
    /// Original depth plus rain alone in metres.
    pub rain_only_available_m: f64,
    /// Represented receiver increment over rain-only storage, in metres.
    pub depth_increment_m: f64,
    /// Actual area times represented receiver increment, in m³.
    pub represented_received_volume_m3: f64,
    /// Reported receiver-minus-interface discrepancy in m³, without a ledger plug.
    pub residual_m3: f64,
    /// Outward bound including reported subtraction rounding, in m³.
    pub residual_bound_m3: f64,
    /// Conservative receiver/merge uncertainty envelope in m³.
    pub uncertainty_m3: f64,
}

/// Shared conservative drainage result with no fabricated positive-source receipt.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentDrainageReceipt {
    /// Retained lower excess-root bracket in metres.
    pub root_lower_m: f64,
    /// Retained upper excess-root bracket in metres.
    pub root_upper_m: f64,
    /// Accepted represented depth in metres.
    pub depth_m: f64,
    /// Authoritative post-source budget minus accepted depth in metres.
    pub export_depth_m: f64,
    /// Accepted storage in m³.
    pub storage_after_m3: f64,
    /// Full conservative export in m³.
    pub export_volume_m3: f64,
    /// Endpoint depth-law rate in m/s. None denotes instantaneous overflow.
    pub endpoint_rate_m_per_s: Option<f64>,
    /// Endpoint discharge in m³/s. None denotes instantaneous overflow.
    pub endpoint_discharge_m3_per_s: Option<f64>,
    /// Active endpoint-law depth in metres, zero for an inactive law gate.
    pub law_depth_m: f64,
    /// Active endpoint-law volume in m³.
    pub law_volume_m3: f64,
    /// Signed independent constitutive residual in m³.
    pub constitutive_residual_m3: f64,
    /// Constitutive tolerance scale in m³, without a unit floor.
    pub constitutive_scale_m3: f64,
}

/// One checked donor partition. The complementary outlet is calculated by subtraction.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentPartitionReceipt {
    /// Declared routed fraction.
    pub fraction: f64,
    /// Single authoritative routed volume in m³.
    pub routed_volume_m3: f64,
    /// Complementary direct outlet volume in m³.
    pub outlet_volume_m3: f64,
    /// Independent mathematical partition residual in m³.
    pub residual_m3: f64,
    /// Outward partition envelope in m³.
    pub bound_m3: f64,
}

/// Complete active-subarea source and drainage witness.
#[derive(Clone, Debug, PartialEq)]
pub struct SubareaReceipt {
    /// Ordered catchment identity.
    pub catchment: usize,
    /// Fixed physical subarea identity.
    pub subarea: Subarea,
    /// Interval-start depth in metres.
    pub old_depth_m: f64,
    /// Shared local rain rate in m/s.
    pub rain_rate_m_per_s: f64,
    /// Potential evaporation rate in m/s.
    pub potential_evaporation_m_per_s: f64,
    /// Original external destination volume in m³.
    pub external_volume_m3: f64,
    /// Original received interface volumes by donor subarea, in m³.
    pub received_by_donor_m3: [f64; 3],
    /// Complete stable merge and rate-bridge evidence.
    pub bridge: SubcatchmentBridgeReceipt,
    /// Original native/SI soil evidence for positive pervious support only.
    pub soil: Option<SoilSourceReceipt>,
    /// Original returned infiltration in metres, without post-hoc availability capping.
    pub infiltration_depth_m: f64,
    /// Explicit admitted soil water correction in metres.
    pub adjustment_depth_m: f64,
    /// Availability-limited evaporation in metres, after infiltration.
    pub evaporation_depth_m: f64,
    /// Single authoritative post-source drainage budget in metres.
    pub available_depth_m: f64,
    /// Rate used only for filling-event timing, in m/s. It can be signed.
    pub timing_rate_m_per_s: f64,
    /// Whether timing retains the exact primitive no-loss/no-transfer rate path.
    pub direct_timing: bool,
    /// Checked active drainage duration in seconds.
    pub active_duration_s: f64,
    /// Shared represented hydraulic drainage result.
    pub drainage: SubcatchmentDrainageReceipt,
    /// Exact donor debit and complementary outlet evidence.
    pub partition: SubcatchmentPartitionReceipt,
    /// Original storage in m³.
    pub storage_before_m3: f64,
    /// Actual applied rain volume in m³.
    pub rain_volume_m3: f64,
    /// Authoritative received internal volume in m³.
    pub internal_received_volume_m3: f64,
    /// Actual infiltration volume in m³.
    pub infiltration_volume_m3: f64,
    /// Actual capped evaporation volume in m³.
    pub evaporation_volume_m3: f64,
    /// Explicit soil water adjustment volume in m³.
    pub adjustment_volume_m3: f64,
    /// Independent signed subarea water residual in m³.
    pub water_residual_m3: f64,
    /// Independent subarea physical tolerance scale in m³.
    pub water_scale_m3: f64,
    /// Outward rain-area derivation envelope in m³.
    pub rain_area_bound_m3: f64,
}

/// Complete per-catchment support with explicit absent-class markers.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentReceipt {
    /// Ordered catchment identity.
    pub catchment: usize,
    /// Original explicit caller input over this interval.
    pub input: SubcatchmentInput,
    /// Fixed ordered active classes. None marks absent support.
    pub subareas: [Option<SubareaReceipt>; 3],
    /// Independent whole-catchment water residual in m³.
    pub water_residual_m3: f64,
    /// Whole-catchment physical tolerance scale in m³.
    pub water_scale_m3: f64,
    /// Applied minus total-area rain volume in m³, reported without a ledger adjustment.
    pub rain_area_difference_m3: f64,
    /// Outward rain-area comparison envelope in m³.
    pub rain_area_bound_m3: f64,
}

/// One declared same-interval internal interface, including zero-flow observations.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentTransfer {
    /// Ordered catchment identity.
    pub catchment: usize,
    /// Physical donor.
    pub donor: Subarea,
    /// Physical receiver.
    pub receiver: Subarea,
    /// Identical debit/credit interface volume in m³.
    pub volume_m3: f64,
    /// Declared routed fraction.
    pub fraction: f64,
    /// Full donor export in m³.
    pub donor_volume_m3: f64,
    /// Complementary donor outlet volume in m³.
    pub outlet_volume_m3: f64,
    /// Independent partition residual in m³.
    pub partition_residual_m3: f64,
    /// Outward partition envelope in m³.
    pub partition_bound_m3: f64,
}
