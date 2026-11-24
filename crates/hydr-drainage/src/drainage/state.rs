//! Canonical SI state, continuity ledger, and snapshot types for SWMM drainage.

use super::definition::SwmmSurchargeMethod;
use super::error::DrainageError;

/// Stable snapshot schema identifier for `SwmmDrainage`.
pub const SWMM_DRAINAGE_SNAPSHOT_SCHEMA: &str = "hydr-swmm-drainage-checkpoint-1";

/// Stable contract version for the pinned EPA SWMM v5.2.4 drainage wrapper.
pub const SWMM_DRAINAGE_CONTRACT_VERSION: &str = "hydr-swmm-drainage-v5.2.4-1";

/// Exact SWMM internal length conversion factor (`MperFT` in `consts.h`).
pub const FT_TO_M: f64 = 0.3048;

/// Exact SWMM internal volume and flow conversion factor (`FT_TO_M^3`).
pub const FT3_TO_M3: f64 = FT_TO_M * FT_TO_M * FT_TO_M;

/// Input unit system declared by the SWMM `.inp` file.
///
/// SWMM stores all internal routing variables in US feet and cubic feet regardless of the
/// `.inp` unit system; this enum records the input file's declared unit system for auditing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwmmInpUnitSystem {
    /// US customary input units (`CFS`, `GPM`, `MGD`).
    Us,
    /// Metric input units (`CMS`, `LPS`, `MLD`).
    Si,
}

impl SwmmInpUnitSystem {
    pub(crate) fn from_worker_code(code: i32) -> Result<Self, DrainageError> {
        match code {
            0 => Ok(Self::Us),
            1 => Ok(Self::Si),
            other => Err(DrainageError::process(format!(
                "unknown SWMM UnitSystem code `{other}` from worker OPEN"
            ))),
        }
    }
}

/// Network object counts and options reported when the SWMM model opens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwmmNetworkSummary {
    /// Total node count (junctions, outfalls, dividers, storages).
    pub node_count: usize,
    /// Total link count (conduits, pumps, orifices, weirs, outlets).
    pub link_count: usize,
    /// Conduit count.
    pub conduit_count: usize,
    /// Storage node count.
    pub storage_count: usize,
    /// Outfall node count.
    pub outfall_count: usize,
    /// Orifice link count.
    pub orifice_count: usize,
    /// Weir link count.
    pub weir_count: usize,
    /// Native SWMM control-rule count.
    pub rule_count: usize,
    /// Active surcharge method reported by SWMM.
    pub surcharge_method: SwmmSurchargeMethod,
    /// Input unit system reported by SWMM.
    pub inp_unit_system: SwmmInpUnitSystem,
}

/// Canonical SI hydraulic state of one SWMM drainage node.
#[derive(Clone, Debug, PartialEq)]
pub struct SwmmNodeState {
    /// Zero-based SWMM node index.
    pub index: usize,
    /// Node identifier from the `.inp` file.
    pub id: String,
    /// Node invert elevation in meters above the declared vertical datum.
    pub invert_m: f64,
    /// Maximum unpressurized depth in meters (distance from invert to rim).
    pub full_depth_m: f64,
    /// Current water depth in meters (can exceed `full_depth_m` under surcharge).
    pub depth_m: f64,
    /// Current hydraulic head in meters (`invert_m + depth_m`).
    pub head_m: f64,
    /// Current nodal ponded or storage volume in cubic meters.
    pub volume_m3: f64,
    /// Total nodal inflow rate in cubic meters per second.
    pub inflow_m3_per_s: f64,
    /// Total nodal outflow rate in cubic meters per second.
    pub outflow_m3_per_s: f64,
    /// Surface overflow / flooding rate at the node in cubic meters per second.
    pub overflow_m3_per_s: f64,
}

/// Canonical SI hydraulic state of one SWMM drainage link.
#[derive(Clone, Debug, PartialEq)]
pub struct SwmmLinkState {
    /// Zero-based SWMM link index.
    pub index: usize,
    /// Link identifier from the `.inp` file.
    pub id: String,
    /// Current discharge in cubic meters per second.
    pub flow_m3_per_s: f64,
    /// Current water depth in meters.
    pub depth_m: f64,
    /// Current link storage volume in cubic meters.
    pub volume_m3: f64,
    /// Current realized control setting (dimensionless, e.g. `0.0..=1.0`).
    pub setting: f64,
    /// Current target control setting (dimensionless).
    pub target_setting: f64,
    /// Simulation time in days when the link setting was last changed.
    pub time_last_set_days: f64,
}

/// Cumulative flow-routing continuity ledger reported by SWMM, converted to SI cubic meters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwmmContinuityLedger {
    /// Initial stored water volume in the drainage network (`m³`).
    pub initial_storage_m3: f64,
    /// Cumulative external inflow volume (`m³`), including coupled `SET_INFLOW` volume.
    pub external_inflow_m3: f64,
    /// Cumulative outfall discharge volume (`m³`).
    pub outflow_m3: f64,
    /// Cumulative nodal flooding volume (`m³`).
    pub flooding_m3: f64,
    /// Cumulative evaporation loss volume (`m³`).
    pub evaporation_loss_m3: f64,
    /// Cumulative seepage / exfiltration loss volume (`m³`).
    pub seepage_loss_m3: f64,
    /// Current stored water volume in the drainage network (`m³`).
    pub final_storage_m3: f64,
    /// SWMM routing continuity error in percent (`100 * (1 - total_out / total_in)`).
    pub continuity_error_pct: f64,
}

/// Full canonical SI snapshot of observable SWMM drainage state at an accepted time.
#[derive(Clone, Debug, PartialEq)]
pub struct SwmmDrainageState {
    /// Simulation elapsed time in seconds.
    pub time_s: f64,
    /// Last dynamic-wave variable routing step in seconds.
    pub variable_step_s: f64,
    /// Cumulative SI continuity ledger.
    pub continuity: SwmmContinuityLedger,
    /// Per-node SI states in SWMM index order.
    pub nodes: Vec<SwmmNodeState>,
    /// Per-link SI states in SWMM index order.
    pub links: Vec<SwmmLinkState>,
}

/// Portable complete hidden-state checkpoint for `SwmmDrainage`.
#[derive(Clone, Debug, PartialEq)]
pub struct SwmmDrainageSnapshotData {
    /// Snapshot schema version (`SWMM_DRAINAGE_SNAPSHOT_SCHEMA`).
    pub schema: String,
    /// Component contract version (`SWMM_DRAINAGE_CONTRACT_VERSION`).
    pub contract: String,
    /// Lowercase SHA-256 digest of the `.inp` file text.
    pub inp_sha256: String,
    /// Declared vertical datum.
    pub vertical_datum: String,
    /// Declared surcharge formulation.
    pub surcharge_method: SwmmSurchargeMethod,
    /// Accepted simulation time in seconds.
    pub time_s: f64,
    /// Number of accepted steps since model open.
    pub accepted_steps: u64,
    /// Complete worker binary hidden-state payload encoded as lowercase hex.
    pub raw_state_hex: String,
    /// Lowercase SHA-256 digest of `raw_state_hex` bytes.
    pub state_digest_sha256: String,
}
