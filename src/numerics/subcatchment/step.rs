//! Persistent complete source, routing, and drainage tensor receipts.

use super::bridge::Bridge;
use super::transfer::{Merged, Partition};
use crate::cpu::CpuTensor;
use crate::cpu::subcatchment::CpuSubcatchmentInput;
use crate::numerics::reservoir::drainage::DrainageResult;

#[derive(Clone, Debug)]
pub(crate) struct CpuSubcatchmentSoilStep {
    pub infiltration_depth: CpuTensor,
    pub adjustment_depth: CpuTensor,
    pub native_rate: CpuTensor,
    pub native_available_depth: CpuTensor,
    pub native_infiltration_depth: CpuTensor,
    pub native_excess_depth: CpuTensor,
    pub native_roundoff_bound: CpuTensor,
    pub bridge_roundoff_bound: CpuTensor,
    pub total_roundoff_bound: CpuTensor,
}

#[derive(Clone, Debug)]
pub(crate) struct CpuSubareaStep {
    pub old_depth: CpuTensor,
    pub rain_rate: CpuTensor,
    pub potential_evaporation: CpuTensor,
    pub external_volume: CpuTensor,
    pub received_by_donor: [CpuTensor; 3],
    pub merged: Merged,
    pub bridge: Bridge,
    pub soil: Option<CpuSubcatchmentSoilStep>,
    pub infiltration_depth: CpuTensor,
    pub adjustment_depth: CpuTensor,
    pub evaporation_depth: CpuTensor,
    pub available_depth: CpuTensor,
    pub timing_rate: CpuTensor,
    pub direct_timing: CpuTensor,
    pub active_duration: CpuTensor,
    pub drainage: DrainageResult,
    pub partition: Partition,
    pub storage_before: CpuTensor,
    pub rain_volume: CpuTensor,
    pub internal_received_volume: CpuTensor,
    pub infiltration_volume: CpuTensor,
    pub evaporation_volume: CpuTensor,
    pub adjustment_volume: CpuTensor,
    pub water_residual: CpuTensor,
    pub water_scale: CpuTensor,
    pub rain_area_bound: CpuTensor,
    pub adjustment_charge: f64,
    pub uncertainty_charge: f64,
    pub physical_scale_lower: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct CpuSubcatchmentTotals {
    pub storage_before: CpuTensor,
    pub storage_after: CpuTensor,
    pub applied_rain: CpuTensor,
    pub external_runon: CpuTensor,
    pub infiltration: CpuTensor,
    pub evaporation: CpuTensor,
    pub outlet: CpuTensor,
    pub soil_adjustment: CpuTensor,
    pub internal_received: CpuTensor,
    pub water_residual: CpuTensor,
    pub water_scale: CpuTensor,
    pub rain_area_bound: CpuTensor,
    pub catchment_water_residual: CpuTensor,
    pub catchment_water_scale: CpuTensor,
    pub catchment_rain_area_difference: CpuTensor,
    pub catchment_rain_area_bound: CpuTensor,
    pub adjustment_charge: f64,
    pub uncertainty_charge: f64,
    pub physical_scale_lower: f64,
    pub root_iterations: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct CpuSubcatchmentStep {
    pub old_depth: CpuTensor,
    pub input: CpuSubcatchmentInput,
    pub depth: CpuTensor,
    pub phase1: Option<CpuSubareaStep>,
    pub phase2: Option<CpuSubareaStep>,
    pub received_by_donor: [CpuTensor; 3],
    pub totals: CpuSubcatchmentTotals,
}
