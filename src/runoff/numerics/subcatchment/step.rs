//! Persistent complete source, routing, and drainage tensor receipts.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::bridge::Bridge;
use super::transfer::{Merged, Partition};
use crate::runoff::numerics::reservoir::drainage::DrainageResult;
use crate::runoff::tensor::subcatchment::TensorSubcatchmentInput;

#[derive(Clone, Debug)]
pub(crate) struct TensorSubcatchmentSoilStep<R: Runtime<DType = DType>> {
    pub infiltration_depth: Tensor<R>,
    pub adjustment_depth: Tensor<R>,
    pub native_rate: Tensor<R>,
    pub native_available_depth: Tensor<R>,
    pub native_infiltration_depth: Tensor<R>,
    pub native_excess_depth: Tensor<R>,
    pub native_roundoff_bound: Tensor<R>,
    pub bridge_roundoff_bound: Tensor<R>,
    pub total_roundoff_bound: Tensor<R>,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorSubareaStep<R: Runtime<DType = DType>> {
    pub old_depth: Tensor<R>,
    pub rain_rate: Tensor<R>,
    pub potential_evaporation: Tensor<R>,
    pub external_volume: Tensor<R>,
    pub received_by_donor: [Tensor<R>; 3],
    pub merged: Merged<R>,
    pub bridge: Bridge<R>,
    pub soil: Option<TensorSubcatchmentSoilStep<R>>,
    pub infiltration_depth: Tensor<R>,
    pub adjustment_depth: Tensor<R>,
    pub evaporation_depth: Tensor<R>,
    pub available_depth: Tensor<R>,
    pub timing_rate: Tensor<R>,
    pub direct_timing: Tensor<R>,
    pub active_duration: Tensor<R>,
    pub drainage: DrainageResult<R>,
    pub partition: Partition<R>,
    pub storage_before: Tensor<R>,
    pub rain_volume: Tensor<R>,
    pub internal_received_volume: Tensor<R>,
    pub infiltration_volume: Tensor<R>,
    pub evaporation_volume: Tensor<R>,
    pub adjustment_volume: Tensor<R>,
    pub water_residual: Tensor<R>,
    pub water_scale: Tensor<R>,
    pub rain_area_bound: Tensor<R>,
    pub adjustment_charge: f64,
    pub uncertainty_charge: f64,
    pub physical_scale_lower: f64,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorSubcatchmentTotals<R: Runtime<DType = DType>> {
    pub storage_before: Tensor<R>,
    pub storage_after: Tensor<R>,
    pub applied_rain: Tensor<R>,
    pub external_runon: Tensor<R>,
    pub infiltration: Tensor<R>,
    pub evaporation: Tensor<R>,
    pub outlet: Tensor<R>,
    pub soil_adjustment: Tensor<R>,
    pub internal_received: Tensor<R>,
    pub water_residual: Tensor<R>,
    pub water_scale: Tensor<R>,
    pub rain_area_bound: Tensor<R>,
    pub catchment_water_residual: Tensor<R>,
    pub catchment_water_scale: Tensor<R>,
    pub catchment_rain_area_difference: Tensor<R>,
    pub catchment_rain_area_bound: Tensor<R>,
    pub adjustment_charge: f64,
    pub uncertainty_charge: f64,
    pub physical_scale_lower: f64,
    pub root_iterations: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct TensorSubcatchmentStep<R: Runtime<DType = DType>> {
    pub old_depth: Tensor<R>,
    pub input: TensorSubcatchmentInput<R>,
    pub depth: Tensor<R>,
    pub phase1: Option<TensorSubareaStep<R>>,
    pub phase2: Option<TensorSubareaStep<R>>,
    pub received_by_donor: [Tensor<R>; 3],
    pub totals: TensorSubcatchmentTotals<R>,
}
