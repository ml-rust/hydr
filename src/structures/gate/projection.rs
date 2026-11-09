//! Explicit gate geometry projection without host-side hydraulic advancement.

use super::GateHydraulicOperands;
use crate::cpu::structures::gate::CpuGateResult;
use crate::structures::{StructureError, StructureOperatingPoint, projection::read};

pub(crate) struct GateOperandProjection {
    opening: Vec<f64>,
    crown_high: Vec<f64>,
    crown_low: Vec<f64>,
    crown_overflow: Vec<f64>,
    submerged_height: Vec<f64>,
    submerged_height_bound: Vec<f64>,
    free_height: Vec<f64>,
    free_height_bound: Vec<f64>,
    gap: Vec<f64>,
    gap_bound: Vec<f64>,
    lower: Vec<f64>,
    lower_bound: Vec<f64>,
    upper: Vec<f64>,
    upper_bound: Vec<f64>,
    ratio: Vec<f64>,
    k: Vec<f64>,
    closure_residual: Vec<f64>,
    closure_uncertainty: Vec<f64>,
    closure_allowance: Vec<f64>,
    logarithm: Vec<f64>,
}
impl GateOperandProjection {
    pub(crate) fn new(result: &CpuGateResult, n: usize) -> Result<Self, StructureError> {
        let g = &result.geometry;
        Ok(Self {
            opening: read(&g.opening, n)?,
            crown_high: read(&g.crown_high, n)?,
            crown_low: read(&g.crown_low, n)?,
            crown_overflow: read(&g.crown_overflow, n)?,
            submerged_height: read(&g.submerged_height, n)?,
            submerged_height_bound: read(&g.submerged_height_bound, n)?,
            free_height: read(&g.free_height, n)?,
            free_height_bound: read(&g.free_height_bound, n)?,
            gap: read(&g.gap, n)?,
            gap_bound: read(&g.gap_bound, n)?,
            lower: read(&g.lower, n)?,
            lower_bound: read(&g.lower_bound, n)?,
            upper: read(&g.upper, n)?,
            upper_bound: read(&g.upper_bound, n)?,
            ratio: read(&result.ratio, n)?,
            k: read(&result.k, n)?,
            closure_residual: read(&g.closure_residual, n)?,
            closure_uncertainty: read(&g.closure_uncertainty, n)?,
            closure_allowance: read(&g.closure_allowance, n)?,
            logarithm: read(&result.logarithm, n)?,
        })
    }
    pub(crate) fn at(
        &self,
        index: usize,
        operation: StructureOperatingPoint,
    ) -> Result<GateHydraulicOperands, StructureError> {
        let StructureOperatingPoint::GateOpening { opening_m } = operation else {
            return Err(StructureError::numerical(
                "evaluated gate operands require an original opening operation",
            )
            .at(index));
        };
        if opening_m != self.opening[index] {
            return Err(StructureError::numerical(
                "evaluated gate operands changed the original opening",
            )
            .at(index));
        }
        Ok(GateHydraulicOperands {
            opening_m,
            crown_high_m: self.crown_high[index],
            crown_low_m: self.crown_low[index],
            crown_overflow: self.crown_overflow[index] != 0.0,
            submerged_height_m: self.submerged_height[index],
            submerged_height_error_bound_m: self.submerged_height_bound[index],
            free_height_m: self.free_height[index],
            free_height_error_bound_m: self.free_height_bound[index],
            submerged_head_gap_m: self.gap[index],
            submerged_head_gap_error_bound_m: self.gap_bound[index],
            free_lower_head_m: self.lower[index],
            free_lower_head_error_bound_m: self.lower_bound[index],
            free_upper_head_m: self.upper[index],
            free_upper_head_error_bound_m: self.upper_bound[index],
            free_head_ratio: self.ratio[index],
            free_integral_factor_k: self.k[index],
            closure_residual_m: self.closure_residual[index],
            closure_uncertainty_bound_m: self.closure_uncertainty[index],
            closure_physical_allowance_m: self.closure_allowance[index],
            log_discharge_magnitude: self.logarithm[index],
        })
    }
}
