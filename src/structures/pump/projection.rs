//! Explicit evaluated pump operand projection, absent for isolated operations.
use numr::{dtype::DType, runtime::Runtime};

use super::PumpHydraulicOperands;
use crate::structures::projection::read;
use crate::structures::tensor::pump::TensorPumpResult;
use crate::structures::{StructureError, StructureOperatingPoint};

pub(crate) struct PumpOperandProjection {
    head_high: Vec<f64>,
    head_low: Vec<f64>,
    reference_head: Vec<f64>,
    error_bound: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
    reference_discharge: Vec<f64>,
    division_exact: Vec<f64>,
}
impl PumpOperandProjection {
    pub(crate) fn new<R: Runtime<DType = DType>>(
        result: &TensorPumpResult<R>,
        n: usize,
    ) -> Result<Self, StructureError> {
        Ok(Self {
            head_high: read(&result.head_high, n)?,
            head_low: read(&result.head_low, n)?,
            reference_head: read(&result.reference_head, n)?,
            error_bound: read(&result.error_bound, n)?,
            lower: read(&result.lower, n)?,
            upper: read(&result.upper, n)?,
            reference_discharge: read(&result.reference_discharge, n)?,
            division_exact: read(&result.division_exact, n)?,
        })
    }
    pub(crate) fn at(
        &self,
        i: usize,
        operation: StructureOperatingPoint,
    ) -> Result<PumpHydraulicOperands, StructureError> {
        let StructureOperatingPoint::PumpRunning { speed_ratio } = operation else {
            return Err(StructureError::numerical(
                "evaluated pump operands require original running operation",
            )
            .at(i));
        };
        Ok(PumpHydraulicOperands {
            speed_ratio,
            receiving_minus_donor_head_m: self.head_high[i],
            receiving_minus_donor_head_low_m: self.head_low[i],
            reference_head_m: self.reference_head[i],
            normalization_error_bound_m: self.error_bound[i],
            reference_head_lower_m: self.lower[i],
            reference_head_upper_m: self.upper[i],
            reference_discharge_m3_per_s: self.reference_discharge[i],
            division_exact: self.division_exact[i] != 0.0,
        })
    }
}
