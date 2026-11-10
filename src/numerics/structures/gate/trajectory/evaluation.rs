//! Checked Linear point evaluation without derived-opening host projection.

use crate::cpu::{
    CpuTensor,
    structures::{
        CpuStructureContext,
        gate::trajectory::{CpuTrajectoryContext, CpuTrajectoryInput},
    },
};
use crate::structures::{GateTrajectoryAssignment, StructureError};
use numr::ops::IndexingOps;
use solvr::interpolate::LinearInterp1dDiagnostics;

pub(crate) fn evaluate(
    context: &CpuStructureContext,
    trajectory: &CpuTrajectoryContext,
    assignments: &[GateTrajectoryAssignment],
    slots: &[usize],
    time_s: f64,
) -> Result<(CpuTensor, CpuTensor, CpuTrajectoryInput), StructureError> {
    let input = CpuTrajectoryInput::new(context, assignments, slots, time_s)?;
    let c = &context.client;
    let mut opening = context.zero.clone();
    let mut error = context.zero.clone();
    for curve in &trajectory.curves {
        let query = context.tensor(&vec![time_s; curve.count])?;
        let receipt = c
            .interp1d_linear_with_receipts(&curve.times, &curve.openings, &query)
            .map_err(|error| StructureError::from(error).at(slots[curve.assignment]))?;
        opening = c.index_put(&opening, 0, &curve.indices, receipt.values())?;
        error = c.index_put(&error, 0, &curve.indices, receipt.absolute_error())?;
    }
    super::accuracy::check(context, trajectory, &input, &opening, &error)?;
    Ok((opening, error, input))
}
