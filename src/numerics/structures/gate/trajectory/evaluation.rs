//! Checked Linear point evaluation without derived-opening host projection.

use crate::cpu::{
    CpuTensor,
    structures::{
        CpuStructureContext,
        gate::trajectory::{CpuTrajectoryContext, CpuTrajectoryInput},
    },
};
use crate::structures::{GateTrajectoryAssignment, StructureError};
use numr::ops::{IndexingOps, ShapeOps};
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
    let mut opening_parts = Vec::with_capacity(trajectory.curves.len());
    let mut error_parts = Vec::with_capacity(trajectory.curves.len());
    for curve in &trajectory.curves {
        let query = context.tensor(&vec![time_s; curve.count])?;
        let receipt = c
            .interp1d_linear_with_receipts(&curve.times, &curve.openings, &query)
            .map_err(|error| StructureError::from(error).at(slots[curve.assignment]))?;
        opening_parts.push(receipt.values().clone());
        error_parts.push(receipt.absolute_error().clone());
    }
    let (opening, error) = if trajectory.curves.is_empty() {
        (context.zero.clone(), context.zero.clone())
    } else {
        let opening = match opening_parts.as_slice() {
            [single] => single.clone(),
            parts => c.cat(&parts.iter().collect::<Vec<_>>(), 0)?,
        };
        let error = match error_parts.as_slice() {
            [single] => single.clone(),
            parts => c.cat(&parts.iter().collect::<Vec<_>>(), 0)?,
        };
        (
            c.index_put(&context.zero, 0, &trajectory.combined_indices, &opening)?,
            c.index_put(&context.zero, 0, &trajectory.combined_indices, &error)?,
        )
    };
    super::accuracy::check(context, trajectory, &input, &opening, &error)?;
    Ok((opening, error, input))
}
