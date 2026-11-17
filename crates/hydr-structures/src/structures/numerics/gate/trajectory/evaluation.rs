//! Checked Linear point evaluation without derived-opening host projection.
use numr::ops::IndexingOps;
use numr::ops::ShapeOps;
use numr::{dtype::DType, runtime::Runtime};
use solvr::interpolate::LinearInterp1dDiagnostics;

use crate::structures::tensor::{
    TensorStructureContext,
    gate::trajectory::{TensorTrajectoryContext, TensorTrajectoryInput},
};
use crate::structures::{GateTrajectoryAssignment, StructureError};
use numr::tensor::Tensor;

type TrajectoryPointTensors<R> = (Tensor<R>, Tensor<R>, TensorTrajectoryInput<R>);

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    trajectory: &TensorTrajectoryContext<R>,
    assignments: &[GateTrajectoryAssignment],
    slots: &[usize],
    time_s: f64,
) -> Result<TrajectoryPointTensors<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::IndexingOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::ShapeOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>
        + solvr::interpolate::LinearInterp1dDiagnostics<R>,
{
    let input = TensorTrajectoryInput::new(context, assignments, slots, time_s)?;
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
