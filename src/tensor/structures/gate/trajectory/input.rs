//! Original-time support metadata without interpolated host state.
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::{GateTrajectoryAssignment, StructureError};
use crate::tensor::structures::TensorStructureContext;
use numr::tensor::Tensor;

#[derive(Clone, Debug)]
pub(crate) struct TrajectoryPointMetadata {
    pub segment_index: usize,
    pub exact_knot_index: Option<usize>,
    pub closed_by_definition: bool,
}
#[derive(Debug)]
pub(crate) struct TensorTrajectoryInput<R: Runtime<DType = DType>> {
    pub metadata: Vec<TrajectoryPointMetadata>,
    pub closed: Tensor<R>,
    pub minimum: Tensor<R>,
    pub maximum: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorTrajectoryInput<R> {
    pub(crate) fn new(
        context: &TensorStructureContext<R>,
        assignments: &[GateTrajectoryAssignment],
        slots: &[usize],
        time_s: f64,
    ) -> Result<Self, StructureError> {
        let n = context.elements.len();
        let (mut closed, mut minimum, mut maximum) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut metadata = Vec::with_capacity(assignments.len());
        for (assignment, &slot) in assignments.iter().zip(slots) {
            let knots = assignment.trajectory.knots();
            let right = knots.partition_point(|k| k.time_s <= time_s);
            let segment_index = right.saturating_sub(1).min(knots.len() - 2);
            let exact_knot_index = right
                .checked_sub(1)
                .filter(|&index| knots[index].time_s == time_s);
            let pair = &knots[segment_index..=segment_index + 1];
            let closed_by_definition = exact_knot_index
                .map_or(pair[0].opening_m == 0.0 && pair[1].opening_m == 0.0, |i| {
                    knots[i].opening_m == 0.0
                });
            closed[slot] = if closed_by_definition { 1.0 } else { 0.0 };
            minimum[slot] = pair[0].opening_m.min(pair[1].opening_m);
            maximum[slot] = pair[0].opening_m.max(pair[1].opening_m);
            metadata.push(TrajectoryPointMetadata {
                segment_index,
                exact_knot_index,
                closed_by_definition,
            });
        }
        Ok(Self {
            metadata,
            closed: context.tensor(&closed)?,
            minimum: context.tensor(&minimum)?,
            maximum: context.tensor(&maximum)?,
        })
    }
}
