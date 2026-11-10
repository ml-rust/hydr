//! Opaque immutable point tensors with explicit gate-only projections.
use numr::{dtype::DType, runtime::Runtime};

use super::binding::GateTrajectoryData;
use crate::structures::{StructureError, StructureOperatingPoint, StructureOperation};
use crate::tensor::structures::gate::trajectory::TrajectoryPointMetadata;
use numr::tensor::Tensor;
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct GateTrajectoryEvaluationData<R: Runtime<DType = DType>> {
    pub binding: Arc<GateTrajectoryData<R>>,
    pub time_s: f64,
    pub opening: Tensor<R>,
    pub absolute_error: Tensor<R>,
    pub metadata: Vec<TrajectoryPointMetadata>,
}
/// Immutable tensor-backed prescribed point. Projection never repeats interpolation.
#[derive(Clone, Debug)]
pub struct GateTrajectoryEvaluation<R: Runtime<DType = DType>> {
    pub(crate) inner: Arc<GateTrajectoryEvaluationData<R>>,
}
/// One explicitly projected gate point and original-time support metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct GateTrajectoryOutput {
    /// Element index in the complete structure batch.
    pub element: usize,
    /// Original structure identity.
    pub structure_id: String,
    /// Original query time in seconds.
    pub time_s: f64,
    /// Represented opening in metres.
    pub opening_m: f64,
    /// Solvr absolute error enclosure in metres.
    pub absolute_error_m: f64,
    /// Right-sided original segment, except at the terminal knot.
    pub segment_index: usize,
    /// Original knot index when time equality holds.
    pub exact_knot_index: Option<usize>,
    /// Original zero knot or zero plateau contains the query.
    pub closed_by_definition: bool,
}
impl<R: Runtime<DType = DType>> GateTrajectoryEvaluation<R> {
    /// Original supported query time.
    pub fn time_s(&self) -> f64 {
        self.inner.time_s
    }
    /// Project gate rows in original gate subsequence order.
    pub fn outputs(&self) -> Result<Vec<GateTrajectoryOutput>, StructureError> {
        let data = &self.inner;
        let n = data.binding.context.elements.len();
        let opening = crate::structures::projection::read(&data.opening, n)?;
        let error = crate::structures::projection::read(&data.absolute_error, n)?;
        Ok(data
            .binding
            .slots
            .iter()
            .zip(&data.metadata)
            .map(|(&element, metadata)| GateTrajectoryOutput {
                element,
                structure_id: data.binding.context.elements[element].id.clone(),
                time_s: data.time_s,
                opening_m: opening[element],
                absolute_error_m: error[element],
                segment_index: metadata.segment_index,
                exact_knot_index: metadata.exact_knot_index,
                closed_by_definition: metadata.closed_by_definition,
            })
            .collect())
    }
    /// Project gate-only identity-bound operation packets.
    pub fn operations(&self) -> Result<Vec<StructureOperation>, StructureError> {
        let data = &self.inner;
        let opening = crate::structures::projection::read(
            &data.opening,
            data.binding.context.elements.len(),
        )?;
        Ok(data
            .binding
            .slots
            .iter()
            .map(|&element| StructureOperation {
                structure_id: data.binding.context.elements[element].id.clone(),
                operating_point: StructureOperatingPoint::GateOpening {
                    opening_m: opening[element],
                },
            })
            .collect())
    }
}
