//! Persistent original curve tensors and ordered segment-motion witnesses.
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::tensor::TensorStructureContext;
use crate::structures::{GateTrajectoryAssignment, StructureError};
use numr::tensor::Tensor;
use std::collections::BTreeMap;

type CurveIdentity = Vec<(u64, u64)>;
type CurveGroup = (usize, Vec<usize>);

#[derive(Debug)]
pub(crate) struct TensorTrajectoryCurve<R: Runtime<DType = DType>> {
    pub times: Tensor<R>,
    pub openings: Tensor<R>,
    pub assignment: usize,
    pub count: usize,
}
#[derive(Debug)]
pub(crate) struct TensorTrajectoryContext<R: Runtime<DType = DType>> {
    pub curves: Vec<TensorTrajectoryCurve<R>>,
    pub combined_indices: Tensor<R>,
    pub assigned_mask: Tensor<R>,
    pub absolute_limit: Tensor<R>,
    pub relative_limit: Tensor<R>,
    pub maximum: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorTrajectoryContext<R> {
    pub(crate) fn new(
        context: &TensorStructureContext<R>,
        assignments: &[GateTrajectoryAssignment],
        slots: &[usize],
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        crate::structures::numerics::gate::trajectory::check_motion(context, assignments, slots)?;
        let mut groups: BTreeMap<CurveIdentity, CurveGroup> = BTreeMap::new();
        let n = context.elements.len();
        let (mut absolute, mut relative, mut maximum) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut assigned = vec![0.0; n];
        for (assignment, (definition, &slot)) in assignments.iter().zip(slots).enumerate() {
            assigned[slot] = 1.0;
            let knots = definition.trajectory.knots();
            let key = knots
                .iter()
                .map(|k| (k.time_s.to_bits(), k.opening_m.to_bits()))
                .collect();
            groups
                .entry(key)
                .or_insert_with(|| (assignment, Vec::new()))
                .1
                .push(slot);
            absolute[slot] = definition
                .trajectory
                .accuracy()
                .maximum_absolute_opening_error_m;
            relative[slot] = definition
                .trajectory
                .accuracy()
                .maximum_relative_opening_error;
            if let Some(limit) =
                crate::structures::blockage::effective::gate_opening_limit(&context.elements[slot])
            {
                maximum[slot] = limit;
            }
        }
        let mut groups = groups.into_values().collect::<Vec<_>>();
        groups.sort_by_key(|(assignment, _)| *assignment);
        let mut curves = Vec::with_capacity(groups.len());
        let mut combined_slots = Vec::with_capacity(slots.len());
        for (assignment, slots) in groups {
            let knots = assignments[assignment].trajectory.knots();
            combined_slots.extend_from_slice(&slots);
            curves.push(TensorTrajectoryCurve {
                times: context.tensor(&knots.iter().map(|k| k.time_s).collect::<Vec<_>>())?,
                openings: context.tensor(&knots.iter().map(|k| k.opening_m).collect::<Vec<_>>())?,
                assignment,
                count: slots.len(),
            });
        }
        Ok(Self {
            curves,
            combined_indices: indices(context, &combined_slots)?,
            assigned_mask: context.tensor(&assigned)?,
            absolute_limit: context.tensor(&absolute)?,
            relative_limit: context.tensor(&relative)?,
            maximum: context.tensor(&maximum)?,
        })
    }
}
pub(crate) fn indices<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    slots: &[usize],
) -> Result<Tensor<R>, StructureError> {
    let values = slots
        .iter()
        .map(|&i| {
            i64::try_from(i)
                .map_err(|_| StructureError::numerical("trajectory count exceeds I64 indexing"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Tensor::<R>::from_slice(
        &values,
        &[values.len()],
        &context.device,
    )?)
}
