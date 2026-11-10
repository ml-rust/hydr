//! Persistent original curve tensors and ordered segment-motion witnesses.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::{GateTrajectoryAssignment, StructureError};
use std::collections::BTreeMap;

type CurveIdentity = Vec<(u64, u64)>;
type CurveGroup = (usize, Vec<usize>);

#[derive(Debug)]
pub(crate) struct CpuTrajectoryCurve {
    pub times: CpuTensor,
    pub openings: CpuTensor,
    pub indices: CpuTensor,
    pub assignment: usize,
    pub count: usize,
}
#[derive(Debug)]
pub(crate) struct CpuTrajectoryContext {
    pub curves: Vec<CpuTrajectoryCurve>,
    pub absolute_limit: CpuTensor,
    pub relative_limit: CpuTensor,
    pub maximum: CpuTensor,
}
impl CpuTrajectoryContext {
    pub(crate) fn new(
        context: &CpuStructureContext,
        assignments: &[GateTrajectoryAssignment],
        slots: &[usize],
    ) -> Result<Self, StructureError> {
        crate::numerics::structures::gate::trajectory::check_motion(context, assignments, slots)?;
        let mut groups: BTreeMap<CurveIdentity, CurveGroup> = BTreeMap::new();
        let n = context.elements.len();
        let (mut absolute, mut relative, mut maximum) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        for (assignment, (definition, &slot)) in assignments.iter().zip(slots).enumerate() {
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
            if let crate::structures::StructureLaw::HydrostaticRectangularGate(parameters) =
                &context.elements[slot].law
            {
                maximum[slot] = parameters.maximum_opening_m();
            }
        }
        let mut groups = groups.into_values().collect::<Vec<_>>();
        groups.sort_by_key(|(assignment, _)| *assignment);
        let mut curves = Vec::with_capacity(groups.len());
        for (assignment, slots) in groups {
            let knots = assignments[assignment].trajectory.knots();
            curves.push(CpuTrajectoryCurve {
                times: context.tensor(&knots.iter().map(|k| k.time_s).collect::<Vec<_>>())?,
                openings: context.tensor(&knots.iter().map(|k| k.opening_m).collect::<Vec<_>>())?,
                indices: indices(context, &slots)?,
                assignment,
                count: slots.len(),
            });
        }
        Ok(Self {
            curves,
            absolute_limit: context.tensor(&absolute)?,
            relative_limit: context.tensor(&relative)?,
            maximum: context.tensor(&maximum)?,
        })
    }
}
pub(crate) fn indices(
    context: &CpuStructureContext,
    slots: &[usize],
) -> Result<CpuTensor, StructureError> {
    let values = slots
        .iter()
        .map(|&i| {
            i64::try_from(i)
                .map_err(|_| StructureError::numerical("trajectory count exceeds I64 indexing"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CpuTensor::from_slice(
        &values,
        &[values.len()],
        &context.device,
    )?)
}
