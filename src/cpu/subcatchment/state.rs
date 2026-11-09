//! Complete global depths and separate pervious soil state for each topological phase.

use super::{CpuSubareaGroup, CpuSubcatchmentContext};
use crate::cpu::CpuTensor;
use crate::runoff::{SubcatchmentError, SubcatchmentState};
use crate::soil::{ModifiedGreenAmptBatch, NativeSoilState};

#[derive(Clone, Debug)]
pub(crate) struct CpuSubcatchmentState {
    pub depth: CpuTensor,
    pub phase1_soil: Option<ModifiedGreenAmptBatch>,
    pub phase2_soil: Option<ModifiedGreenAmptBatch>,
}
impl CpuSubcatchmentState {
    pub(crate) fn new(
        context: &CpuSubcatchmentContext,
        states: &[SubcatchmentState],
    ) -> Result<Self, SubcatchmentError> {
        if states.len() != context.definitions.len() {
            return Err(SubcatchmentError::input(
                "complete state count differs from catchment support",
            ));
        }
        for (i, (state, definition)) in states.iter().zip(&context.definitions).enumerate() {
            state
                .check(&definition.parameters)
                .map_err(|e| e.at_catchment(i))?;
        }
        Ok(Self {
            depth: context.tensor(&states.iter().flat_map(|s| s.depths_m).collect::<Vec<_>>())?,
            phase1_soil: soil_batch(context.phase1.as_ref(), states)?,
            phase2_soil: soil_batch(context.phase2.as_ref(), states)?,
        })
    }
    pub(crate) fn project(
        &self,
        context: &CpuSubcatchmentContext,
    ) -> Result<Vec<SubcatchmentState>, SubcatchmentError> {
        let depths = self.depth.try_to_vec::<f64>()?;
        if depths.len() != context.definitions.len() * 3 {
            return Err(SubcatchmentError::numerical(
                "depth projection differs from complete support",
            ));
        }
        let mut soils = vec![None; context.definitions.len()];
        project_soil(
            context.phase1.as_ref(),
            self.phase1_soil.as_ref(),
            &mut soils,
        )?;
        project_soil(
            context.phase2.as_ref(),
            self.phase2_soil.as_ref(),
            &mut soils,
        )?;
        Ok(depths
            .as_chunks::<3>()
            .0
            .iter()
            .zip(soils)
            .map(|(d, soil)| SubcatchmentState {
                depths_m: [d[0], d[1], d[2]],
                soil,
            })
            .collect())
    }
}
fn soil_batch(
    group: Option<&CpuSubareaGroup>,
    states: &[SubcatchmentState],
) -> Result<Option<ModifiedGreenAmptBatch>, SubcatchmentError> {
    let Some(group) = group else {
        return Ok(None);
    };
    if group.soil_slots.is_empty() {
        return Ok(None);
    }
    let native: Vec<NativeSoilState> = group
        .soil_slots
        .iter()
        .map(|&lane| {
            let slot = group
                .slots
                .get(lane)
                .ok_or_else(|| SubcatchmentError::input("soil role map exceeds phase support"))?;
            states.get(slot / 3).and_then(|s| s.soil).ok_or_else(|| {
                SubcatchmentError::input("pervious role lacks complete antecedent memory")
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(Some(
        ModifiedGreenAmptBatch::from_native_states(&group.soil_parameters, &native)
            .map_err(|e| group.map_soil_error(e))?,
    ))
}
fn project_soil(
    group: Option<&CpuSubareaGroup>,
    batch: Option<&ModifiedGreenAmptBatch>,
    soils: &mut [Option<NativeSoilState>],
) -> Result<(), SubcatchmentError> {
    match (group, batch) {
        (None, None) => Ok(()),
        (Some(group), None) if group.soil_slots.is_empty() => Ok(()),
        (Some(group), Some(batch)) => {
            let states = batch.native_states().map_err(|e| group.map_soil_error(e))?;
            if states.len() != group.soil_slots.len() {
                return Err(SubcatchmentError::numerical(
                    "native projection differs from pervious role support",
                ));
            }
            for (&lane, state) in group.soil_slots.iter().zip(states) {
                let slot = group.slots.get(lane).ok_or_else(|| {
                    SubcatchmentError::numerical("soil role map exceeds phase support")
                })?;
                let destination = soils.get_mut(slot / 3).ok_or_else(|| {
                    SubcatchmentError::numerical("soil output exceeds full catchment support")
                })?;
                *destination = Some(state);
            }
            Ok(())
        }
        _ => Err(SubcatchmentError::numerical(
            "persistent soil batch and immutable phase-role support disagree",
        )),
    }
}
