//! Complete global depths and separate pervious soil state for each topological phase.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::{TensorSubareaGroup, TensorSubcatchmentContext};
use crate::runoff::{SubcatchmentError, SubcatchmentState};
use crate::soil::{ModifiedGreenAmptBatch, NativeSoilState};

#[derive(Clone, Debug)]
pub(crate) struct TensorSubcatchmentState<R: Runtime<DType = DType>> {
    pub depth: Tensor<R>,
    pub phase1_soil: Option<ModifiedGreenAmptBatch<R>>,
    pub phase2_soil: Option<ModifiedGreenAmptBatch<R>>,
}
impl<R> TensorSubcatchmentState<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
    pub(crate) fn new(
        context: &TensorSubcatchmentContext<R>,
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
    pub(crate) fn continuation_from(&self, snapshot: &Self) -> Result<Self, SubcatchmentError> {
        let restore_soil = |target: &Option<ModifiedGreenAmptBatch<R>>,
                            source: &Option<ModifiedGreenAmptBatch<R>>| {
            match (target, source) {
                (Some(target), Some(source)) => {
                    let mut target = target.clone();
                    target
                        .restore_trusted(source)
                        .map_err(SubcatchmentError::from)?;
                    Ok(Some(target))
                }
                (None, None) => Ok(None),
                _ => Err(SubcatchmentError::input(
                    "trusted soil phase support differs. Restore into identical subcatchment definitions",
                )),
            }
        };
        Ok(Self {
            depth: snapshot.depth.clone(),
            phase1_soil: restore_soil(&self.phase1_soil, &snapshot.phase1_soil)?,
            phase2_soil: restore_soil(&self.phase2_soil, &snapshot.phase2_soil)?,
        })
    }

    pub(crate) fn project(
        &self,
        context: &TensorSubcatchmentContext<R>,
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
fn soil_batch<R>(
    group: Option<&TensorSubareaGroup<R>>,
    states: &[SubcatchmentState],
) -> Result<Option<ModifiedGreenAmptBatch<R>>, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
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
        ModifiedGreenAmptBatch::<R>::from_native_states(
            group.reservoir.client.clone(),
            &group.soil_parameters,
            &native,
        )
        .map_err(|e| group.map_soil_error(e))?,
    ))
}
fn project_soil<R>(
    group: Option<&TensorSubareaGroup<R>>,
    batch: Option<&ModifiedGreenAmptBatch<R>>,
    soils: &mut [Option<NativeSoilState>],
) -> Result<(), SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedNewtonAlgorithms<R>,
{
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
