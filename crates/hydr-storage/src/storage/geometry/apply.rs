//! Applying a pool curve epoch while keeping stored volume and counters bit-identical.

use crate::storage::tensor::TensorPoolContext;
use crate::storage::{LevelPoolBatch, LevelPoolCurve, LevelPoolError, LevelPoolSnapshotData};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use std::sync::Arc;

impl<R> LevelPoolBatch<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + SortingOps<R>
        + Interp1dAlgorithms<R>,
{
    /// Returns a copy of this batch with updated stage-volume `curves`, keeping every stored
    /// volume and cumulative counter bit-identical.
    ///
    /// Rejects without modifying state when any pool's current volume exceeds its new curve's
    /// top knot, or when a curve's vertical reference differs from the declared pool's.
    #[doc(hidden)]
    pub fn coupling_with_curves(&self, curves: &[LevelPoolCurve]) -> Result<Self, LevelPoolError> {
        let current = self.elements();
        if curves.len() != current.len() {
            return Err(LevelPoolError::input(format!(
                "curve count={}, pools={}: supply one curve per pool in pool order",
                curves.len(),
                current.len()
            )));
        }
        let mut updated = Vec::with_capacity(current.len());
        for (index, (element, curve)) in current.iter().zip(curves).enumerate() {
            if curve.vertical_reference() != element.curve.vertical_reference() {
                return Err(LevelPoolError::input(format!(
                    "pool={:?} curve vertical_reference={:?} differs from declared {:?}",
                    element.id,
                    curve.vertical_reference(),
                    element.curve.vertical_reference()
                ))
                .at(index));
            }
            let mut next = element.clone();
            next.curve = curve.clone();
            updated.push(next);
        }
        let mut new_context = TensorPoolContext::<R>::new_with_sources(
            self.context.client.clone(),
            updated,
            self.context.config.clone(),
            self.context.source_policies.clone(),
        )?;
        new_context.initial_maximum = self.context.initial_maximum.clone();
        let new_context = Arc::new(new_context);
        // Check that every pool's current volume fits the new curve's closed volume support.
        crate::storage::numerics::curves::evaluate(&new_context, &self.state.volume_m3)?;
        Ok(Self {
            context: new_context,
            state: self.state.clone(),
            time: self.time,
            steps: self.steps,
            last: self.last.clone(),
            context_generation: self.context_generation.wrapping_add(1),
        })
    }

    /// Restores a coordinator-owned pool package whose curves may reflect a scheduled curve epoch.
    ///
    /// `self` must hold the original `t = 0` pool definitions; `initial_volume_m3` is checked
    /// against `self`'s original curves while `data.states` and `data.last_step` are checked
    /// against `data.elements`.
    #[doc(hidden)]
    pub fn coupling_restore_data(
        &self,
        data: &LevelPoolSnapshotData,
    ) -> Result<Self, LevelPoolError> {
        let mut batch = self.clone();
        batch.restore_data_with_curve_epochs(data)?;
        Ok(batch)
    }
}
