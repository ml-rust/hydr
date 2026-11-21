//! Culvert roots of the latest law evaluation inside one coordinator attempt.
//!
//! Each law evaluation of an attempt starts its culvert root searches at the roots the one
//! before it left, then leaves its own. The roots only set where each search starts, and every
//! enclosure stays certified. The accepted start laws seed them, so every attempt of a trial
//! runs the same sequence, and a restart repeats it bit for bit.

use crate::structures::StructureError;
use hydr_structures::internal::{CouplingStructures, CulvertHint, TensorStructureResult};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    ReduceOps, ScalarOps, ShapeOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use std::cell::RefCell;

/// Culvert roots of one attempt's latest law evaluation.
pub(crate) struct CulvertRoots<R: Runtime<DType = DType>>(RefCell<Option<CulvertHint<R>>>);

impl<R: Runtime<DType = DType>> CulvertRoots<R> {
    /// Roots seeded by the accepted start `law`, if any.
    pub fn new(law: Option<&TensorStructureResult<R>>) -> Self {
        Self(RefCell::new(
            law.and_then(|law| law.culvert_hint()).cloned(),
        ))
    }

    /// Laws of `structures` at `stage`, `bed` and `wet` of endpoints `[A, B]`, with the culvert
    /// root searches started at the latest roots. Their own roots then become the latest.
    pub fn evaluate(
        &self,
        structures: &CouplingStructures<R>,
        stage: [&Tensor<R>; 2],
        bed: [&Tensor<R>; 2],
        wet: [&Tensor<R>; 2],
    ) -> Result<TensorStructureResult<R>, StructureError>
    where
        R::Client: CumulativeOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + CorrectlyRoundedOps<R>
            + IndexingOps<R>
            + ReduceOps<R>
            + ScalarOps<R>
            + ShapeOps<R>
            + TwoSumOps<R>
            + TypeConversionOps<R>
            + UnaryOps<R>
            + Interp1dAlgorithms<R>,
    {
        let hint = self.0.borrow().clone();
        let law = structures.evaluate_heads_from(stage, bed, wet, hint.as_ref())?;
        if let Some(latest) = law.culvert_hint() {
            self.0.replace(Some(latest.clone()));
        }
        Ok(law)
    }
}
