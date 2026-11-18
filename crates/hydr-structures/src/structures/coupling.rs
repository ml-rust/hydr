//! Cached static operations for device-resident connected-pool trials.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::{
    StructureBatch, StructureElement, StructureError, StructureInputErrorKind, StructureOperation,
};
use crate::structures::tensor::{
    TensorStaticOperations, TensorStructureHeads, TensorStructureInput, TensorStructureResult,
    check_structure_tensor,
};

#[doc(hidden)]
#[derive(Debug)]
pub struct CouplingStructures<R: Runtime<DType = DType>> {
    batch: StructureBatch<R>,
    tensors: TensorStaticOperations<R>,
}

#[cfg(test)]
mod tests;

impl<R: Runtime<DType = DType>> CouplingStructures<R> {
    #[doc(hidden)]
    pub fn new(
        client: R::Client,
        elements: Vec<StructureElement>,
        operations: Vec<StructureOperation>,
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let batch = StructureBatch::<R>::new(client, elements)?;
        let tensors = TensorStaticOperations::<R>::new(&batch.context, &operations)?;
        Ok(Self { batch, tensors })
    }

    /// The same checked laws under `operations`, one per element in element order.
    ///
    /// The elements stay checked once. Only the operation tensors change.
    #[doc(hidden)]
    pub fn with_operations(&self, operations: &[StructureOperation]) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let tensors = TensorStaticOperations::<R>::new(&self.batch.context, operations)?;
        Ok(Self {
            batch: self.batch.clone(),
            tensors,
        })
    }

    #[doc(hidden)]
    pub fn evaluate_heads(
        &self,
        stage_a: &Tensor<R>,
        stage_b: &Tensor<R>,
        bed_a: &Tensor<R>,
        bed_b: &Tensor<R>,
        wet_a: &Tensor<R>,
        wet_b: &Tensor<R>,
    ) -> Result<TensorStructureResult<R>, StructureError>
    where
        R::Client: numr::ops::CumulativeOps<R>
            + numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::CorrectlyRoundedOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::ShapeOps<R>
            + numr::ops::TwoSumOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>
            + solvr::interpolate::Interp1dAlgorithms<R>,
    {
        let context = &self.batch.context;
        let heads = TensorStructureHeads::from_tensors(
            context, stage_a, stage_b, bed_a, bed_b, wet_a, wet_b,
        )?;
        let input = TensorStructureInput::with_static_operations(context, heads, &self.tensors)?;
        crate::structures::numerics::proposal(context, &input)
    }

    #[doc(hidden)]
    pub fn held_volumes(
        &self,
        result: &TensorStructureResult<R>,
        dt_s: f64,
    ) -> Result<Tensor<R>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::CompareOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::ScalarOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        if !dt_s.is_finite() || dt_s <= 0.0 {
            return Err(StructureError::input(
                StructureInputErrorKind::Definition,
                format!("held duration={dt_s}: supply a finite positive duration in seconds"),
            ));
        }
        check_structure_tensor(&self.batch.context, &result.discharge, "discharge")?;
        crate::structures::numerics::held::volumes(&self.batch.context, &result.discharge, dt_s)
    }
}
