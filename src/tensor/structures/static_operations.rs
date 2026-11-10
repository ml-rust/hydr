//! Static manual operations retained for repeated tensor-head evaluation.

use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::operations::CheckedOperations;
use super::{TensorStructureContext, gate::TensorGateInput, pump::TensorPumpInput};
use crate::structures::{StructureError, StructureOperation};

#[derive(Debug)]
pub(crate) struct TensorStaticOperations<R: Runtime<DType = DType>> {
    gate: Option<TensorGateInput<R>>,
    pump: Option<TensorPumpInput<R>>,
    pub(super) isolated: Tensor<R>,
}

impl<R: Runtime<DType = DType>> TensorStaticOperations<R> {
    pub(crate) fn new(
        context: &TensorStructureContext<R>,
        operations: &[StructureOperation],
    ) -> Result<Self, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::IndexingOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        let checked = CheckedOperations::new(context, operations)?;
        Ok(Self {
            gate: TensorGateInput::new(context, &checked)?,
            pump: TensorPumpInput::new(context, &checked)?,
            isolated: checked.isolated().clone(),
        })
    }

    pub(super) fn gate(&self) -> Option<TensorGateInput<R>> {
        self.gate.as_ref().map(|input| TensorGateInput {
            opening: input.opening.clone(),
        })
    }

    pub(super) fn pump(&self) -> Option<TensorPumpInput<R>> {
        self.pump.as_ref().map(|input| TensorPumpInput {
            running: input.running.clone(),
            isolated: input.isolated.clone(),
            speed: input.speed.clone(),
            power_of_two: input.power_of_two.clone(),
        })
    }
}
