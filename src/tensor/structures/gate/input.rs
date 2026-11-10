//! Supplied opening upload or retained trajectory tensors for immutable proposals.
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::{StructureError, StructureOperatingPoint};
use crate::tensor::structures::{TensorStructureContext, operations::CheckedOperations};
use numr::tensor::Tensor;

#[derive(Debug)]
pub(crate) struct TensorGateInput<R: Runtime<DType = DType>> {
    pub opening: Tensor<R>,
}
impl<R: Runtime<DType = DType>> TensorGateInput<R> {
    pub(crate) fn new(
        context: &TensorStructureContext<R>,
        operations: &CheckedOperations<'_, R>,
    ) -> Result<Option<Self>, StructureError>
    where
        R::Client: numr::ops::BinaryOps<R>
            + numr::ops::ConditionalOps<R>
            + numr::ops::ReduceOps<R>
            + numr::ops::TypeConversionOps<R>
            + numr::ops::UnaryOps<R>,
    {
        if context.gate.is_none() {
            return Ok(None);
        }
        let values = (0..operations.len())
            .map(|index| match operations.supplied(index) {
                Some(StructureOperatingPoint::GateOpening { opening_m }) => opening_m,
                _ => 0.0,
            })
            .collect::<Vec<_>>();
        let manual = context.tensor(&values)?;
        let opening = match operations.derived_opening() {
            Some((derived, mask)) => context.client.where_cond(mask, derived, &manual)?,
            None => manual,
        };
        context.finite(&opening, "assembled finite actual gate opening")?;
        Ok(Some(Self { opening }))
    }
}
