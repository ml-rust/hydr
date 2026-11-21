//! Shared complete-law point evaluation in passive, pump, gate, then culvert order.
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult,
};

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
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
    let passive = super::evaluate(context, input)?;
    let result = super::pump::evaluate(context, input, passive)?;
    let result = super::gate::evaluate(context, input, result)?;
    match &context.culvert {
        Some(culvert) => culvert.evaluate(context, input, result),
        None => Ok(result),
    }
}
