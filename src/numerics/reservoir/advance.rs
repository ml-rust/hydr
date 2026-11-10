//! Primitive source preparation followed by the shared conservative drainage core.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::check::finish;
use super::receipt::TensorReservoirStep;
use super::{algebra, drainage};
use crate::runoff::ReservoirError;
use crate::tensor::reservoir_storage::{TensorReservoirContext, TensorReservoirState};

pub(crate) fn advance<R>(
    context: &TensorReservoirContext<R>,
    state: &TensorReservoirState<R>,
    dt: f64,
    incoming: &Tensor<R>,
) -> Result<TensorReservoirStep<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let prepared = algebra::prepare(context, &state.depth, incoming, dt)?;
    let drained = drainage::drain(&prepared.hydraulic)?;
    finish(context, prepared, &state.depth, incoming, drained)
}
