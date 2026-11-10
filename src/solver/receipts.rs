use super::{EngineError, StepDiagnostics};
use crate::tensor::{
    TensorDomain, land::TensorLand, land_sources::SourceTrial, transport::TensorTransport,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

#[derive(Clone)]
pub(super) enum StepReceipt<R: Runtime<DType = DType>> {
    Device(Box<DeviceStepReceipt<R>>),
    Imported(Box<StepDiagnostics>),
}

#[derive(Clone)]
pub(super) struct DeviceStepReceipt<R: Runtime<DType = DType>> {
    pub scalar: StepDiagnostics,
    pub forcing_index: usize,
    pub flux: TensorTransport<R>,
    pub sources: SourceTrial<R>,
    pub start_depth: Tensor<R>,
}

impl<R: Runtime<DType = DType>> StepReceipt<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    pub(super) fn project(
        &self,
        context: &TensorDomain<R>,
        land: &TensorLand<R>,
    ) -> Result<StepDiagnostics, EngineError> {
        let receipt = match self {
            Self::Device(receipt) => receipt,
            Self::Imported(step) => return Ok(step.as_ref().clone()),
        };
        let source = &context.sources[receipt.forcing_index];
        let mut step = receipt.scalar.clone();
        step.faces = receipt.flux.results(context, step.dt)?;
        step.transmissive_states = source
            .exterior
            .results(receipt.flux.transmissive_state.as_ref())?;
        step.boundary_traces = match &receipt.flux.boundary_trace {
            Some(trace) => source.characteristic.results(context, trace)?,
            None => Vec::new(),
        };
        step.sources = receipt.sources.receipts(land, &receipt.start_depth)?;
        Ok(step)
    }
}
