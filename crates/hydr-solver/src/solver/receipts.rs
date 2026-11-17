use super::StepDiagnostics;
use super::tensor::{
    TensorDomain, land::TensorLand, land_sources::SourceTrial, transport::TensorTransport,
};
use crate::contract::EngineError;
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
    /// Full-grid coordinator exchange depths. Standalone steps hold none.
    pub exchange: Option<ExchangeDepths<R>>,
}

/// Full-grid exchange volumes over cell area, zero outside the linked cells.
#[derive(Clone)]
pub(super) struct ExchangeDepths<R: Runtime<DType = DType>> {
    pub inflow: Tensor<R>,
    pub withdrawal: Tensor<R>,
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
        let exchange = receipt
            .exchange
            .as_ref()
            .map(|depths| [&depths.inflow, &depths.withdrawal]);
        step.sources = receipt
            .sources
            .receipts(land, &receipt.start_depth, exchange)?;
        Ok(step)
    }
}

impl<R: Runtime<DType = DType>> StepReceipt<R> {
    /// Returns the accepted step's start time in seconds.
    pub(super) fn start(&self) -> f64 {
        match self {
            Self::Device(receipt) => receipt.scalar.start,
            Self::Imported(step) => step.start,
        }
    }
}

/// Moves a local forcing index between engines whose windows follow different prefixes.
///
/// # Errors
/// Rejects an index outside the destination window of `len` intervals.
pub(super) fn rebase_forcing_index(
    index: usize,
    from_prefix: usize,
    to_prefix: usize,
    len: usize,
) -> Result<usize, EngineError> {
    index
        .checked_add(from_prefix)
        .and_then(|absolute| absolute.checked_sub(to_prefix))
        .filter(|local| *local < len)
        .ok_or_else(|| {
            EngineError::rejected(
                "step receipt",
                format!(
                    "forcing_index={index}, source prefix={from_prefix}, destination prefix={to_prefix}, destination intervals={len}: restore into an engine whose forcing window holds the receipt interval"
                ),
            )
        })
}

#[cfg(test)]
mod tests {
    use super::rebase_forcing_index;

    #[test]
    fn rebase_moves_index_by_prefix_difference() {
        assert_eq!(rebase_forcing_index(2, 0, 0, 5).ok(), Some(2));
        assert_eq!(rebase_forcing_index(0, 39, 0, 50).ok(), Some(39));
        assert_eq!(rebase_forcing_index(3, 10, 12, 5).ok(), Some(1));
        assert!(rebase_forcing_index(1, 10, 12, 5).is_err());
        assert!(rebase_forcing_index(4, 39, 0, 43).is_err());
        assert!(rebase_forcing_index(usize::MAX, 1, 0, 5).is_err());
    }
}
