//! Persistent authoritative volume and element-level lifetime counters.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::TensorPoolContext;
use crate::storage::{LevelPoolError, LevelPoolState};

#[derive(Clone, Debug)]
pub(crate) struct TensorPoolState<R: Runtime<DType = DType>> {
    pub sources: Option<super::TensorPoolSourceState<R>>,
    pub volume_m3: Tensor<R>,
    pub initial_volume_m3: Tensor<R>,
    pub cumulative_incoming_volume_m3: Tensor<R>,
    pub cumulative_outgoing_volume_m3: Tensor<R>,
    pub incoming_roundoff_bound_m3: Tensor<R>,
    pub outgoing_roundoff_bound_m3: Tensor<R>,
}
impl<R> TensorPoolState<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    pub(crate) fn new(
        context: &TensorPoolContext<R>,
        states: &[LevelPoolState],
    ) -> Result<Self, LevelPoolError> {
        if states.len() != context.elements.len() {
            return Err(LevelPoolError::input(format!(
                "state count={},element count={}: supply explicit state per pool",
                states.len(),
                context.elements.len()
            )));
        }
        for (index, (state, element)) in states.iter().zip(&context.elements).enumerate() {
            if !state.volume_m3.is_finite()
                || state.volume_m3 < 0.0
                || state.volume_m3 > element.curve.maximum_volume_m3()
            {
                return Err(LevelPoolError::input(format!(
                    "volume={},maximum={}: supply explicit finite closed-range storage",
                    state.volume_m3,
                    element.curve.maximum_volume_m3()
                ))
                .at(index));
            }
        }
        let volume = context.tensor(&states.iter().map(|s| s.volume_m3).collect::<Vec<_>>())?;
        Ok(Self {
            sources: context
                .source_policies
                .as_ref()
                .map(|_| super::TensorPoolSourceState::<R>::zero(context)),
            initial_volume_m3: volume.clone(),
            volume_m3: volume,
            cumulative_incoming_volume_m3: context.zero.clone(),
            cumulative_outgoing_volume_m3: context.zero.clone(),
            incoming_roundoff_bound_m3: context.zero.clone(),
            outgoing_roundoff_bound_m3: context.zero.clone(),
        })
    }
}
