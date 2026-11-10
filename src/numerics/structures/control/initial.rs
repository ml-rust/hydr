//! Numerical initial qualification consistency without historical replay.

use crate::structures::{StructureError, StructureInputErrorKind};
use crate::tensor::structures::control::{TensorControlContext, TensorControlState};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps};
use numr::runtime::Runtime;

pub(crate) fn check_initial<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    state: &TensorControlState<R>,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.structures.client;
    let qualifies = super::relay::qualifies(context, &state.high, &state.sample.stage)?;
    context.reject(&c.mul(&state.sample.valid,&c.ne(&state.pending,&qualifies)?)?,Some(StructureInputErrorKind::Definition),"initial antecedent qualification disagrees with pending timer: retain required timer or supply fresh memory")?;
    let delay = c.where_cond(&state.high, &context.low_delay, &context.high_delay)?;
    let timer = super::timers::evaluate(
        context,
        &state.pending,
        &state.since,
        &state.sample.observed,
        &delay,
        false,
    )?;
    context.reject(&timer.ready,Some(StructureInputErrorKind::Definition),"initial qualification timer is already ready at its antecedent sample: supply completed transition state")
}
