//! Tensor-only bounded last-transition selection.

use crate::structures::StructureError;
use crate::structures::tensor::control::{
    TensorControlContext, TensorControlSample, TensorControlTimer,
};
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn sample<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    mask: &Tensor<R>,
    new: &TensorControlSample<R>,
    old: &TensorControlSample<R>,
) -> Result<TensorControlSample<R>, StructureError>
where
    R::Client: numr::ops::ConditionalOps<R>,
{
    let c = &context.structures.client;
    Ok(TensorControlSample {
        valid: c.where_cond(mask, &new.valid, &old.valid)?,
        observed: c.where_cond(mask, &new.observed, &old.observed)?,
        available: c.where_cond(mask, &new.available, &old.available)?,
        stage: c.where_cond(mask, &new.stage, &old.stage)?,
    })
}
pub(super) fn timer<R: Runtime<DType = DType>>(
    context: &TensorControlContext<R>,
    mask: &Tensor<R>,
    new: &TensorControlTimer<R>,
    old: &TensorControlTimer<R>,
) -> Result<TensorControlTimer<R>, StructureError>
where
    R::Client: numr::ops::ConditionalOps<R>,
{
    let c = &context.structures.client;
    macro_rules! choose {
        ($field:ident) => {
            c.where_cond(mask, &new.$field, &old.$field)?
        };
    }
    Ok(TensorControlTimer {
        active: choose!(active),
        started: choose!(started),
        evaluated: choose!(evaluated),
        delay: choose!(delay),
        elapsed_high: choose!(elapsed_high),
        elapsed_low: choose!(elapsed_low),
        ready: choose!(ready),
        allowed_high: choose!(allowed_high),
        allowed_low: choose!(allowed_low),
        overflow: choose!(overflow),
        within: choose!(within),
    })
}
