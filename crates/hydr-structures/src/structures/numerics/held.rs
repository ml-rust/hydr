//! Original signed Q-times-duration proposals without donor or storage admission.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ScalarOps;
use numr::ops::UnaryOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::structures::tensor::TensorStructureContext;
use numr::tensor::Tensor;

pub(crate) fn volumes<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    discharge: &Tensor<R>,
    dt: f64,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    let volume = c.mul_scalar(discharge, dt)?;
    context.finite(&volume, "held-head signed Q*dt proposal overflow")?;
    context.reject(
        &c.mul(
            &c.gt(&c.abs(discharge)?, &context.zero)?,
            &c.eq(&volume, &context.zero)?,
        )?,
        "active held-head volume underflows to zero",
    )?;
    Ok(volume)
}
