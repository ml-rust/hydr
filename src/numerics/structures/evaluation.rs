//! Complete supported proposal composition without physical source aggregation.
use numr::ops::BinaryOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use crate::structures::StructureError;
use crate::tensor::structures::{
    TensorStructureContext, TensorStructureInput, TensorStructureResult,
};

pub(crate) fn evaluate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    input: &TensorStructureInput<R>,
) -> Result<TensorStructureResult<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    super::admission::check(context, input)?;
    let class = super::direction::classify(context, input)?;
    let orifice = super::orifice::evaluate(context, &class)?;
    let free = super::weir::free(context, &class)?;
    let submerged = super::weir::submerged(context, &class)?;
    let c = &context.client;
    let magnitude = c.where_cond(
        &class.orifice,
        &orifice.magnitude,
        &c.where_cond(&class.free_weir, &free.magnitude, &submerged.magnitude)?,
    )?;
    let discharge = c.where_cond(
        &class.active,
        &c.mul(&class.direction, &magnitude)?,
        &context.zero,
    )?;
    context.finite(&discharge, "final signed structure proposal")?;
    Ok(TensorStructureResult {
        gate: None,
        pump: None,
        discharge,
        regime: class.regime,
        status: class.status,
        direction: class.direction,
        head_gap: c.where_cond(&class.orifice, &orifice.gap, &submerged.gap)?,
        upstream_head: c.where_cond(&class.free_weir, &free.upstream, &submerged.upstream)?,
        downstream_head: submerged.downstream,
        ratio: submerged.ratio,
        correction_k: submerged.k,
        log_magnitude: c.where_cond(
            &class.orifice,
            &orifice.logarithm,
            &c.where_cond(&class.free_weir, &free.logarithm, &submerged.logarithm)?,
        )?,
    })
}
