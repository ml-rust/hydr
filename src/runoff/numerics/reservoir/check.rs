//! Primitive source-ledger checks composed with shared hydraulic admission.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::algebra::{self, Prepared};
use super::drainage::{self, DrainageResult, DrainageRoot};
use super::receipt::TensorReservoirStep;
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

pub(crate) fn check_step<R>(
    context: &TensorReservoirContext<R>,
    step: &TensorReservoirStep<R>,
    dt: f64,
) -> Result<(), ReservoirError>
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
    for (name, value) in fields(step) {
        drainage::check_metadata(context, value, name)?;
        context.finite(value, name)?;
    }
    let prepared = algebra::prepare(context, &step.old_depth, &step.incoming_rate, dt)?;
    let drained = drainage::admit_drainage(
        &prepared.hydraulic,
        DrainageRoot {
            lower: step.root_lower.clone(),
            upper: step.root_upper.clone(),
            depth: step.depth.clone(),
            iterations: step.root_iterations,
        },
    )?;
    let expected = finish(
        context,
        prepared,
        &step.old_depth,
        &step.incoming_rate,
        drained,
    )?;
    for ((name, value), (_, expected)) in fields(step).into_iter().zip(fields(&expected)) {
        drainage::equal(context, value, expected, name)?;
    }
    Ok(())
}

pub(super) fn finish<R>(
    context: &TensorReservoirContext<R>,
    prepared: Prepared<'_, R>,
    old: &Tensor<R>,
    incoming: &Tensor<R>,
    drained: DrainageResult<R>,
) -> Result<TensorReservoirStep<R>, ReservoirError>
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
    let DrainageResult {
        lower,
        upper,
        depth,
        export_depth,
        storage_after,
        export_volume,
        endpoint_rate,
        endpoint_discharge,
        law_depth,
        law_volume,
        constitutive_residual,
        constitutive_scale,
        iterations,
    } = drained;
    let c = &context.client;
    let water_residual = c.sub(
        &c.sub(&storage_after, &prepared.storage_before)?,
        &c.sub(&prepared.incoming_volume, &export_volume)?,
    )?;
    let water_scale = c.maximum(
        &c.maximum(&prepared.storage_before, &storage_after)?,
        &c.maximum(&prepared.incoming_volume, &export_volume)?,
    )?;
    drainage::check_residual(
        context,
        &water_residual,
        &water_scale,
        &context.one,
        "interval water residual",
    )?;
    Ok(TensorReservoirStep {
        old_depth: old.clone(),
        incoming_rate: incoming.clone(),
        incoming_depth: prepared.incoming_depth,
        available_depth: prepared.hydraulic.available_depth().clone(),
        active_duration: prepared.hydraulic.active_duration().clone(),
        root_lower: lower,
        root_upper: upper,
        depth,
        export_depth,
        storage_before: prepared.storage_before,
        storage_after,
        incoming_volume: prepared.incoming_volume,
        export_volume,
        endpoint_rate,
        endpoint_discharge,
        law_depth,
        law_volume,
        constitutive_residual,
        constitutive_scale,
        water_residual,
        water_scale,
        root_iterations: iterations,
    })
}

fn fields<R>(step: &TensorReservoirStep<R>) -> [(&str, &Tensor<R>); 21]
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
    [
        ("old depth", &step.old_depth),
        ("incoming rate", &step.incoming_rate),
        ("incoming depth", &step.incoming_depth),
        ("available depth", &step.available_depth),
        ("active duration", &step.active_duration),
        ("root lower", &step.root_lower),
        ("root upper", &step.root_upper),
        ("accepted depth", &step.depth),
        ("export depth", &step.export_depth),
        ("storage before", &step.storage_before),
        ("storage after", &step.storage_after),
        ("incoming volume", &step.incoming_volume),
        ("export volume", &step.export_volume),
        ("endpoint rate", &step.endpoint_rate),
        ("endpoint discharge", &step.endpoint_discharge),
        ("law depth", &step.law_depth),
        ("law volume", &step.law_volume),
        ("constitutive residual", &step.constitutive_residual),
        ("constitutive scale", &step.constitutive_scale),
        ("water residual", &step.water_residual),
        ("water scale", &step.water_scale),
    ]
}
