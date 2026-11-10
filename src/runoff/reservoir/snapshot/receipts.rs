//! Reconstruct checked receipt tensors without replaying reservoir integration.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::restore::require;
use crate::numerics::reservoir::{TensorReservoirStep, check_step};
use crate::runoff::{ReservoirError, ReservoirReceipt, ReservoirStepDiagnostics};
use crate::tensor::reservoir_storage::TensorReservoirContext;

pub(super) fn import<R>(
    context: &TensorReservoirContext<R>,
    diagnostics: &ReservoirStepDiagnostics,
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
    require(
        diagnostics.receipts.len() == context.elements.len(),
        format!(
            "receipt count={}, elements={}: require complete ordered support",
            diagnostics.receipts.len(),
            context.elements.len()
        ),
    )?;
    for (index, receipt) in diagnostics.receipts.iter().enumerate() {
        require(
            receipt.element == index,
            format!(
                "receipt element={}, expected={index}: retain element order",
                receipt.element
            ),
        )
        .map_err(|error| error.at_element(index))?;
        let instantaneous = context.elements[index].parameters.is_instantaneous();
        require(
            receipt.endpoint_rate_m_per_s.is_none() == instantaneous
                && receipt.endpoint_discharge_m3_per_s.is_none() == instantaneous,
            "endpoint rate/discharge presence differs from the declared reservoir branch",
        )
        .map_err(|error| error.at_element(index))?;
    }
    let tensor = |select| from_receipts(context, &diagnostics.receipts, select);
    let step = TensorReservoirStep {
        old_depth: tensor(|r| r.old_depth_m)?,
        incoming_rate: tensor(|r| r.incoming_rate_m_per_s)?,
        incoming_depth: tensor(|r| r.incoming_depth_m)?,
        available_depth: tensor(|r| r.available_depth_m)?,
        active_duration: tensor(|r| r.active_duration_s)?,
        root_lower: tensor(|r| r.root_lower_m)?,
        root_upper: tensor(|r| r.root_upper_m)?,
        depth: tensor(|r| r.depth_m)?,
        export_depth: tensor(|r| r.export_depth_m)?,
        storage_before: tensor(|r| r.storage_before_m3)?,
        storage_after: tensor(|r| r.storage_after_m3)?,
        incoming_volume: tensor(|r| r.incoming_volume_m3)?,
        export_volume: tensor(|r| r.export_volume_m3)?,
        endpoint_rate: tensor(|r| r.endpoint_rate_m_per_s.unwrap_or(0.0))?,
        endpoint_discharge: tensor(|r| r.endpoint_discharge_m3_per_s.unwrap_or(0.0))?,
        law_depth: tensor(|r| r.law_depth_m)?,
        law_volume: tensor(|r| r.law_volume_m3)?,
        constitutive_residual: tensor(|r| r.constitutive_residual_m3)?,
        constitutive_scale: tensor(|r| r.constitutive_scale_m3)?,
        water_residual: tensor(|r| r.water_residual_m3)?,
        water_scale: tensor(|r| r.water_scale_m3)?,
        root_iterations: diagnostics.root_iterations,
    };
    check_step(context, &step, diagnostics.dt)?;
    Ok(step)
}

fn from_receipts<R>(
    context: &TensorReservoirContext<R>,
    receipts: &[ReservoirReceipt],
    select: fn(&ReservoirReceipt) -> f64,
) -> Result<Tensor<R>, ReservoirError>
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
    let values: Vec<_> = receipts.iter().map(select).collect();
    context.tensor(&values)
}
