//! Explicit caller-boundary projections of accepted tensor receipts.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

use super::accounting::aggregate_step;
use super::batch::StoredStep;
use super::{ReservoirError, ReservoirReceipt, ReservoirStepDiagnostics};
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

/// Raw element output for one represented accepted interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ReservoirOutput {
    /// Ordered element index.
    pub element: usize,
    /// Accepted depth in metres.
    pub depth_m: f64,
    /// Accepted represented storage in m³.
    pub storage_m3: f64,
    /// Actual interval input in m³.
    pub incoming_volume_m3: f64,
    /// Conservative interval export in m³.
    pub export_volume_m3: f64,
    /// Endpoint discharge in m³/s. None denotes explicit instantaneous overflow.
    pub endpoint_discharge_m3_per_s: Option<f64>,
    /// Independent represented water residual in m³.
    pub water_residual_m3: f64,
    /// Finite-roughness constitutive residual in m³. Instantaneous overflow records zero.
    pub constitutive_residual_m3: f64,
}

pub(super) fn project_step<R>(
    context: &TensorReservoirContext<R>,
    stored: &StoredStep<R>,
) -> Result<ReservoirStepDiagnostics, ReservoirError>
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
    let step = &stored.step;
    let dt = stored.end - stored.start;
    crate::runoff::numerics::reservoir::check_step(context, step, dt)?;
    let totals = aggregate_step(context, step)?;
    let tensors = [
        &step.old_depth,
        &step.incoming_rate,
        &step.incoming_depth,
        &step.available_depth,
        &step.active_duration,
        &step.root_lower,
        &step.root_upper,
        &step.depth,
        &step.export_depth,
        &step.storage_before,
        &step.storage_after,
        &step.incoming_volume,
        &step.export_volume,
        &step.endpoint_rate,
        &step.endpoint_discharge,
        &step.law_depth,
        &step.law_volume,
        &step.constitutive_residual,
        &step.constitutive_scale,
        &step.water_residual,
        &step.water_scale,
    ];
    let fields = tensors
        .iter()
        .map(|t| t.try_to_vec::<f64>())
        .collect::<Result<Vec<_>, _>>()?;
    if fields.iter().any(|f| f.len() != context.elements.len()) {
        return Err(ReservoirError::numerical(
            "receipt projection support differs from ordered reservoir elements",
        ));
    }
    let receipts = context
        .elements
        .iter()
        .enumerate()
        .map(|(i, element)| {
            let finite = !element.parameters.is_instantaneous();
            ReservoirReceipt {
                element: i,
                old_depth_m: fields[0][i],
                incoming_rate_m_per_s: fields[1][i],
                incoming_depth_m: fields[2][i],
                available_depth_m: fields[3][i],
                active_duration_s: fields[4][i],
                root_lower_m: fields[5][i],
                root_upper_m: fields[6][i],
                depth_m: fields[7][i],
                export_depth_m: fields[8][i],
                storage_before_m3: fields[9][i],
                storage_after_m3: fields[10][i],
                incoming_volume_m3: fields[11][i],
                export_volume_m3: fields[12][i],
                endpoint_rate_m_per_s: finite.then_some(fields[13][i]),
                endpoint_discharge_m3_per_s: finite.then_some(fields[14][i]),
                law_depth_m: fields[15][i],
                law_volume_m3: fields[16][i],
                constitutive_residual_m3: fields[17][i],
                constitutive_scale_m3: fields[18][i],
                water_residual_m3: fields[19][i],
                water_scale_m3: fields[20][i],
            }
        })
        .collect();
    Ok(ReservoirStepDiagnostics {
        start: stored.start,
        end: stored.end,
        dt,
        root_iterations: step.root_iterations,
        storage_before_m3: totals.before,
        storage_after_m3: totals.after,
        incoming_volume_m3: totals.incoming,
        export_volume_m3: totals.export,
        constitutive_residual_m3: totals.constitutive,
        constitutive_scale_m3: totals.constitutive_scale,
        water_residual_m3: totals.water,
        water_scale_m3: totals.water_scale,
        receipts,
    })
}
