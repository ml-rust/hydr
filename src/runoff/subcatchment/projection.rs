//! Lazy complete interval projection without numerical continuation through host arrays.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::optimize::roots::BatchedNewtonAlgorithms;

use super::accounting::aggregate_step;
use super::batch::StoredStep;
use super::projection_hydraulic::read;
use super::projection_phase;
use super::{
    SubcatchmentError, SubcatchmentInput, SubcatchmentReceipt, SubcatchmentStepDiagnostics,
    SubcatchmentTransfer,
};
use crate::tensor::subcatchment::TensorSubcatchmentContext;

pub(super) fn project_step<R>(
    context: &TensorSubcatchmentContext<R>,
    stored: &StoredStep<R>,
) -> Result<SubcatchmentStepDiagnostics, SubcatchmentError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    let step = &stored.step;
    let dt = stored.end - stored.start;
    crate::numerics::subcatchment::check_step(context, step, dt)?;
    let totals = aggregate_step(context, step)?;
    let n = context.definitions.len();
    let flat = n * 3;
    let rain = read(&step.input.rain, flat)?;
    let evaporation = read(&step.input.evaporation, flat)?;
    let external = read(&step.input.external, flat)?;
    let residual = read(&step.totals.catchment_water_residual, n)?;
    let scale = read(&step.totals.catchment_water_scale, n)?;
    let rain_difference = read(&step.totals.catchment_rain_area_difference, n)?;
    let rain_bound = read(&step.totals.catchment_rain_area_bound, n)?;
    let mut receipts: Vec<_> = (0..n)
        .map(|i| SubcatchmentReceipt {
            catchment: i,
            input: SubcatchmentInput {
                rain_rate_m_per_s: rain[i * 3],
                potential_evaporation_m_per_s: evaporation[i * 3],
                external_runon_volumes_m3: [
                    external[i * 3],
                    external[i * 3 + 1],
                    external[i * 3 + 2],
                ],
            },
            subareas: std::array::from_fn(|_| None),
            water_residual_m3: residual[i],
            water_scale_m3: scale[i],
            rain_area_difference_m3: rain_difference[i],
            rain_area_bound_m3: rain_bound[i],
        })
        .collect();
    for (group, phase) in [
        (context.phase1.as_ref(), step.phase1.as_ref()),
        (context.phase2.as_ref(), step.phase2.as_ref()),
    ] {
        match (group, phase) {
            (None, None) => {}
            (Some(group), Some(phase)) => {
                for receipt in projection_phase::phase(group, phase)? {
                    let catchment = receipt.catchment;
                    let subarea = receipt.subarea.index();
                    let target = receipts.get_mut(catchment).ok_or_else(|| {
                        SubcatchmentError::numerical("projected phase exceeds catchment support")
                    })?;
                    target.subareas[subarea] = Some(receipt);
                }
            }
            _ => {
                return Err(SubcatchmentError::numerical(
                    "phase receipts differ from static role support",
                ));
            }
        }
    }
    let mut transfers = Vec::new();
    for receipt in &receipts {
        let parameters = &context.definitions[receipt.catchment].parameters;
        if let Some(receiver) = parameters.receiver() {
            for donor in receipt.subareas.iter().flatten() {
                if donor.partition.fraction > 0.0 {
                    transfers.push(SubcatchmentTransfer {
                        catchment: receipt.catchment,
                        donor: donor.subarea,
                        receiver,
                        volume_m3: donor.partition.routed_volume_m3,
                        fraction: donor.partition.fraction,
                        donor_volume_m3: donor.drainage.export_volume_m3,
                        outlet_volume_m3: donor.partition.outlet_volume_m3,
                        partition_residual_m3: donor.partition.residual_m3,
                        partition_bound_m3: donor.partition.bound_m3,
                    });
                }
            }
        }
    }
    Ok(SubcatchmentStepDiagnostics {
        start: stored.start,
        end: stored.end,
        dt,
        root_iterations: step.totals.root_iterations,
        storage_before_m3: totals.before,
        storage_after_m3: totals.after,
        applied_rain_volume_m3: totals.rain,
        external_runon_volume_m3: totals.external,
        infiltration_volume_m3: totals.infiltration,
        evaporation_volume_m3: totals.evaporation,
        outlet_volume_m3: totals.outlet,
        soil_water_adjustment_volume_m3: totals.adjustment,
        soil_water_adjustment_abs_charge_m3: step.totals.adjustment_charge,
        internal_transfer_volume_m3: totals.internal,
        transfer_uncertainty_charge_m3: step.totals.uncertainty_charge,
        physical_scale_lower_m3: step.totals.physical_scale_lower,
        water_residual_m3: totals.residual,
        water_scale_m3: totals.scale,
        rain_area_difference_m3: totals.rain_difference,
        rain_area_bound_m3: totals.rain_bound,
        phase1: step.phase1.as_ref().map(projection_phase::diagnostics),
        phase2: step.phase2.as_ref().map(projection_phase::diagnostics),
        receipts,
        transfers,
    })
}
