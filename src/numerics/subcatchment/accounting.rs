//! Independent full-support ledgers and catchment rainfall-area evidence.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::bounds;
use super::step::{TensorSubareaStep, TensorSubcatchmentTotals};
use crate::runoff::SubcatchmentError;
use crate::tensor::subcatchment::{
    TensorSubareaGroup, TensorSubcatchmentContext, TensorSubcatchmentInput,
};

pub(super) fn totals<R>(
    ctx: &TensorSubcatchmentContext<R>,
    input: &TensorSubcatchmentInput<R>,
    phases: [Option<(&TensorSubareaGroup<R>, &TensorSubareaStep<R>)>; 2],
    dt: f64,
) -> Result<TensorSubcatchmentTotals<R>, SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let c = &ctx.client;
    let mut tensors: [Tensor<R>; 12] = std::array::from_fn(|_| ctx.zero.clone());
    let mut adjustment_charge = 0.0;
    let mut uncertainty_charge = 0.0;
    let mut physical_scale_lower = 0.0;
    let mut root_iterations = 0usize;
    for (group, step) in phases.into_iter().flatten() {
        let fields = [
            &step.storage_before,
            &step.drainage.storage_after,
            &step.rain_volume,
            &step.external_volume,
            &step.infiltration_volume,
            &step.evaporation_volume,
            &step.partition.outlet,
            &step.adjustment_volume,
            &step.internal_received_volume,
            &step.water_residual,
            &step.water_scale,
            &step.rain_area_bound,
        ];
        for (target, value) in tensors.iter_mut().zip(fields) {
            *target = c.scatter(target, 0, &group.indices, value)?;
        }
        adjustment_charge = bounds::add_charge(adjustment_charge, step.adjustment_charge)?;
        uncertainty_charge = bounds::add_charge(uncertainty_charge, step.uncertainty_charge)?;
        physical_scale_lower =
            bounds::add_scale_lower(physical_scale_lower, step.physical_scale_lower)?;
        root_iterations = root_iterations
            .checked_add(step.drainage.iterations)
            .ok_or_else(|| {
                SubcatchmentError::numerical("combined phase root work overflows usize")
            })?;
    }
    let [
        storage_before,
        storage_after,
        applied_rain,
        external_runon,
        infiltration,
        evaporation,
        outlet,
        soil_adjustment,
        internal_received,
        water_residual,
        water_scale,
        rain_area_bound,
    ] = tensors;
    let before = sum_classes(ctx, &storage_before)?;
    let after = sum_classes(ctx, &storage_after)?;
    let rain = sum_classes(ctx, &applied_rain)?;
    let external = sum_classes(ctx, &external_runon)?;
    let adjustment = sum_classes(ctx, &soil_adjustment)?;
    let losses = c.add(
        &c.add(
            &sum_classes(ctx, &infiltration)?,
            &sum_classes(ctx, &evaporation)?,
        )?,
        &sum_classes(ctx, &outlet)?,
    )?;
    let inflow = c.add(&c.add(&rain, &external)?, &adjustment)?;
    ctx.finite(&losses, "catchment losses and outlet volume")?;
    ctx.finite(&inflow, "catchment external credits")?;
    let catchment_water_residual = c.sub(&c.sub(&after, &before)?, &c.sub(&inflow, &losses)?)?;
    let catchment_water_scale =
        c.maximum(&c.maximum(&before, &after)?, &c.maximum(&inflow, &losses)?)?;
    check_residual(
        ctx,
        &catchment_water_residual,
        &catchment_water_scale,
        "whole-catchment water residual",
    )?;
    let (catchment_rain_area_difference, catchment_rain_area_bound) =
        rain_area(ctx, input, &applied_rain, &rain_area_bound, dt)?;
    Ok(TensorSubcatchmentTotals {
        storage_before,
        storage_after,
        applied_rain,
        external_runon,
        infiltration,
        evaporation,
        outlet,
        soil_adjustment,
        internal_received,
        water_residual,
        water_scale,
        rain_area_bound,
        catchment_water_residual,
        catchment_water_scale,
        catchment_rain_area_difference,
        catchment_rain_area_bound,
        adjustment_charge,
        uncertainty_charge,
        physical_scale_lower,
        root_iterations,
    })
}

pub(super) fn sum_classes<R>(
    ctx: &TensorSubcatchmentContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let matrix = value.reshape(&[ctx.definitions.len(), 3])?;
    let parts = [
        matrix.narrow(1, 0, 1)?.squeeze(Some(1)),
        matrix.narrow(1, 1, 1)?.squeeze(Some(1)),
        matrix.narrow(1, 2, 1)?.squeeze(Some(1)),
    ];
    let sum = ctx
        .client
        .add(&ctx.client.add(&parts[0], &parts[1])?, &parts[2])?;
    ctx.finite(&sum, "catchment class sum")?;
    Ok(sum)
}

fn rain_area<R>(
    ctx: &TensorSubcatchmentContext<R>,
    input: &TensorSubcatchmentInput<R>,
    applied: &Tensor<R>,
    bound: &Tensor<R>,
    dt: f64,
) -> Result<(Tensor<R>, Tensor<R>), SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let c = &ctx.client;
    let shape = [ctx.definitions.len(), 3];
    let values = applied.reshape(&shape)?;
    let bounds_matrix = bound.reshape(&shape)?;
    let value = |index| -> Result<Tensor<R>, SubcatchmentError> {
        Ok(values.narrow(1, index, 1)?.squeeze(Some(1)))
    };
    let area_bound = |index| -> Result<Tensor<R>, SubcatchmentError> {
        Ok(bounds_matrix.narrow(1, index, 1)?.squeeze(Some(1)))
    };
    let first_sum = c.add(&value(0)?, &value(1)?)?;
    let applied_sum = c.add(&first_sum, &value(2)?)?;
    let rain_depth = c.mul_scalar(&c.gather(&input.rain, 0, &ctx.catchment_slots)?, dt)?;
    let reference = c.mul(&ctx.total_area, &rain_depth)?;
    ctx.finite(&reference, "total-area rainfall diagnostic")?;
    let difference = c.sub(&applied_sum, &reference)?;
    ctx.finite(&difference, "rain-area discrepancy")?;
    let mut envelope = upper_add(ctx, &area_bound(0)?, &area_bound(1)?)?;
    for term in [
        area_bound(2)?,
        rounding(ctx, &first_sum)?,
        rounding(ctx, &applied_sum)?,
        rounding(ctx, &reference)?,
        rounding(ctx, &difference)?,
    ] {
        envelope = upper_add(ctx, &envelope, &term)?;
    }
    // Zero rainfall carries no area-discrepancy charge or fabricated arithmetic correction.
    let positive = c.gt(&rain_depth, &ctx.catchment_zero)?;
    let envelope =
        numr::ops::ConditionalOps::where_cond(c, &positive, &envelope, &ctx.catchment_zero)?;
    ctx.reject_mask(
        &c.gt(&c.abs(&difference)?, &envelope)?,
        "rain-area arithmetic envelope",
        &[("difference", &difference), ("bound", &envelope)],
        &[],
        "Rainfall-area discrepancy exceeds its independent arithmetic proof",
    )?;
    Ok((difference, envelope))
}

fn rounding<R>(
    ctx: &TensorSubcatchmentContext<R>,
    value: &Tensor<R>,
) -> Result<Tensor<R>, SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let result = bounds::rounding_with_client(&ctx.client, value)?;
    ctx.finite(&result, "catchment arithmetic rounding bound")?;
    Ok(result)
}

fn upper_add<R>(
    ctx: &TensorSubcatchmentContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>, SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let sum = ctx.client.add(a, b)?;
    ctx.finite(&sum, "catchment envelope sum")?;
    let result = ctx.client.add(&sum, &rounding(ctx, &sum)?)?;
    ctx.finite(&result, "catchment outward envelope")?;
    Ok(result)
}

pub(super) fn check_residual<R>(
    ctx: &TensorSubcatchmentContext<R>,
    residual: &Tensor<R>,
    scale: &Tensor<R>,
    quantity: &str,
) -> Result<(), SubcatchmentError>
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
        + TwoSumOps<R>,
{
    let c = &ctx.client;
    ctx.finite(residual, quantity)?;
    ctx.finite(scale, "catchment physical scale")?;
    let tolerance = c.add_scalar(
        &c.mul_scalar(scale, ctx.config.reservoir.volume_relative)?,
        ctx.config.reservoir.volume_absolute,
    )?;
    ctx.finite(&tolerance, "catchment volume tolerance")?;
    ctx.reject_mask(&c.gt(&c.abs(residual)?, &tolerance)?, quantity,
        &[("residual", residual), ("scale", scale), ("tolerance", &tolerance)], &[],
        "Physical conservation rejects the complete interval. Shorten the interval or rescale inputs")
}
