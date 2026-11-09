//! Reusable interval and imported-receipt admission without root replay.

use super::algebra::{self, Prepared};
use super::receipt::CpuReservoirStep;
use crate::cpu::CpuTensor;
use crate::cpu::reservoir_storage::CpuReservoirContext;
use crate::runoff::ReservoirError;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, UnaryOps};

pub(crate) fn check_step(
    context: &CpuReservoirContext,
    step: &CpuReservoirStep,
    dt: f64,
) -> Result<(), ReservoirError> {
    for (name, value) in fields(step) {
        algebra::check_metadata(context, value, name)?;
        context.finite(value, name)?;
    }
    let prepared = algebra::prepare(context, &step.old_depth, &step.incoming_rate, dt)?;
    let expected = finish(
        context,
        prepared,
        &step.old_depth,
        &step.incoming_rate,
        RootResult {
            lower: step.root_lower.clone(),
            upper: step.root_upper.clone(),
            depth: step.depth.clone(),
            iterations: step.root_iterations,
        },
    )?;
    for ((name, value), (_, expected)) in fields(step).into_iter().zip(fields(&expected)) {
        equal(context, value, expected, name)?;
    }
    Ok(())
}

pub(super) struct RootResult {
    pub lower: CpuTensor,
    pub upper: CpuTensor,
    pub depth: CpuTensor,
    pub iterations: usize,
}

pub(super) fn finish(
    context: &CpuReservoirContext,
    prepared: Prepared,
    old: &CpuTensor,
    incoming: &CpuTensor,
    root: RootResult,
) -> Result<CpuReservoirStep, ReservoirError> {
    let RootResult {
        lower,
        upper,
        depth,
        iterations,
    } = root;
    let c = &context.client;
    check_roots(context, &prepared, &lower, &upper, iterations)?;
    context.finite(&depth, "accepted reservoir depth")?;
    let outside = c.maximum(
        &c.lt(&depth, &context.zero)?,
        &c.gt(&depth, &prepared.available)?,
    )?;
    context.reject_mask(
        &outside,
        "accepted depth",
        &[("depth", &depth), ("available", &prepared.available)],
        &[],
        "Require depth within available nonnegative water",
    )?;
    equal(
        context,
        &depth,
        &algebra::candidate_depth(context, &prepared, &upper)?,
        "accepted depth reconstruction",
    )?;
    context.reject_mask(&c.mul(&prepared.active, &c.le(&depth, &context.depression)?)?, "represented implicit excess", &[("depth", &depth), ("depression", &context.depression), ("root upper", &upper)], &[], "Positive implicit excess collapses during reconstruction. Rescale storage or change the interval")?;
    let accepted_excess = c.where_cond(
        &prepared.active,
        &c.sub(&depth, &context.depression)?,
        &context.zero,
    )?;
    context.reject_mask(
        &c.mul(&prepared.active, &c.le(&accepted_excess, &context.zero)?)?,
        "accepted excess",
        &[("excess", &accepted_excess)],
        &[],
        "Require positive represented excess for active finite-roughness drainage",
    )?;
    let export_depth = c.sub(&prepared.available, &depth)?;
    context.finite(&export_depth, "export depth")?;
    context.reject_mask(
        &c.mul(&prepared.active, &c.eq(&export_depth, &context.zero)?)?,
        "represented export",
        &[("available", &prepared.available), ("depth", &depth)],
        &[],
        "Positive drainage becomes unrepresentable. Rescale storage or change the interval",
    )?;
    let endpoint_rate = algebra::law(context, &accepted_excess, &prepared.active)?;
    let endpoint_discharge = algebra::volume(context, &endpoint_rate, "endpoint discharge")?;
    let storage_after = algebra::volume(context, &depth, "accepted storage")?;
    let export_volume = algebra::volume(context, &export_depth, "exported volume")?;
    let law_depth = c.mul(&prepared.duration, &endpoint_rate)?;
    context.finite(&law_depth, "constitutive interval depth")?;
    context.reject_mask(
        &c.mul(
            &c.gt(&endpoint_rate, &context.zero)?,
            &c.eq(&law_depth, &context.zero)?,
        )?,
        "constitutive interval depth",
        &[("rate", &endpoint_rate), ("duration", &prepared.duration)],
        &[],
        "Positive constitutive depth underflows. Rescale the reservoir or interval",
    )?;
    let law_volume = algebra::volume(context, &law_depth, "constitutive interval volume")?;
    let finite_roughness = c.sub(&context.one, &context.instantaneous)?;
    let constitutive_residual = c.where_cond(
        &finite_roughness,
        &c.sub(&export_volume, &law_volume)?,
        &context.zero,
    )?;
    let constitutive_scale = c.where_cond(
        &finite_roughness,
        &c.maximum(&c.abs(&export_volume)?, &c.abs(&law_volume)?)?,
        &context.zero,
    )?;
    check_residual(
        context,
        &constitutive_residual,
        &constitutive_scale,
        &finite_roughness,
        "constitutive volume residual",
    )?;
    let water_residual = c.sub(
        &c.sub(&storage_after, &prepared.storage_before)?,
        &c.sub(&prepared.incoming_volume, &export_volume)?,
    )?;
    let water_scale = c.maximum(
        &c.maximum(&prepared.storage_before, &storage_after)?,
        &c.maximum(&prepared.incoming_volume, &export_volume)?,
    )?;
    check_residual(
        context,
        &water_residual,
        &water_scale,
        &context.one,
        "interval water residual",
    )?;
    Ok(CpuReservoirStep {
        old_depth: old.clone(),
        incoming_rate: incoming.clone(),
        incoming_depth: prepared.incoming_depth,
        available_depth: prepared.available,
        active_duration: prepared.duration,
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

pub(super) fn check_roots(
    context: &CpuReservoirContext,
    prepared: &Prepared,
    lower: &CpuTensor,
    upper: &CpuTensor,
    iterations: usize,
) -> Result<(), ReservoirError> {
    let c = &context.client;
    if iterations > context.config.root_iterations
        || (iterations != 0 && !context.any(&prepared.active)?)
    {
        return Err(ReservoirError::numerical(format!(
            "root iterations={iterations}: require work within {} rounds and no work for inactive drainage",
            context.config.root_iterations
        )));
    }
    for (name, bound) in [("root lower", lower), ("root upper", upper)] {
        context.finite(bound, name)?;
    }
    let outside = c.maximum(
        &c.lt(lower, &context.zero)?,
        &c.maximum(&c.gt(lower, upper)?, &c.gt(upper, &prepared.excess)?)?,
    )?;
    context.reject_mask(
        &outside,
        "root bracket",
        &[
            ("lower", lower),
            ("upper", upper),
            ("available excess", &prepared.excess),
        ],
        &[],
        "Require an ordered bracket inside available excess, with zero inactive bounds",
    )?;
    let width = c.sub(upper, lower)?;
    let scale = c.maximum(&c.abs(lower)?, &c.abs(upper)?)?;
    let tolerance = c.add_scalar(
        &c.mul_scalar(&scale, context.config.root_relative_depth)?,
        context.config.root_absolute_depth,
    )?;
    context.finite(&tolerance, "root width tolerance")?;
    context.reject_mask(&c.mul(&prepared.active, &c.maximum(&c.le(&tolerance, &context.zero)?, &c.gt(&width, &tolerance)?)?)?, "root width", &[("width", &width), ("tolerance", &tolerance)], &[], "Bracket exceeds representable root tolerance. Increase root work or select an admitted tolerance")?;
    let lower_rate = algebra::law(context, lower, &prepared.active)?;
    let upper_rate = algebra::law(context, upper, &prepared.active)?;
    let lower_residual = c.sub(
        &c.add(lower, &c.mul(&prepared.duration, &lower_rate)?)?,
        &prepared.excess,
    )?;
    let upper_residual = c.sub(
        &c.add(upper, &c.mul(&prepared.duration, &upper_rate)?)?,
        &prepared.excess,
    )?;
    context.finite(&lower_residual, "root lower residual")?;
    context.finite(&upper_residual, "root upper residual")?;
    context.reject_mask(
        &c.maximum(
            &c.gt(&lower_residual, &context.zero)?,
            &c.lt(&upper_residual, &context.zero)?,
        )?,
        "root signs",
        &[
            ("lower residual", &lower_residual),
            ("upper residual", &upper_residual),
        ],
        &[],
        "Root bracket does not enclose the implicit law. Restore matching receipts",
    )?;
    Ok(())
}

fn check_residual(
    context: &CpuReservoirContext,
    residual: &CpuTensor,
    scale: &CpuTensor,
    active: &CpuTensor,
    quantity: &str,
) -> Result<(), ReservoirError> {
    let c = &context.client;
    context.finite(residual, quantity)?;
    context.finite(scale, "physical volume tolerance scale")?;
    let tolerance = c.add_scalar(
        &c.mul_scalar(scale, context.config.volume_relative)?,
        context.config.volume_absolute,
    )?;
    context.finite(&tolerance, "physical volume tolerance")?;
    context.reject_mask(&c.mul(active, &c.gt(&c.abs(residual)?, &tolerance)?)?, quantity, &[("residual", residual), ("scale", scale), ("tolerance", &tolerance)], &[], "Physical volume admission rejects the interval. Shorten the interval, tighten the root, or rescale the reservoir")
}

fn equal(
    context: &CpuReservoirContext,
    actual: &CpuTensor,
    expected: &CpuTensor,
    quantity: &str,
) -> Result<(), ReservoirError> {
    context.reject_mask(
        &context.client.ne(actual, expected)?,
        quantity,
        &[("actual", actual), ("expected", expected)],
        &[],
        "Receipt disagrees with represented reservoir algebra. Restore a matching complete receipt",
    )
}

fn fields(step: &CpuReservoirStep) -> [(&str, &CpuTensor); 21] {
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
