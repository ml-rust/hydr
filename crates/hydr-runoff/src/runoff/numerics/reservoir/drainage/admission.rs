//! Root, represented-state, and constitutive admission independent of source ledgers.

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use super::algebra;
use super::budget::DrainageBudget;
use super::receipt::{DrainageResult, DrainageRoot};
use crate::runoff::ReservoirError;
use crate::runoff::tensor::reservoir_storage::TensorReservoirContext;

pub(crate) fn admit_drainage<R>(
    prepared: &DrainageBudget<'_, R>,
    root: DrainageRoot<R>,
) -> Result<DrainageResult<R>, ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let context = prepared.context();
    for (name, value) in [
        ("root lower", &root.lower),
        ("root upper", &root.upper),
        ("accepted depth", &root.depth),
    ] {
        algebra::check_metadata(context, value, name)?;
    }
    let DrainageRoot {
        lower,
        upper,
        depth,
        iterations,
    } = root;
    let c = &context.client;
    check_roots(context, prepared, &lower, &upper, iterations)?;
    context.finite(&depth, "accepted reservoir depth")?;
    let outside = c.maximum(
        &c.lt(&depth, &context.zero)?,
        &c.gt(&depth, prepared.available_depth())?,
    )?;
    context.reject_mask(
        &outside,
        "accepted depth",
        &[("depth", &depth), ("available", prepared.available_depth())],
        &[],
        "Require depth within available nonnegative water",
    )?;
    equal(
        context,
        &depth,
        &algebra::candidate_depth(context, prepared, &upper)?,
        "accepted depth reconstruction",
    )?;
    context.reject_mask(&c.mul(prepared.active(), &c.le(&depth, &context.depression)?)?, "represented implicit excess", &[("depth", &depth), ("depression", &context.depression), ("root upper", &upper)], &[], "Positive implicit excess collapses during reconstruction. Rescale storage or change the interval")?;
    let accepted_excess = c.where_cond(
        prepared.active(),
        &c.sub(&depth, &context.depression)?,
        &context.zero,
    )?;
    context.reject_mask(
        &c.mul(prepared.active(), &c.le(&accepted_excess, &context.zero)?)?,
        "accepted excess",
        &[("excess", &accepted_excess)],
        &[],
        "Require positive represented excess for active finite-roughness drainage",
    )?;
    let export_depth = c.sub(prepared.available_depth(), &depth)?;
    context.finite(&export_depth, "export depth")?;
    context.reject_mask(
        &c.mul(prepared.active(), &c.eq(&export_depth, &context.zero)?)?,
        "represented export",
        &[("available", prepared.available_depth()), ("depth", &depth)],
        &[],
        "Positive drainage becomes unrepresentable. Rescale storage or change the interval",
    )?;
    let endpoint_rate = algebra::law(context, &accepted_excess, prepared.active())?;
    let endpoint_discharge = algebra::volume(context, &endpoint_rate, "endpoint discharge")?;
    let storage_after = algebra::volume(context, &depth, "accepted storage")?;
    let export_volume = algebra::volume(context, &export_depth, "exported volume")?;
    let law_depth = c.mul(prepared.active_duration(), &endpoint_rate)?;
    context.finite(&law_depth, "constitutive interval depth")?;
    context.reject_mask(
        &c.mul(
            &c.gt(&endpoint_rate, &context.zero)?,
            &c.eq(&law_depth, &context.zero)?,
        )?,
        "constitutive interval depth",
        &[
            ("rate", &endpoint_rate),
            ("duration", prepared.active_duration()),
        ],
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
    Ok(DrainageResult {
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
    })
}

fn check_roots<R>(
    context: &TensorReservoirContext<R>,
    prepared: &DrainageBudget<'_, R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    iterations: usize,
) -> Result<(), ReservoirError>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    let c = &context.client;
    if iterations > context.config.root_iterations
        || (iterations != 0 && !context.any(prepared.active())?)
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
        &c.maximum(&c.gt(lower, upper)?, &c.gt(upper, prepared.excess())?)?,
    )?;
    context.reject_mask(
        &outside,
        "root bracket",
        &[
            ("lower", lower),
            ("upper", upper),
            ("available excess", prepared.excess()),
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
    context.reject_mask(&c.mul(prepared.active(), &c.maximum(&c.le(&tolerance, &context.zero)?, &c.gt(&width, &tolerance)?)?)?, "root width", &[("width", &width), ("tolerance", &tolerance)], &[], "Bracket exceeds representable root tolerance. Increase root work or select an admitted tolerance")?;
    let lower_rate = algebra::law(context, lower, prepared.active())?;
    let upper_rate = algebra::law(context, upper, prepared.active())?;
    let lower_residual = c.sub(
        &c.add(lower, &c.mul(prepared.active_duration(), &lower_rate)?)?,
        prepared.excess(),
    )?;
    let upper_residual = c.sub(
        &c.add(upper, &c.mul(prepared.active_duration(), &upper_rate)?)?,
        prepared.excess(),
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

pub(crate) fn check_residual<R>(
    context: &TensorReservoirContext<R>,
    residual: &Tensor<R>,
    scale: &Tensor<R>,
    active: &Tensor<R>,
    quantity: &str,
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

pub(in crate::runoff::numerics::reservoir) fn equal<R>(
    context: &TensorReservoirContext<R>,
    actual: &Tensor<R>,
    expected: &Tensor<R>,
    quantity: &str,
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
    context.reject_mask(
        &context.client.ne(actual, expected)?,
        quantity,
        &[("actual", actual), ("expected", expected)],
        &[],
        "Receipt disagrees with represented reservoir algebra. Restore a matching complete receipt",
    )
}

#[cfg(test)]
mod tests {
    use super::super::super::fixture::{cpu_client, element};
    use super::super::{drain, prepare_budget};
    use super::*;
    use crate::runoff::ReservoirConfig;
    use numr::runtime::cpu::CpuRuntime;

    #[test]
    fn foreign_root_witness_uses_the_budget_construction_context() -> Result<(), ReservoirError> {
        let context_a = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![element("same-identity", 1.0 / 32.0, 32.0)?],
            ReservoirConfig::default(),
        )?;
        let context_b = TensorReservoirContext::<CpuRuntime>::new(
            cpu_client(),
            vec![element("same-identity", 1.0 / 32.0, 64.0)?],
            ReservoirConfig::default(),
        )?;
        let available = 0.25 * 0.3048;
        let budget_a = prepare_budget(&context_a, context_a.tensor(&[available])?, 1.0, |_| {
            context_a.tensor(&[1.0])
        })?;
        let budget_b = prepare_budget(&context_b, context_b.tensor(&[available])?, 1.0, |_| {
            context_b.tensor(&[1.0])
        })?;
        let from_a = drain(&budget_a)?;
        let error = admit_drainage(
            &budget_b,
            DrainageRoot {
                lower: from_a.lower,
                upper: from_a.upper,
                depth: from_a.depth,
                iterations: from_a.iterations,
            },
        )
        .err()
        .ok_or_else(|| {
            ReservoirError::numerical(
                "a physically different context must reject the foreign root witness",
            )
        })?;
        assert_eq!(error.element(), Some(0));
        assert!(error.to_string().contains("root signs"));
        let from_b = drain(&budget_b)?;
        assert!(from_b.export_volume.item::<f64>()? > 0.0);
        Ok(())
    }
}
