//! Shape-preserving outward arithmetic and checked exact expansions.

use crate::structures::StructureError;
use crate::structures::tensor::actuator::TensorActuatorContext;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, TwoSumOps, TypeConversionOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

pub(super) fn finite<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::TypeConversionOps<R> + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    Ok(c.sub(
        &ctx.one,
        &c.maximum(
            &c.cast(&c.isnan(x)?, x.dtype())?,
            &c.cast(&c.isinf(x)?, x.dtype())?,
        )?,
    )?)
}
pub(super) fn bound<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(crate::numerics::rounding::rounding_with_client(
        &ctx.structures.client,
        x,
    )?)
}
pub(super) fn up<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(ctx.structures.client.add(x, &bound(ctx, x)?)?)
}
pub(super) fn down<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ScalarOps<R> + numr::ops::UnaryOps<R>,
{
    Ok(ctx.structures.client.sub(x, &bound(ctx, x)?)?)
}
pub(super) fn not<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    x: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>,
{
    Ok(ctx.structures.client.sub(&ctx.one, x)?)
}
pub(super) fn used<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    active: &Tensor<R>,
    valid: &Tensor<R>,
    message: &str,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::ReduceOps<R>,
{
    ctx.reject(
        &ctx.structures.client.mul(active, &not(ctx, valid)?)?,
        message,
    )
}
pub(super) fn sum<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let a = c.where_cond(active, a, &ctx.zero)?;
    let b = c.where_cond(active, b, &ctx.zero)?;
    let high = c.add(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &high)?,
        "actuator used expansion high is nonfinite: rescale inputs",
    )?;
    let (high, low) = c.two_sum(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &low)?,
        "actuator used TwoSum residual is nonfinite: rescale inputs",
    )?;
    Ok((high, low))
}
pub(super) fn outward_add<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let a = c.where_cond(active, a, &ctx.zero)?;
    let b = c.where_cond(active, b, &ctx.zero)?;
    let s = c.add(&a, &b)?;
    used(
        ctx,
        active,
        &finite(ctx, &s)?,
        "actuator uncertainty sum overflows: rescale inputs",
    )?;
    let both = c.mul(&c.ne(&a, &ctx.zero)?, &c.ne(&b, &ctx.zero)?)?;
    let value = c.where_cond(&both, &up(ctx, &s)?, &s)?;
    used(
        ctx,
        active,
        &finite(ctx, &value)?,
        "actuator outward uncertainty overflows: rescale inputs",
    )?;
    Ok(value)
}
pub(super) fn difference_upper<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    a: &Tensor<R>,
    b: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::ScalarOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let (h, l) = sum(ctx, a, &c.neg(b)?, active)?;
    let padded = c.minimum(&up(ctx, &h)?, &ctx.maximum)?;
    Ok(c.where_cond(&c.gt(&l, &ctx.zero)?, &padded, &h)?)
}
pub(super) fn width<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    active: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TwoSumOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &ctx.structures.client;
    let (h, l) = sum(ctx, upper, &c.neg(lower)?, active)?;
    let valid = c.maximum(
        &c.lt(&h, &ctx.window)?,
        &c.mul(&c.eq(&h, &ctx.window)?, &c.le(&l, &ctx.zero)?)?,
    )?;
    used(
        ctx,
        active,
        &valid,
        "actuator full arrival window exceeds explicit seconds allowance",
    )?;
    Ok((h, l))
}
pub(super) fn count<R: Runtime<DType = DType>>(
    ctx: &TensorActuatorContext<R>,
    old: &Tensor<R>,
    increment: &Tensor<R>,
) -> Result<Tensor<R>, StructureError>
where
    R::Client: numr::ops::BinaryOps<R> + numr::ops::CompareOps<R> + numr::ops::ReduceOps<R>,
{
    let c = &ctx.structures.client;
    let next = c.add(old, increment)?;
    let absorbed = c.mul(&c.gt(increment, &ctx.zero)?, &c.le(&next, old)?)?;
    ctx.reject(
        &c.maximum(
            &absorbed,
            &c.gt(&next, &ctx.constant((1u64 << 53) as f64)?)?,
        )?,
        "actuator exact counter capacity exceeded",
    )?;
    Ok(next)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn exact_counter_capacity_preserves_last_increment_and_rejects_absorption()
    -> Result<(), Box<dyn std::error::Error>> {
        let text = || "Explicit analytical fixture".to_string();
        let element = StructureElement {
            id: "gate".into(),
            endpoint_a: "a".into(),
            endpoint_b: "b".into(),
            datum: text(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::HydrostaticRectangularGate(GateParameters::new(
                1.0,
                0.0,
                1.0,
                1.0,
                9.81,
                GateApplicability {
                    coefficient_provenance: text(),
                    hydrostatic_quiescent_pool_assertion: text(),
                    free_submerged_aperture_applicability: text(),
                    unobstructed_aperture_assertion: text(),
                    symmetric_reversal: true,
                },
            )?),
            condition: StructureCondition::AsBuilt,
        };
        let batch = StructureBatch::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            vec![element],
        )?
        .with_gate_actuators(
            vec![GateActuatorDefinition {
                actuator_id: "actuator".into(),
                structure_id: "gate".into(),
                opening_rate_m_per_s: 1.0,
                closing_rate_m_per_s: 1.0,
                accuracy: GateActuatorAccuracy {
                    maximum_absolute_opening_error_m: 0.0,
                    maximum_relative_opening_error: 0.0,
                    maximum_arrival_window_s: 0.0,
                },
                rate_provenance: text(),
                ideal_response_applicability: text(),
            }],
            vec![GateActuatorInitialState {
                actuator_id: "actuator".into(),
                opening_m: 0.0,
                target_opening_m: 0.0,
            }],
            GateActuatorSchedule {
                initial_time_s: 0.0,
                target_times_s: Vec::new(),
                end_s: 1.0,
            },
        )?;
        let ctx = &batch.context;
        let maximum = (1u64 << 53) as f64;
        assert_eq!(
            count(ctx, &ctx.constant(maximum - 1.0)?, &ctx.one)?.item::<f64>()?,
            maximum
        );
        assert_eq!(
            count(ctx, &ctx.constant(maximum)?, &ctx.zero)?.item::<f64>()?,
            maximum
        );
        assert!(matches!(
            count(ctx, &ctx.constant(maximum)?, &ctx.one),
            Err(StructureError::Numerical { .. })
        ));
        Ok(())
    }
}
