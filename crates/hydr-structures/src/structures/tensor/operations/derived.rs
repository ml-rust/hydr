//! Typed static ownership and tensor metadata admission without state projection.
use numr::ops::BinaryOps;
use numr::ops::CompareOps;
use numr::ops::ConditionalOps;
use numr::{dtype::DType, runtime::Runtime};

use super::{DerivedPumpOperations, GateBinding, TensorStructureContext};
use crate::structures::{StructureError, StructureInputErrorKind, StructureLaw};
use numr::runtime::Device;
use numr::tensor::Tensor;

#[derive(Clone, Copy)]
enum Law {
    Gate,
    Pump,
}
pub(super) fn gate_slots<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    slots: &[usize],
) -> Result<Vec<f64>, StructureError> {
    slots_for(context, slots, Law::Gate)
}
pub(super) fn pump_slots<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    slots: &[usize],
) -> Result<Vec<f64>, StructureError> {
    slots_for(context, slots, Law::Pump)
}
fn slots_for<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    slots: &[usize],
    law: Law,
) -> Result<Vec<f64>, StructureError> {
    let mut flags = vec![0.0; context.elements.len()];
    let mut previous = None;
    for &slot in slots {
        if slot >= flags.len() || previous.is_some_and(|old| old >= slot) {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                match law {
                    Law::Gate => "derived gates must follow unique complete-context order",
                    Law::Pump => "derived pumps must follow unique complete-context order",
                },
            ));
        }
        let supported = matches!(
            (&law, &context.elements[slot].law),
            (Law::Gate, StructureLaw::HydrostaticRectangularGate(_))
                | (Law::Pump, StructureLaw::Pump(_))
        );
        if !supported {
            return Err(StructureError::input(
                StructureInputErrorKind::Identity,
                match law {
                    Law::Gate => "derived operation ownership requires an actual gate",
                    Law::Pump => "derived operation ownership requires an actual pump",
                },
            )
            .at(slot));
        }
        flags[slot] = 1.0;
        previous = Some(slot);
    }
    Ok(flags)
}
fn layout<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    values: &[&Tensor<R>],
    message: &str,
) -> Result<(), StructureError> {
    for value in values {
        if value.shape() != [context.elements.len()]
            || value.dtype() != DType::F64
            || !value.device().is_same(&context.device)
        {
            return Err(StructureError::numerical(message));
        }
    }
    Ok(())
}
pub(super) fn gate<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    gate: &GateBinding<'_, R>,
    flags: &[f64],
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    layout(
        context,
        &[gate.mask, gate.opening, &gate.isolated],
        &format!(
            "derived gate tensor metadata differs from complete {} F64 structure layout",
            R::name().to_ascii_uppercase()
        ),
    )?;
    let c = &context.client;
    context.reject(
        &c.ne(gate.mask, &context.tensor(flags)?)?,
        "derived gate ownership mask disagrees with checked static slots",
    )?;
    binary(
        context,
        gate.mask,
        &gate.isolated,
        "derived gate isolation mask must contain only checked binary flags",
    )?;
    context.reject(
        &c.mul(
            gate.mask,
            &c.mul(&gate.isolated, &c.ne(gate.opening, &context.zero)?)?,
        )?,
        "derived gate closure certificate contradicts represented opening",
    )
}
fn binary<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    mask: &Tensor<R>,
    value: &Tensor<R>,
    message: &str,
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    let c = &context.client;
    context.reject(
        &c.mul(
            mask,
            &c.mul(&c.ne(value, &context.zero)?, &c.ne(value, &context.one)?)?,
        )?,
        message,
    )
}
pub(super) fn pump<R: Runtime<DType = DType>>(
    context: &TensorStructureContext<R>,
    pump: &DerivedPumpOperations<'_, R>,
    flags: &[f64],
) -> Result<(), StructureError>
where
    R::Client: numr::ops::BinaryOps<R>
        + numr::ops::CompareOps<R>
        + numr::ops::ConditionalOps<R>
        + numr::ops::ReduceOps<R>
        + numr::ops::TypeConversionOps<R>
        + numr::ops::UnaryOps<R>,
{
    layout(
        context,
        &[
            pump.mask,
            pump.running,
            pump.isolated,
            pump.speed,
            pump.power_of_two,
        ],
        &format!(
            "derived pump tensor metadata differs from complete {} F64 structure layout",
            R::name().to_ascii_uppercase()
        ),
    )?;
    let c = &context.client;
    context.reject(
        &c.ne(pump.mask, &context.tensor(flags)?)?,
        "derived pump ownership mask disagrees with checked static slots",
    )?;
    for value in [pump.running, pump.isolated, pump.power_of_two] {
        binary(
            context,
            pump.mask,
            value,
            "derived pump running, isolation, and power metadata must contain only checked binary flags",
        )?;
    }
    context.reject(
        &c.mul(
            pump.mask,
            &c.ne(&c.add(pump.running, pump.isolated)?, &context.one)?,
        )?,
        "derived pump running and stopped isolation must be exclusive and exhaustive",
    )?;
    let speed = c.where_cond(pump.mask, pump.speed, &context.one)?;
    context.positive(&speed, "derived positive finite pump speed operand")?;
    context.reject(
        &c.mul(
            pump.mask,
            &c.mul(
                pump.isolated,
                &c.maximum(
                    &c.ne(pump.speed, &context.one)?,
                    &c.ne(pump.power_of_two, &context.one)?,
                )?,
            )?,
        )?,
        "derived stopped pump requires neutral internal speed and power operands",
    )
}
