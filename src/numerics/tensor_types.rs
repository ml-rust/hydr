//! Runtime tensor face inputs and scalar error reductions.

use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps, UnaryOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

/// Finite F64 one-dimensional arrays in a common face normal/tangent frame.
#[derive(Clone)]
pub(crate) struct TensorFaceState<R: Runtime<DType = DType>> {
    pub h: Tensor<R>,
    pub q_normal: Tensor<R>,
    pub q_tangent: Tensor<R>,
    pub bed: Tensor<R>,
}

/// Shared conservative flux and normal pressure additions for each side.
pub(crate) struct TensorFaceFlux<R: Runtime<DType = DType>> {
    pub shared: [Tensor<R>; 3],
    pub left_pressure: Tensor<R>,
    pub right_pressure: Tensor<R>,
    pub speed: Tensor<R>,
}

pub(crate) fn constant<R>(like: &Tensor<R>, value: f64) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
{
    Ok(Tensor::<R>::full_scalar(
        like.shape(),
        DType::F64,
        value,
        like.device(),
    )?)
}

pub(crate) fn any<R, C>(client: &C, mask: &Tensor<R>) -> Result<bool, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R>,
{
    if mask.numel() == 0 {
        return Ok(false);
    }
    Ok(client.max(mask, &[0], false)?.item::<f64>()? != 0.0)
}

pub(crate) fn check_finite<R, C>(
    client: &C,
    value: &Tensor<R>,
    operation: &'static str,
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + CompareOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    if value.numel() == 0 {
        return Ok(());
    }
    let finite = client.le(&client.abs(value)?, &constant(value, f64::MAX)?)?;
    if client.min(&finite, &[0], false)?.item::<f64>()? != 1.0 {
        return Err(EngineError::rejected(
            operation,
            "nonfinite tensor value: inspect face states and numerical scales",
        ));
    }
    Ok(())
}

pub(crate) fn check_state<R, C>(
    client: &C,
    state: &TensorFaceState<R>,
    shape: &[usize],
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    for value in [&state.h, &state.q_normal, &state.q_tangent, &state.bed] {
        if !value.device().is_same(client.device()) {
            return Err(EngineError::rejected(
                "face inputs",
                "tensor belongs to another device context: rebuild face inputs with the selected client",
            ));
        }
        if value.shape() != shape || value.dtype() != DType::F64 {
            return Err(EngineError::rejected(
                "face inputs",
                "require equal one-dimensional F64 tensor shapes",
            ));
        }
        check_finite(client, value, "face inputs")?;
    }
    let zero = constant(&state.h, 0.0)?;
    if any(client, &client.lt(&state.h, &zero)?)? {
        return Err(EngineError::rejected(
            "face inputs",
            "negative face depth: inspect input state",
        ));
    }
    let dry = client.eq(&state.h, &zero)?;
    let discharge = client.maximum(
        &client.abs(&state.q_normal)?,
        &client.abs(&state.q_tangent)?,
    )?;
    if any(client, &client.mul(&dry, &client.gt(&discharge, &zero)?)?)? {
        return Err(EngineError::rejected(
            "face inputs",
            "dry face has discharge: inspect input state",
        ));
    }
    Ok(())
}
