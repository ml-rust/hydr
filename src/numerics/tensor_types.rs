//! CPU tensor face inputs and scalar error reductions.

use crate::solver::EngineError;
use numr::dtype::DType;
use numr::prelude::{BinaryOps, CompareOps, CpuClient, ReduceOps, UnaryOps};

pub(crate) use crate::cpu::CpuTensor;

/// Finite F64 one-dimensional arrays in a common face normal/tangent frame.
#[derive(Clone)]
pub(crate) struct TensorFaceState {
    pub h: CpuTensor,
    pub q_normal: CpuTensor,
    pub q_tangent: CpuTensor,
    pub bed: CpuTensor,
}

/// Shared conservative flux and normal pressure additions for each side.
pub(crate) struct TensorFaceFlux {
    pub shared: [CpuTensor; 3],
    pub left_pressure: CpuTensor,
    pub right_pressure: CpuTensor,
    pub speed: CpuTensor,
}

pub(super) fn constant(like: &CpuTensor, value: f64) -> Result<CpuTensor, EngineError> {
    Ok(CpuTensor::full_scalar(
        like.shape(),
        DType::F64,
        value,
        like.device(),
    )?)
}

pub(super) fn any(client: &CpuClient, mask: &CpuTensor) -> Result<bool, EngineError> {
    if mask.numel() == 0 {
        return Ok(false);
    }
    Ok(client.max(mask, &[0], false)?.item::<f64>()? != 0.0)
}

pub(super) fn check_finite(
    client: &CpuClient,
    value: &CpuTensor,
    operation: &'static str,
) -> Result<(), EngineError> {
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

pub(super) fn check_state(
    client: &CpuClient,
    state: &TensorFaceState,
    shape: &[usize],
) -> Result<(), EngineError> {
    for value in [&state.h, &state.q_normal, &state.q_tangent, &state.bed] {
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
