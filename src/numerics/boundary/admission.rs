use super::types::TensorBoundaryRequest;
use crate::numerics::tensor_types::{
    CpuTensor, TensorFaceState, any, check_finite, check_state, constant,
};
use crate::solver::{BoundaryFace, EngineError};
use numr::dtype::DType;
use numr::prelude::{BinaryOps, CompareOps, CpuClient, ScalarOps};

pub(super) fn check_inputs(
    client: &CpuClient,
    interior: &TensorFaceState,
    request: &TensorBoundaryRequest,
    faces: &[BoundaryFace],
) -> Result<(), EngineError> {
    if faces.is_empty() {
        return Err(EngineError::rejected(
            "boundary trace inputs",
            "empty boundary batch: skip trace construction for an empty face set",
        ));
    }
    check_state(client, interior, &[faces.len()])?;
    for (name, tensor) in [
        ("stage depth", &request.stage_depth),
        ("outward discharge", &request.outward_discharge),
        ("discharge mask", &request.discharge_mask),
        ("incoming tangent", &request.inflow_tangent),
        ("incoming tangent mask", &request.has_inflow_tangent),
    ] {
        if tensor.shape() != [faces.len()] || tensor.dtype() != DType::F64 {
            return Err(EngineError::rejected(
                "boundary trace inputs",
                format!(
                    "{name}: require a one-dimensional F64 tensor with {} faces",
                    faces.len()
                ),
            ));
        }
        check_finite(client, tensor, "boundary trace inputs")?;
    }
    let zero = constant(&interior.h, 0.0)?;
    reject_mask(
        client,
        &client.le(&interior.h, &zero)?,
        faces,
        "Dry interior has no admitted wet characteristic. Supply a supported exterior-state boundary",
        &[("interior depth", &interior.h)],
    )?;
    reject_mask(
        client,
        &client.le(&request.stage_depth, &zero)?,
        faces,
        "Stage depth is nonpositive. Supply a wet stage or a supported exterior-state boundary",
        &[("stage depth", &request.stage_depth)],
    )?;
    let one = constant(&interior.h, 1.0)?;
    for (name, tensor) in [
        ("discharge mask", &request.discharge_mask),
        ("incoming tangent mask", &request.has_inflow_tangent),
    ] {
        let invalid = client.mul(
            &client.rsub_scalar(&client.eq(tensor, &zero)?, 1.0)?,
            &client.rsub_scalar(&client.eq(tensor, &one)?, 1.0)?,
        )?;
        reject_mask(
            client,
            &invalid,
            faces,
            "Boundary mode mask is not binary. Supply only zero or one",
            &[(name, tensor)],
        )?;
    }
    Ok(())
}

pub(super) fn reject_mask(
    client: &CpuClient,
    mask: &CpuTensor,
    faces: &[BoundaryFace],
    message: &str,
    details: &[(&str, &CpuTensor)],
) -> Result<(), EngineError> {
    if !any(client, mask)? {
        return Ok(());
    }
    // Field readback occurs only after rejection, to identify the physical face and exact input.
    let flags = mask.try_to_vec::<f64>()?;
    let lane = flags
        .iter()
        .position(|value| *value != 0.0)
        .ok_or_else(|| {
            EngineError::rejected(
                "boundary trace",
                "inconsistent reduction: inspect numr CPU comparisons",
            )
        })?;
    let face = faces.get(lane).copied().ok_or_else(|| {
        EngineError::rejected(
            "boundary trace",
            "face metadata shape differs: rebuild the boundary cache",
        )
    })?;
    let mut reason = message.to_owned();
    for (name, tensor) in details {
        let values = tensor.try_to_vec::<f64>()?;
        let value = values.get(lane).ok_or_else(|| {
            EngineError::rejected(
                "boundary trace",
                "diagnostic shape differs: rebuild the boundary cache",
            )
        })?;
        reason.push_str(&format!(". {name}={value:.17e}"));
    }
    Err(EngineError::Boundary { face, reason })
}
