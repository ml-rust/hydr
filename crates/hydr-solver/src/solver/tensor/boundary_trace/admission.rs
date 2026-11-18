use super::types::TensorBoundaryRequest;
use crate::contract::{BoundaryFace, EngineError};
use crate::numerics::tensor_types::{TensorFaceState, any, check_finite, check_state, constant};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;

pub(super) fn check_inputs<R, C>(
    client: &C,
    interior: &TensorFaceState<R>,
    request: &TensorBoundaryRequest<R>,
    faces: &[BoundaryFace],
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ReduceOps<R> + ScalarOps<R> + UnaryOps<R>,
{
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
        if !tensor.device().is_same(client.device()) {
            return Err(EngineError::rejected(
                "boundary trace inputs",
                format!(
                    "{name}: tensor belongs to another device context. Rebuild the request with the selected client"
                ),
            ));
        }
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

pub(super) fn reject_mask<R, C>(
    client: &C,
    mask: &Tensor<R>,
    faces: &[BoundaryFace],
    message: &str,
    details: &[(&str, &Tensor<R>)],
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R>,
{
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
                "inconsistent reduction: inspect numr comparisons",
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

#[cfg(test)]
mod tests {
    use super::super::fixture::{G, Lane, lane, run};
    use super::*;
    use crate::contract::Axis;

    #[test]
    fn dry_critical_supercritical_and_inadmissible_targets_reject() {
        for request in [
            Lane {
                h: 0.0,
                u: 0.0,
                ..lane()
            },
            Lane {
                u: G.sqrt(),
                ..lane()
            },
            Lane { u: 4.0, ..lane() },
            Lane { u: -4.0, ..lane() },
            Lane {
                stage: 0.0,
                ..lane()
            },
            Lane {
                stage: 0.01,
                ..lane()
            },
            Lane {
                stage: 10.0,
                ..lane()
            },
            Lane {
                discharge: Some(100.0),
                ..lane()
            },
        ] {
            assert!(
                matches!(run(&[request]), Err(EngineError::Boundary { .. })),
                "h={}, u={}, stage={}, q={:?}",
                request.h,
                request.u,
                request.stage,
                request.discharge
            );
        }
    }

    #[test]
    fn a_bad_second_lane_identifies_its_face_in_mixed_boundary_batches() {
        let expected = BoundaryFace {
            axis: Axis::X,
            column: 0,
            row: 1,
        };
        for bad in [
            Lane {
                discharge: Some(100.0),
                ..lane()
            },
            Lane {
                stage: 0.01,
                ..lane()
            },
        ] {
            assert!(
                matches!(run(&[lane(), bad]), Err(EngineError::Boundary { face, .. }) if face == expected)
            );
        }
    }
}
