use super::admission::reject_mask;
use crate::contract::{BoundaryFace, EngineError, SolverConfig};
use crate::numerics::tensor_types::{TensorFaceState, check_state, constant};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};

pub(crate) fn check_transmissive<R, C>(
    client: &C,
    state: &TensorFaceState<R>,
    faces: &[BoundaryFace],
    config: &SolverConfig,
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    check_state(client, state, &[faces.len()])?;
    let zero = constant(&state.h, 0.0)?;
    let wet = client.gt(&state.h, &zero)?;
    let depth = client.where_cond(&wet, &state.h, &constant(&state.h, 1.0)?)?;
    let velocity = client.div(&state.q_normal, &depth)?;
    let tangent = client.div(&state.q_tangent, &depth)?;
    let maximum = constant(&state.h, f64::MAX)?;
    let finite = client.mul(
        &client.le(&client.abs(&velocity)?, &maximum)?,
        &client.le(&client.abs(&tangent)?, &maximum)?,
    )?;
    reject_mask(
        client,
        &client.rsub_scalar(&finite, 1.0)?,
        faces,
        "Transmissive velocity is unrepresentable. Supply compatible exterior data or rescale the state",
        &[
            ("normal velocity", &velocity),
            ("tangent velocity", &tangent),
        ],
    )?;
    let c = client.mul_scalar(&client.sqrt(&state.h)?, config.gravity.sqrt())?;
    let margin = client.mul_scalar(
        &client.maximum(&client.abs(&velocity)?, &c)?,
        config.boundary.critical_relative,
    )?;
    let invalid = client.mul(&wet, &client.le(&client.sub(&velocity, &c)?, &margin)?)?;
    reject_mask(
        client,
        &invalid,
        faces,
        "Transmissive boundary has incoming or unresolved critical characteristics. Supply complete exterior-state data",
        &[
            ("depth", &state.h),
            ("outward velocity", &velocity),
            ("celerity", &c),
        ],
    )
}
