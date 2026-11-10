use super::admission::reject_mask;
use super::types::TensorBoundaryRequest;
use crate::numerics::tensor_types::{any, check_finite, constant};
use crate::numerics::tensor_wave::curve;
use crate::solver::{BoundaryFace, EngineError, SolverConfig};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};

pub(super) fn depth<R, C>(
    client: &C,
    hi: &Tensor<R>,
    ui: &Tensor<R>,
    request: &TensorBoundaryRequest<R>,
    faces: &[BoundaryFace],
    gravity: f64,
    config: &SolverConfig,
) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let mask = &request.discharge_mask;
    if !any(client, mask)? {
        return Ok(hi.clone());
    }
    let one = constant(hi, 1.0)?;
    let zero = constant(hi, 0.0)?;
    // Stage lanes use benign wet values before either root callback evaluates its expressions.
    let hi = client.where_cond(mask, hi, &one)?;
    let ui = client.where_cond(mask, ui, &zero)?;
    let target = client.where_cond(mask, &request.outward_discharge, &zero)?;
    let ci = client.mul_scalar(&client.sqrt(&hi)?, gravity.sqrt())?;
    let sonic_c = client.mul_scalar(&client.add(&ui, &client.mul_scalar(&ci, 2.0)?)?, 1.0 / 3.0)?;
    let sonic_sqrt_h = client.div_scalar(&sonic_c, gravity.sqrt())?;
    let lower = client.mul(&sonic_sqrt_h, &sonic_sqrt_h)?;
    reject_mask(
        client,
        &client.le(&lower, &zero)?,
        faces,
        "Sonic lower depth is unrepresentable. Increase the physical depth scale",
        &[("interior depth", &hi), ("sonic depth", &lower)],
    )?;
    let options = BatchedBisectionOptions {
        atol: f64::from_bits(1),
        rtol: config.root_relative_depth.min(1e-14),
        max_iter: config.root_iterations,
        max_expansions: config.root_iterations,
        expansion_factor: 2.0,
    };
    let seed = client.mul_scalar(&hi, 2.0)?;
    check_finite(client, &seed, "boundary sonic upper seed")?;
    let sonic = client
        .bisect_batched(
            |h| {
                client.add(
                    &client.sub(&ui, &curve(client, h, &hi, gravity)?)?,
                    &client.mul_scalar(&client.sqrt(h)?, gravity.sqrt())?,
                )
            },
            &hi,
            &seed,
            &options,
        )
        .map_err(root_error)?;
    // u+c decreases. Its retained lower endpoint remains on the subcritical side.
    let upper = sonic.lower;
    let discharge = |h: &Tensor<R>| -> numr::error::Result<Tensor<R>> {
        client.mul(h, &client.sub(&ui, &curve(client, h, &hi, gravity)?)?)
    };
    let maximum = discharge(&lower)?;
    let minimum = discharge(&upper)?;
    check_finite(client, &maximum, "boundary maximum discharge")?;
    check_finite(client, &minimum, "boundary minimum discharge")?;
    let outside = client.maximum(
        &client.gt(&target, &maximum)?,
        &client.lt(&target, &minimum)?,
    )?;
    reject_mask(
        client,
        &client.mul(mask, &outside)?,
        faces,
        "Discharge lies outside the connected subcritical branch. Supply an admissible rate or complete exterior state",
        &[
            ("requested discharge", &target),
            ("minimum discharge", &minimum),
            ("maximum discharge", &maximum),
        ],
    )?;
    // Critical rate endpoints lose velocity information. Require resolved separation from each endpoint independently.
    let upper_gap = client.sub(&maximum, &target)?;
    let lower_gap = client.sub(&target, &minimum)?;
    let upper_scale = client.maximum(&client.abs(&maximum)?, &client.abs(&target)?)?;
    let lower_scale = client.maximum(&client.abs(&minimum)?, &client.abs(&target)?)?;
    for value in [&upper_gap, &lower_gap, &upper_scale, &lower_scale] {
        check_finite(client, value, "boundary critical-rate separation")?;
    }
    let unresolved = client.maximum(
        &client.le(
            &upper_gap,
            &client.mul_scalar(&upper_scale, config.boundary.critical_relative)?,
        )?,
        &client.le(
            &lower_gap,
            &client.mul_scalar(&lower_scale, config.boundary.critical_relative)?,
        )?,
    )?;
    reject_mask(
        client,
        &client.mul(mask, &unresolved)?,
        faces,
        "Discharge cannot distinguish a critical endpoint at numerical precision. Supply a resolved subcritical rate or complete exterior state",
        &[
            ("requested discharge", &target),
            ("minimum discharge", &minimum),
            ("maximum discharge", &maximum),
        ],
    )?;
    let options = BatchedBisectionOptions {
        max_expansions: 0,
        ..options
    };
    let result = client
        .bisect_batched(
            |h| client.sub(&discharge(h)?, &target),
            &lower,
            &upper,
            &options,
        )
        .map_err(root_error)?;
    Ok(result.upper)
}

fn root_error(error: solvr::optimize::OptimizeError) -> EngineError {
    EngineError::rejected(
        "boundary root",
        format!("{error}: inspect the subcritical branch and configured root tolerances"),
    )
}
