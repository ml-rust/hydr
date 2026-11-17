//! Homogeneous HLL fluxes for already reconstructed rectangular-water states.

use super::tensor_types::{TensorFaceState, check_finite, check_state, constant};
use super::tensor_wave;
use crate::contract::{EngineError, SolverConfig};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

/// Shared flux and signal bounds without hydrostatic side corrections.
pub struct TensorHomogeneousFlux<R: Runtime<DType = DType>> {
    pub shared: [Tensor<R>; 3],
    /// Signal magnitude `max(|lower|, |upper|)`.
    pub speed: Tensor<R>,
    /// Signed lower signal bound.
    pub lower: Tensor<R>,
    /// Signed upper signal bound.
    pub upper: Tensor<R>,
}

/// Calculate HLL fluxes without bed reconstruction or side-pressure corrections.
///
/// Inputs contain equal-length finite F64 arrays in a common normal/tangent frame.
pub fn homogeneous_hll_tensor<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorHomogeneousFlux<R>, EngineError>
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
    check_inputs(client, left, right, gravity, config)?;
    let result = calculate(client, left, right, gravity, config)?;
    check_finite(client, &result.speed, "face speed")?;
    Ok(result)
}

/// Calculate checked signed HLL signal bounds `(lower, upper)` without unused fluxes.
pub fn homogeneous_wave_bounds_tensor<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<(Tensor<R>, Tensor<R>), EngineError>
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
    check_inputs(client, left, right, gravity, config)?;
    let (lower, upper) = tensor_wave::bounds(client, left, right, gravity, config)?;
    check_finite(client, &lower, "lower face speed")?;
    check_finite(client, &upper, "upper face speed")?;
    Ok((lower, upper))
}

pub(super) fn check_inputs<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<(), EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    if left.h.shape().len() != 1 {
        return Err(EngineError::rejected(
            "face inputs",
            "require one-dimensional tensor arrays",
        ));
    }
    check_state(client, left, left.h.shape())?;
    check_state(client, right, left.h.shape())?;
    if !gravity.is_finite()
        || gravity <= 0.0
        || !config.root_absolute_depth.is_finite()
        || config.root_absolute_depth <= 0.0
        || !config.root_relative_depth.is_finite()
        || config.root_relative_depth < 0.0
        || config.root_iterations == 0
    {
        return Err(EngineError::rejected(
            "face config",
            "require positive finite gravity and absolute depth tolerance, finite nonnegative relative tolerance, and positive iteration limits",
        ));
    }
    Ok(())
}

// Surface fluxes retain their side-pressure check order around this calculation.
pub(super) fn calculate<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorHomogeneousFlux<R>, EngineError>
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
    let zero = constant(&left.h, 0.0)?;
    let one = constant(&left.h, 1.0)?;
    let (sl, sr) = tensor_wave::bounds(client, left, right, gravity, config)?;
    let left_only = client.ge(&sl, &zero)?;
    let right_only = client.le(&sr, &zero)?;
    let crossing = client.mul(&client.lt(&sl, &zero)?, &client.gt(&sr, &zero)?)?;
    let span = client.where_cond(&crossing, &client.sub(&sr, &sl)?, &one)?;
    check_finite(client, &span, "selected HLL wave span")?;
    let wl = client.div(&sr, &span)?;
    let wr = client.div(&client.neg(&sl)?, &span)?;
    let diffusion = client.mul(&wl, &sl)?;
    let fl = physical_flux(client, left, gravity)?;
    let fr = physical_flux(client, right, gravity)?;
    let ul = [&left.h, &left.q_normal, &left.q_tangent];
    let ur = [&right.h, &right.q_normal, &right.q_tangent];
    let component = |index| -> Result<Tensor<R>, EngineError> {
        let middle = client.add(
            &client.add(&client.mul(&wl, &fl[index])?, &client.mul(&wr, &fr[index])?)?,
            &client.mul(&diffusion, &client.sub(ur[index], ul[index])?)?,
        )?;
        let selected = client.where_cond(
            &left_only,
            &fl[index],
            &client.where_cond(&right_only, &fr[index], &middle)?,
        )?;
        check_finite(client, &selected, "selected HLL flux")?;
        Ok(selected)
    };
    let shared = [component(0)?, component(1)?, component(2)?];
    let speed = client.maximum(&client.abs(&sl)?, &client.abs(&sr)?)?;
    Ok(TensorHomogeneousFlux {
        shared,
        speed,
        lower: sl,
        upper: sr,
    })
}

fn physical_flux<R, C>(
    client: &C,
    state: &TensorFaceState<R>,
    gravity: f64,
) -> Result<[Tensor<R>; 3], EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R>,
{
    let wet = client.gt(&state.h, &constant(&state.h, 0.0)?)?;
    let denominator = client.where_cond(&wet, &state.h, &constant(&state.h, 1.0)?)?;
    let velocity = client.div(&state.q_normal, &denominator)?;
    Ok([
        state.q_normal.clone(),
        client.add(
            &client.mul(&state.q_normal, &velocity)?,
            &client.mul(&client.mul_scalar(&state.h, 0.5 * gravity)?, &state.h)?,
        )?,
        client.mul(&state.q_tangent, &velocity)?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    fn state(
        device: &CpuDevice,
        depth: f64,
        discharge: f64,
        bed: f64,
    ) -> Result<TensorFaceState<CpuRuntime>, EngineError> {
        let make = |value| Tensor::<CpuRuntime>::from_slice(&[value], &[1], device);
        Ok(TensorFaceState {
            h: make(depth)?,
            q_normal: make(discharge)?,
            q_tangent: make(0.0)?,
            bed: make(bed)?,
        })
    }

    #[test]
    fn homogeneous_flux_uses_supplied_reconstructed_states() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let result = homogeneous_hll_tensor(
            &client,
            &state(&device, 2.0, 6.0, 0.0)?,
            &state(&device, 2.0, 6.0, 100.0)?,
            9.81,
            &SolverConfig::default(),
        )?;
        for (actual, expected) in result.shared.iter().zip([6.0, 37.62, 0.0]) {
            assert!((actual.item::<f64>()? - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn homogeneous_flux_rejects_dry_discharge() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let result = homogeneous_hll_tensor(
            &client,
            &state(&device, 0.0, 1.0, 0.0)?,
            &state(&device, 1.0, 0.0, 0.0)?,
            9.81,
            &SolverConfig::default(),
        );
        assert!(matches!(
            result,
            Err(EngineError::Rejected { reason, .. }) if reason.contains("dry face has discharge")
        ));
        Ok(())
    }
}
