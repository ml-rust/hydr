//! Hydrostatic HLL calculations executed through numr tensor operations.

use super::tensor_hll;
use super::tensor_types::{TensorFaceFlux, TensorFaceState, check_finite, constant};
use crate::contract::{EngineError, SolverConfig};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Device, Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;

/// Compute batched fluxes from equal-length finite F64 face state arrays.
///
/// The caller assigns face identities and rotates normal/tangent components for y-faces.
/// Reflected input states implement walls. Positive films remain wet.
pub(crate) fn hydrostatic_hll_tensor<R, C>(
    client: &C,
    left: TensorFaceState<R>,
    right: TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorFaceFlux<R>, EngineError>
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
    calculate(client, &left, &right, gravity, config).map_err(|error| {
        EngineError::rejected(
            "tensor face flux",
            format!(
                "{error}; left {}; right {}; inspect the corresponding input faces",
                state_summary(client, &left),
                state_summary(client, &right)
            ),
        )
    })
}

fn calculate<R, C>(
    client: &C,
    left: &TensorFaceState<R>,
    right: &TensorFaceState<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorFaceFlux<R>, EngineError>
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
    tensor_hll::check_inputs(client, left, right, gravity, config)?;
    let bed = client.maximum(&left.bed, &right.bed)?;
    let l = reconstruct(client, left, &bed)?;
    let r = reconstruct(client, right, &bed)?;
    let lp = pressure_correction(client, &left.h, &l.h, gravity)?;
    let rp = pressure_correction(client, &right.h, &r.h, gravity)?;
    let homogeneous = tensor_hll::calculate(client, &l, &r, gravity, config)?;
    check_finite(client, &lp, "left pressure correction")?;
    check_finite(client, &rp, "right pressure correction")?;
    check_finite(client, &homogeneous.speed, "face speed")?;
    Ok(TensorFaceFlux {
        shared: homogeneous.shared,
        left_pressure: lp,
        right_pressure: rp,
        speed: homogeneous.speed,
    })
}

fn reconstruct<R, C>(
    client: &C,
    state: &TensorFaceState<R>,
    bed: &Tensor<R>,
) -> Result<TensorFaceState<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + UnaryOps<R>,
{
    let zero = constant(&state.h, 0.0)?;
    let one = constant(&state.h, 1.0)?;
    let step = client.sub(bed, &state.bed)?;
    check_finite(client, &step, "hydrostatic bed step")?;
    // Removing the bed step before subtraction retains films at large vertical datums.
    let h = client.maximum(&client.sub(&state.h, &step)?, &zero)?;
    let denominator = client.where_cond(&client.gt(&state.h, &zero)?, &state.h, &one)?;
    let ratio = client.div(&h, &denominator)?;
    Ok(TensorFaceState {
        h,
        q_normal: client.mul(&state.q_normal, &ratio)?,
        q_tangent: client.mul(&state.q_tangent, &ratio)?,
        bed: bed.clone(),
    })
}

fn pressure_correction<R, C>(
    client: &C,
    h: &Tensor<R>,
    reconstructed: &Tensor<R>,
    gravity: f64,
) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + CompareOps<R> + ConditionalOps<R> + ScalarOps<R>,
{
    let correction = client.mul(
        &client.mul_scalar(&client.sub(h, reconstructed)?, 0.5 * gravity)?,
        &client.add(h, reconstructed)?,
    )?;
    Ok(client.where_cond(
        &client.eq(h, reconstructed)?,
        &constant(h, 0.0)?,
        &correction,
    )?)
}

fn state_summary<R, C>(client: &C, state: &TensorFaceState<R>) -> String
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ReduceOps<R>,
{
    let mut result = format!("shape={:?}", state.h.shape());
    if state.h.shape().len() == 1 && state.h.numel() > 0 {
        for (name, value) in [
            ("h", &state.h),
            ("qn", &state.q_normal),
            ("qt", &state.q_tangent),
            ("bed", &state.bed),
        ] {
            if !value.device().is_same(client.device()) {
                result.push_str(&format!(" {name}=different device context"));
                continue;
            }
            let minimum = client.min(value, &[0], false).and_then(|x| x.item::<f64>());
            let maximum = client.max(value, &[0], false).and_then(|x| x.item::<f64>());
            result.push_str(&format!(" {name}=[{minimum:?},{maximum:?}]"));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn tensor_batches_match_independent_scalar_face_arithmetic() -> Result<(), EngineError> {
        use crate::numerics::flux::{FaceState, WaveConfig, hydrostatic_hll};
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let count = 48;
        let mut left = Vec::with_capacity(count);
        let mut right = Vec::with_capacity(count);
        for index in 0..count {
            let x = index as f64;
            let hl = 0.2 + 0.05 * (index % 11) as f64;
            let hr = 0.1 + 0.07 * (index % 7) as f64;
            left.push(FaceState {
                h: hl,
                q_normal: hl * x.sin(),
                q_tangent: hl * x.cos(),
                bed: 0.03 * (index % 5) as f64,
            });
            right.push(FaceState {
                h: hr,
                q_normal: hr * (0.3 * x).cos(),
                q_tangent: -hr * x.sin(),
                bed: 0.04 * (index % 3) as f64,
            });
        }
        let tensor = |states: &[FaceState]| -> Result<TensorFaceState<CpuRuntime>, EngineError> {
            let make =
                |values: Vec<f64>| Tensor::<CpuRuntime>::from_slice(&values, &[count], &device);
            Ok(TensorFaceState {
                h: make(states.iter().map(|s| s.h).collect())?,
                q_normal: make(states.iter().map(|s| s.q_normal).collect())?,
                q_tangent: make(states.iter().map(|s| s.q_tangent).collect())?,
                bed: make(states.iter().map(|s| s.bed).collect())?,
            })
        };
        let result = hydrostatic_hll_tensor(
            &client,
            tensor(&left)?,
            tensor(&right)?,
            9.81,
            &SolverConfig::default(),
        )?;
        let shared = [
            result.shared[0].try_to_vec::<f64>()?,
            result.shared[1].try_to_vec::<f64>()?,
            result.shared[2].try_to_vec::<f64>()?,
        ];
        let pl = result.left_pressure.try_to_vec::<f64>()?;
        let pr = result.right_pressure.try_to_vec::<f64>()?;
        let speeds = result.speed.try_to_vec::<f64>()?;
        for index in 0..count {
            let expected = hydrostatic_hll(left[index], right[index], 9.81, WaveConfig::default())
                .map_err(|error| EngineError::rejected("scalar oracle", error.to_string()))?;
            for (component, values) in shared.iter().enumerate() {
                assert!(
                    (values[index] - expected.shared[component]).abs() <= 1e-9,
                    "face={index}, component={component}"
                );
            }
            assert!((pl[index] - expected.left_pressure).abs() <= 1e-9);
            assert!((pr[index] - expected.right_pressure).abs() <= 1e-9);
            assert!((speeds[index] - expected.speed).abs() <= 1e-9);
        }
        Ok(())
    }

    fn state(
        device: &CpuDevice,
        h: f64,
        q: f64,
        qt: f64,
        bed: f64,
    ) -> Result<TensorFaceState<CpuRuntime>, EngineError> {
        let make = |x| Tensor::<CpuRuntime>::from_slice(&[x], &[1], device);
        Ok(TensorFaceState {
            h: make(h)?,
            q_normal: make(q)?,
            q_tangent: make(qt)?,
            bed: make(bed)?,
        })
    }

    #[test]
    fn tensor_uniform_flow_transports_tangent_momentum() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let result = hydrostatic_hll_tensor(
            &client,
            state(&device, 2.0, 6.0, -8.0, 0.0)?,
            state(&device, 2.0, 6.0, -8.0, 0.0)?,
            9.81,
            &SolverConfig::default(),
        )?;
        for (actual, expected) in result.shared.iter().zip([6.0, 37.62, -24.0]) {
            assert!((actual.item::<f64>()? - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn tensor_compressed_films_have_ordered_finite_bounds() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        for (hl, ql, hr, qr) in [
            (
                5.7660025090732485e-86,
                1.0669783925783614e-85,
                4.440970925099901e-89,
                8.183764274505537e-89,
            ),
            (
                5.594580270536242e-47,
                1.0997289785984347e-46,
                7.233289121933466e-49,
                1.4121132034678347e-48,
            ),
            (
                9.880806210145298e-43,
                2.178525493033415e-42,
                2.1069039122226898e-45,
                4.5328001501395363e-45,
            ),
        ] {
            let result = hydrostatic_hll_tensor(
                &client,
                state(&device, hl, ql, 0.0, 0.0)?,
                state(&device, hr, qr, 0.0, 0.0)?,
                9.81,
                &SolverConfig::default(),
            )?;
            assert!(result.speed.item::<f64>()?.is_finite());
        }
        Ok(())
    }

    #[test]
    fn tensor_stepped_lake_retains_both_pressures() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let result = hydrostatic_hll_tensor(
            &client,
            state(&device, 2.0, 0.0, 0.0, 0.0)?,
            state(&device, 1.0, 0.0, 0.0, 1.0)?,
            9.81,
            &SolverConfig::default(),
        )?;
        assert_eq!(result.shared[0].item::<f64>()?, 0.0);
        assert!(
            (result.shared[1].item::<f64>()? + result.left_pressure.item::<f64>()? - 19.62).abs()
                < 1e-12
        );
        assert!(
            (result.shared[1].item::<f64>()? + result.right_pressure.item::<f64>()? - 4.905).abs()
                < 1e-12
        );
        Ok(())
    }

    #[test]
    fn tensor_dry_and_vacuum_bounds_cover_analytic_fronts() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        for (hl, ql, hr, qr, expected) in [
            (1.0, 0.0, 0.0, 0.0, 2.0 * 9.81_f64.sqrt()),
            (0.0, 0.0, 1.0, 0.0, 2.0 * 9.81_f64.sqrt()),
            (1.0, -8.0, 1.0, 8.0, 8.0 + 9.81_f64.sqrt()),
            (0.0, 0.0, 0.0, 0.0, 0.0),
        ] {
            let result = hydrostatic_hll_tensor(
                &client,
                state(&device, hl, ql, 0.0, 0.0)?,
                state(&device, hr, qr, 0.0, 0.0)?,
                9.81,
                &SolverConfig::default(),
            )?;
            let speed = result.speed.item::<f64>()?;
            assert!(speed >= expected);
            assert!((speed - expected).abs() < 1e-12);
        }
        Ok(())
    }

    #[test]
    fn tensor_root_iteration_exhaustion_returns_an_error() -> Result<(), EngineError> {
        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let config = SolverConfig {
            root_iterations: 1,
            ..SolverConfig::default()
        };
        let result = hydrostatic_hll_tensor(
            &client,
            state(&device, 1.0, 0.0, 0.0, 0.0)?,
            state(&device, 0.25, 0.0, 0.0, 0.0)?,
            9.81,
            &config,
        );
        assert!(
            matches!(result, Err(EngineError::Rejected { reason, .. }) if reason.contains("iteration limit reached"))
        );
        Ok(())
    }
}
