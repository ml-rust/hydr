//! Hydrostatic HLL calculations executed through numr CPU tensor operations.

use super::tensor_types::{
    CpuTensor, TensorFaceFlux, TensorFaceState, check_finite, check_state, constant,
};
use super::tensor_wave;
use crate::solver::{EngineError, SolverConfig};
use numr::prelude::{
    BinaryOps, CompareOps, ConditionalOps, CpuClient, ReduceOps, ScalarOps, UnaryOps,
};

/// Compute batched fluxes from equal-length finite F64 face state arrays.
///
/// The caller assigns face identities and rotates normal/tangent components for y-faces.
/// Reflected input states implement walls. Positive films remain wet.
pub(crate) fn hydrostatic_hll_tensor(
    client: &CpuClient,
    left: TensorFaceState,
    right: TensorFaceState,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorFaceFlux, EngineError> {
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

fn calculate(
    client: &CpuClient,
    left: &TensorFaceState,
    right: &TensorFaceState,
    gravity: f64,
    config: &SolverConfig,
) -> Result<TensorFaceFlux, EngineError> {
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
    let zero = constant(&left.h, 0.0)?;
    let one = constant(&left.h, 1.0)?;
    let bed = client.maximum(&left.bed, &right.bed)?;
    let l = reconstruct(client, left, &bed)?;
    let r = reconstruct(client, right, &bed)?;
    let lp = pressure_correction(client, &left.h, &l.h, gravity)?;
    let rp = pressure_correction(client, &right.h, &r.h, gravity)?;
    let (sl, sr) = tensor_wave::bounds(client, &l, &r, gravity, config)?;
    let left_only = client.ge(&sl, &zero)?;
    let right_only = client.le(&sr, &zero)?;
    let crossing = client.mul(&client.lt(&sl, &zero)?, &client.gt(&sr, &zero)?)?;
    let span = client.where_cond(&crossing, &client.sub(&sr, &sl)?, &one)?;
    check_finite(client, &span, "selected HLL wave span")?;
    let wl = client.div(&sr, &span)?;
    let wr = client.div(&client.neg(&sl)?, &span)?;
    let diffusion = client.mul(&wl, &sl)?;
    let fl = physical_flux(client, &l, gravity)?;
    let fr = physical_flux(client, &r, gravity)?;
    let ul = [&l.h, &l.q_normal, &l.q_tangent];
    let ur = [&r.h, &r.q_normal, &r.q_tangent];
    let component = |index| -> Result<CpuTensor, EngineError> {
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
    check_finite(client, &lp, "left pressure correction")?;
    check_finite(client, &rp, "right pressure correction")?;
    check_finite(client, &speed, "face speed")?;
    Ok(TensorFaceFlux {
        shared,
        left_pressure: lp,
        right_pressure: rp,
        speed,
    })
}

fn reconstruct(
    client: &CpuClient,
    state: &TensorFaceState,
    bed: &CpuTensor,
) -> Result<TensorFaceState, EngineError> {
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

fn pressure_correction(
    client: &CpuClient,
    h: &CpuTensor,
    reconstructed: &CpuTensor,
    gravity: f64,
) -> Result<CpuTensor, EngineError> {
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

fn physical_flux(
    client: &CpuClient,
    state: &TensorFaceState,
    gravity: f64,
) -> Result<[CpuTensor; 3], EngineError> {
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

fn state_summary(client: &CpuClient, state: &TensorFaceState) -> String {
    let mut result = format!("shape={:?}", state.h.shape());
    if state.h.shape().len() == 1 && state.h.numel() > 0 {
        for (name, value) in [
            ("h", &state.h),
            ("qn", &state.q_normal),
            ("qt", &state.q_tangent),
            ("bed", &state.bed),
        ] {
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
    use numr::prelude::{CpuDevice, CpuRuntime, Runtime};

    #[test]
    fn tensor_batches_match_independent_scalar_face_arithmetic() -> Result<(), EngineError> {
        use crate::numerics::flux::{FaceState, WaveConfig, hydrostatic_hll};
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
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
        let tensor = |states: &[FaceState]| -> Result<TensorFaceState, EngineError> {
            let make = |values: Vec<f64>| CpuTensor::from_slice(&values, &[count], &device);
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
    ) -> Result<TensorFaceState, EngineError> {
        let make = |x| CpuTensor::from_slice(&[x], &[1], device);
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
        let client = CpuRuntime::default_client(&device);
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
        let client = CpuRuntime::default_client(&device);
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
        let client = CpuRuntime::default_client(&device);
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
        let client = CpuRuntime::default_client(&device);
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
        let client = CpuRuntime::default_client(&device);
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
