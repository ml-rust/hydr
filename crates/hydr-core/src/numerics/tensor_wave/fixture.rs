use crate::contract::EngineError;
use crate::numerics::tensor_types::TensorFaceState;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

pub(super) type TestResult = Result<(), Box<dyn std::error::Error>>;

pub(super) const GRAVITY: f64 = 9.81;
pub(super) const DEPTHS: [(f64, f64); 4] = [
    (3.209e-105, 2.370e-162),
    (2.370e-162, 3.209e-105),
    (1.0, 1e-120),
    (1e-120, 1.0),
];

pub(super) fn setup() -> Result<
    (
        CpuClient,
        TensorFaceState<CpuRuntime>,
        TensorFaceState<CpuRuntime>,
    ),
    EngineError,
> {
    let device = CpuDevice::new();
    let tensor = |values: &[f64]| {
        Tensor::<CpuRuntime>::from_slice(values, &[DEPTHS.len()], &device)
            .map_err(EngineError::from)
    };
    let left = TensorFaceState {
        h: tensor(&DEPTHS.map(|(left, _)| left))?,
        q_normal: tensor(&[0.0; DEPTHS.len()])?,
        q_tangent: tensor(&[0.0; DEPTHS.len()])?,
        bed: tensor(&[0.0; DEPTHS.len()])?,
    };
    let right = TensorFaceState {
        h: tensor(&DEPTHS.map(|(_, right)| right))?,
        q_normal: left.q_normal.clone(),
        q_tangent: left.q_tangent.clone(),
        bed: left.bed.clone(),
    };
    Ok((
        CpuClient::new(device).with_parallelism(ParallelismConfig::new(Some(1), None)),
        left,
        right,
    ))
}

// Normalize H by sqrt(hDeep*hShallow) and the residual by sqrt(g*hDeep).
fn normalized_residual(z: f64, deep: f64, shallow: f64) -> f64 {
    let depth_ratio = shallow.sqrt() / deep.sqrt();
    let rarefaction = 2.0 * ((z * depth_ratio).sqrt() - 1.0);
    let shock = (z - depth_ratio) * (0.5 * (1.0 + depth_ratio / z)).sqrt();
    rarefaction + shock
}

pub(super) fn physical_upper(deep: f64, shallow: f64) -> f64 {
    let (mut lo, mut hi) = (2.0, 3.0);
    for _ in 0..80 {
        let middle = 0.5 * (lo + hi);
        if normalized_residual(middle, deep, shallow) >= 0.0 {
            hi = middle;
        } else {
            lo = middle;
        }
    }
    hi * deep.sqrt() * shallow.sqrt()
}
