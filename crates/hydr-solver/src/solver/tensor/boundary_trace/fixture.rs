use super::{TensorBoundaryRequest, TensorBoundaryTrace, characteristic_trace};
use crate::contract::{Axis, BoundaryFace, EngineError, SolverConfig};
use crate::numerics::TensorFaceState;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

pub(super) type TestResult = Result<(), Box<dyn std::error::Error>>;
pub(super) const G: f64 = 9.81;

#[derive(Clone, Copy)]
pub(super) struct Lane {
    pub(super) h: f64,
    pub(super) u: f64,
    pub(super) v: f64,
    pub(super) stage: f64,
    pub(super) discharge: Option<f64>,
    pub(super) incoming: Option<f64>,
}

pub(super) fn run(lanes: &[Lane]) -> Result<TensorBoundaryTrace<CpuRuntime>, EngineError> {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    let make =
        |values: Vec<f64>| Tensor::<CpuRuntime>::from_slice(&values, &[lanes.len()], &device);
    let interior = TensorFaceState {
        h: make(lanes.iter().map(|lane| lane.h).collect())?,
        q_normal: make(lanes.iter().map(|lane| lane.h * lane.u).collect())?,
        q_tangent: make(lanes.iter().map(|lane| lane.h * lane.v).collect())?,
        bed: make(vec![0.0; lanes.len()])?,
    };
    let request = TensorBoundaryRequest {
        stage_depth: make(lanes.iter().map(|lane| lane.stage).collect())?,
        outward_discharge: make(
            lanes
                .iter()
                .map(|lane| lane.discharge.unwrap_or(0.0))
                .collect(),
        )?,
        discharge_mask: make(
            lanes
                .iter()
                .map(|lane| if lane.discharge.is_some() { 1.0 } else { 0.0 })
                .collect(),
        )?,
        inflow_tangent: make(
            lanes
                .iter()
                .map(|lane| lane.incoming.unwrap_or(0.0))
                .collect(),
        )?,
        has_inflow_tangent: make(
            lanes
                .iter()
                .map(|lane| if lane.incoming.is_some() { 1.0 } else { 0.0 })
                .collect(),
        )?,
    };
    let faces: Vec<_> = (0..lanes.len())
        .map(|row| BoundaryFace {
            axis: Axis::X,
            column: 0,
            row,
        })
        .collect();
    characteristic_trace(
        &client,
        &interior,
        &request,
        &faces,
        G,
        &SolverConfig::default(),
    )
}

pub(super) fn lane() -> Lane {
    Lane {
        h: 1.0,
        u: 0.5,
        v: 0.2,
        stage: 1.0,
        discharge: None,
        incoming: Some(-0.4),
    }
}

pub(super) fn velocity(interior: Lane, depth: f64) -> f64 {
    let wave = if depth <= interior.h {
        2.0 * ((G * depth).sqrt() - (G * interior.h).sqrt())
    } else {
        (depth - interior.h) * (0.5 * G * (1.0 / depth + 1.0 / interior.h)).sqrt()
    };
    interior.u - wave
}
