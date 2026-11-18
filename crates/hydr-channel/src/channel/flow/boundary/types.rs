use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use std::sync::Arc;

pub(in crate::channel::flow) struct EndpointInterior<R: Runtime<DType = DType>> {
    pub area: Tensor<R>,
    pub discharge: Tensor<R>,
    pub depth: Tensor<R>,
    pub velocity: Tensor<R>,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct NativeBoundaryTraces<R: Runtime<DType = DType>> {
    pub depth_m: Tensor<R>,
    pub area_m2: Tensor<R>,
    pub outward_velocity_m_per_s: Tensor<R>,
    pub outward_discharge_m3_per_s: Tensor<R>,
    pub momentum_flux_m4_per_s2: Tensor<R>,
    pub signal_speed_bound_m_per_s: Tensor<R>,
    pub velocity_compatibility_abs_bound_m_per_s: Tensor<R>,
    pub water_compatibility_abs_bound_m3_per_s: Tensor<R>,
    pub momentum_compatibility_abs_bound_m4_per_s2: Tensor<R>,
    pub water_flux_width_m3_per_s: Tensor<R>,
    pub momentum_flux_width_m4_per_s2: Tensor<R>,
    pub cap_compatibility_approximation: Tensor<R>,
    pub reservoir: Option<NativeReservoirTraces<R>>,
    pub endpoints: Vec<usize>,
    pub root_iterations: Vec<usize>,
}

/// Rates at every direct end face: scalar traces and flux ports.
///
/// Each tensor holds one entry per endpoint in `endpoints`, in that order.
pub(in crate::channel::flow) struct DirectBoundaryFrame<R: Runtime<DType = DType>> {
    /// Direct endpoints in increasing order: `0` left, `1` right.
    pub endpoints: Vec<usize>,
    /// Water rate along the increasing reach coordinate, in m³/s.
    pub longitudinal_water: Tensor<R>,
    /// Momentum flux along the increasing reach coordinate, including pressure, in m⁴/s².
    pub momentum_flux_m4_per_s2: Tensor<R>,
    /// Unsigned bound on the face signal speeds, in m/s.
    pub signal_speed_bound_m_per_s: Tensor<R>,
    pub water_underflow_bound: Tensor<R>,
    pub momentum_underflow_bound: Tensor<R>,
    /// Physical water flux bracket width. Zero at a flux port.
    pub water_flux_width_m3_per_s: Tensor<R>,
    /// Physical momentum flux bracket width. Zero at a flux port.
    pub momentum_flux_width_m4_per_s2: Tensor<R>,
    /// Largest trace root iteration count. Zero without a scalar trace.
    pub root_iterations: usize,
    /// Receipts of the scalar trace endpoints. `None` when every direct endpoint is a port.
    pub traces: Option<Arc<NativeBoundaryTraces<R>>>,
    /// Exterior dry-front states of the flux port endpoints.
    pub ports: Vec<PortFront<R>>,
}

/// Exterior dry-front state of one flux port endpoint.
pub(in crate::channel::flow) struct PortFront<R: Runtime<DType = DType>> {
    /// `0` left, `1` right.
    pub endpoint: usize,
    /// Into-reach invariant `u + φ` of the exterior water. `None` when the exterior is dry.
    pub invariant: Option<Tensor<R>>,
}

#[derive(Clone, Debug)]
pub(in crate::channel::flow) struct NativeReservoirTraces<R: Runtime<DType = DType>> {
    pub depth_lower_m: Tensor<R>,
    pub depth_upper_m: Tensor<R>,
    pub head_compatibility_abs_bound_m: Tensor<R>,
    pub dissipation_head_lower_m: Tensor<R>,
    pub dissipation_head_upper_m: Tensor<R>,
    pub endpoints: Vec<usize>,
}
