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
    pub endpoints: Vec<usize>,
    pub root_iterations: Vec<usize>,
}

pub(in crate::channel::flow) struct ScalarBoundaryFrame<R: Runtime<DType = DType>> {
    pub longitudinal_water: Tensor<R>,
    pub water_underflow_bound: Tensor<R>,
    pub momentum_underflow_bound: Tensor<R>,
    pub traces: Arc<NativeBoundaryTraces<R>>,
}
