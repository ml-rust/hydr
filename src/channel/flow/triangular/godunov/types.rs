use super::super::{faces::Face, wave::SideBounds};
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

pub(in crate::channel::flow::triangular) struct ValueBounds<R: Runtime<DType = DType>> {
    pub nominal: Tensor<R>,
    pub lower: Tensor<R>,
    pub upper: Tensor<R>,
}

pub(in crate::channel::flow::triangular) struct StarBracket<R: Runtime<DType = DType>> {
    pub depth: Tensor<R>,
    pub depth_lower: Tensor<R>,
    pub depth_upper: Tensor<R>,
    pub velocity: Tensor<R>,
    pub velocity_lower: Tensor<R>,
    pub velocity_upper: Tensor<R>,
    pub sound_lower: Tensor<R>,
    pub sound_upper: Tensor<R>,
    pub left_shock: Tensor<R>,
    pub right_shock: Tensor<R>,
    pub left_shock_lower: Tensor<R>,
    pub left_shock_upper: Tensor<R>,
    pub right_shock_lower: Tensor<R>,
    pub right_shock_upper: Tensor<R>,
    pub iterations: usize,
}

pub(in crate::channel::flow::triangular) struct RiemannInput<'a, R: Runtime<DType = DType>> {
    pub left: &'a Face<R>,
    pub right: &'a Face<R>,
    pub left_bounds: &'a SideBounds<R>,
    pub right_bounds: &'a SideBounds<R>,
    pub lower_signal: &'a Tensor<R>,
    pub upper_signal: &'a Tensor<R>,
}
