use super::algebra::Bounds;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

pub(super) struct Depth<R: Runtime<DType = DType>> {
    pub value: Tensor<R>,
    pub bounds: Bounds<R>,
    pub approximate: bool,
    pub iterations: usize,
}

pub(super) enum DepthRequest<R: Runtime<DType = DType>> {
    Known(Box<Depth<R>>),
    Discharge(Box<super::discharge::RootInput<R>>),
    Reservoir(Box<super::reservoir::RootInput<R>>),
}
