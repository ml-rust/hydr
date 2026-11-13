use super::super::context::Context;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

pub(crate) struct BoundaryAreaTable<R: Runtime<DType = DType>> {
    pub(super) model: Arc<Context<R>>,
    pub(super) areas: CertifiedExpansion<R>,
    pub(super) cap_indices: Tensor<R>,
}

pub(crate) struct BoundaryAreas<R: Runtime<DType = DType>> {
    pub(super) model: Arc<Context<R>>,
    pub(super) owners: Tensor<R>,
    pub(super) areas: CertifiedExpansion<R>,
    pub(super) cap_indices: Tensor<R>,
}
