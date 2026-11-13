use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::common::arithmetic::CertifiedExpansion;

pub(crate) use super::source_velocity::SourceVelocity;

/// Captured exact conserved points, independent of caller storage.
pub(crate) struct CapturedPointInput<R: Runtime<DType = DType>> {
    pub(crate) owners: Tensor<R>,
    pub(crate) left_area: CertifiedExpansion<R>,
    pub(crate) left_discharge: CertifiedExpansion<R>,
    pub(crate) right_area: CertifiedExpansion<R>,
    pub(crate) right_discharge: CertifiedExpansion<R>,
    pub(crate) left_dry: Tensor<R>,
    pub(crate) right_dry: Tensor<R>,
}
