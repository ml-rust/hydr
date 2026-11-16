use super::source_sonic_types::SonicSide;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum SourceSonicQuadratureOutcome {
    Advanced = 0,
    LevelLimit = 1,
    WorkLimit = 2,
    NotApplicable = 3,
    Unresolved = 4,
}

#[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
pub(super) struct SourceSonicQuadratureProgress<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) outcome: Tensor<R>,
    pub(super) old_level: Tensor<R>,
    pub(super) new_level: Tensor<R>,
    pub(super) bracket_improved: Tensor<R>,
    pub(super) additional_residual_points: usize,
    pub(super) additional_simpson_value_points: usize,
    pub(super) additional_simpson_derivative_points: usize,
    pub(super) additional_mapped_points: usize,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
