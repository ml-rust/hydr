use super::source_sonic_types::SonicSide;
use numr::{dtype::DType, runtime::Runtime, tensor::Tensor};
use solvr::optimize::CertifiedRootTolerance;

/// The tightest admissible depth tolerance. Bisection then stops only at its arithmetic floor,
/// at exact width, or at a work ceiling.
pub(super) const TIGHTEST_DEPTH: CertifiedRootTolerance =
    CertifiedRootTolerance { atol: 0., rtol: 0. };

/// Per-row result of one explicit quadrature continuation call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum PointQuadratureOutcome {
    Advanced = 0,
    LevelLimit = 1,
    WorkLimit = 2,
    NotApplicable = 3,
    Unresolved = 4,
}

/// Rows follow the requested owners. `side` names the sonic side, and wet star calls hold `None`.
pub(super) struct PointQuadratureProgress<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: Option<SonicSide>,
    pub(super) outcome: Tensor<R>,
    pub(super) old_level: Tensor<R>,
    pub(super) new_level: Tensor<R>,
    pub(super) bracket_improved: Tensor<R>,
    pub(super) additional_residual_points: usize,
    pub(super) additional_simpson_value_points: usize,
    pub(super) additional_simpson_derivative_points: usize,
    pub(super) additional_mapped_points: usize,
    pub(super) callback_points: usize,
    #[cfg_attr(not(test), expect(dead_code, reason = "tests read it"))]
    pub(super) point_limit: usize,
}
