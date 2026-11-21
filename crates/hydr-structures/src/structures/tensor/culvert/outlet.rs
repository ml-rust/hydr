//! Compact `[m]` outlet-control rows: length, roughness, entrance loss, slope and gravity.

use super::Bounds;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One row per culvert outlet, aligned with the rows of its barrel context.
pub(crate) struct TensorOutletContext<R: Runtime<DType = DType>> {
    /// Barrel length `L` in metres.
    pub(crate) length: Tensor<R>,
    /// Manning `n` in `s/m^(1/3)`.
    pub(crate) manning: Tensor<R>,
    /// Entrance loss coefficient `ke`.
    pub(crate) entrance: Tensor<R>,
    /// Signed barrel slope `S`, positive when the barrel falls toward the outlet.
    pub(crate) slope: Tensor<R>,
    /// Gravity in metres per second squared.
    pub(crate) gravity: Tensor<R>,
    /// `B / D`, the span over the rise.
    pub(crate) beta: Bounds<R>,
}
