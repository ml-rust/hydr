//! Compact `[m]` inlet-control rows: coefficients, slope term, band limits and anchors.

use super::Bounds;
use numr::dtype::DType;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// One row per culvert inlet, aligned with the rows of its barrel context.
///
/// Every tensor field lives on the device. Admission certified on every row that the anchor at
/// `X = 4` lies above the anchor at `X = 3.5`, so the headwater rises across the transition.
pub(crate) struct TensorInletContext<R: Runtime<DType = DType>> {
    /// 1 on form-1 rows, 0 on form-2 rows.
    pub(crate) form_one: Tensor<R>,
    /// Coefficient `K`.
    pub(crate) k: Tensor<R>,
    /// Exponent `M`.
    pub(crate) m: Tensor<R>,
    /// Coefficient `c`.
    pub(crate) c: Tensor<R>,
    /// Coefficient `Y`.
    pub(crate) y: Tensor<R>,
    /// Gravity in metres per second squared.
    pub(crate) gravity: Tensor<R>,
    /// Blocked share `f` of the inlet face. The headwater at `Q` is the unblocked headwater at
    /// `Q / (1 - f)`.
    pub(crate) face: Tensor<R>,
    /// `true` when some row has a positive face blockage. Without one, a query runs at `Q`.
    pub(crate) faced: bool,
    /// `Ks S`, with `Ks = -0.5` or `+0.7` mitered and the signed barrel slope `S`.
    pub(crate) slope_term: Bounds<R>,
    /// `A D^0.5 / Ku` in cubic metres per second, so `X = Q / scale`.
    pub(crate) scale: Bounds<R>,
    /// Discharge `3.5 scale` at the unsubmerged limit.
    pub(crate) unsubmerged_discharge: Bounds<R>,
    /// Discharge `4 scale` at the submerged limit.
    pub(crate) submerged_discharge: Bounds<R>,
    /// Unsubmerged headwater at exactly `X = 3.5`, in metres.
    pub(crate) low: Bounds<R>,
    /// Submerged headwater at exactly `X = 4`, in metres.
    pub(crate) high: Bounds<R>,
}

impl<R: Runtime<DType = DType>> TensorInletContext<R> {
    /// Gathers the rows at the I64 indices `rows`, in their order. Indices may repeat.
    pub(crate) fn select(&self, c: &R::Client, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: IndexingOps<R>,
    {
        let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        Ok(Self {
            form_one: take(&self.form_one)?,
            k: take(&self.k)?,
            m: take(&self.m)?,
            c: take(&self.c)?,
            y: take(&self.y)?,
            gravity: take(&self.gravity)?,
            face: take(&self.face)?,
            faced: self.faced,
            slope_term: pair(&self.slope_term)?,
            scale: pair(&self.scale)?,
            unsubmerged_discharge: pair(&self.unsubmerged_discharge)?,
            submerged_discharge: pair(&self.submerged_discharge)?,
            low: pair(&self.low)?,
            high: pair(&self.high)?,
        })
    }
}
