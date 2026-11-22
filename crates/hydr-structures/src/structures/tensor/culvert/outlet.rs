//! Compact `[m]` outlet-control rows: length, roughness, entrance loss, slope, gravity and the
//! arc bands of the march.

use super::{Bounds, TensorBarrelContext};
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::numerics::culvert::outlet::{ArcShape, Outline};
use numr::dtype::DType;
use numr::ops::IndexingOps;
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
    /// Arc bands in units of the rise, `None` when no row is an arc barrel.
    pub(crate) arcs: Option<ArcShape<R>>,
}

impl<R: Runtime<DType = DType>> TensorOutletContext<R> {
    /// Gathers the rows at the I64 indices `rows`, in their order. Indices may repeat.
    pub(crate) fn select(&self, c: &R::Client, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
        let [lower, upper] = &self.beta;
        Ok(Self {
            length: take(&self.length)?,
            manning: take(&self.manning)?,
            entrance: take(&self.entrance)?,
            slope: take(&self.slope)?,
            gravity: take(&self.gravity)?,
            beta: [take(lower)?, take(upper)?],
            arcs: self
                .arcs
                .as_ref()
                .map(|arcs| arcs.select(c, rows))
                .transpose()?,
        })
    }

    /// The shape view of the rows of `barrel`, the barrel context these rows align with.
    pub(crate) fn outline<'s>(&'s self, barrel: &'s TensorBarrelContext<R>) -> Outline<'s, R> {
        Outline {
            circular: &barrel.circular,
            beta: &self.beta,
            arcs: self.arcs.as_ref(),
        }
    }
}
