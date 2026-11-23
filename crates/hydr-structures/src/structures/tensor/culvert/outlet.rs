//! Compact `[m]` outlet-control rows: length, roughness, entrance loss, slope, gravity, the
//! shape terms of the march and the open sections it reads above each floor.

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
    /// Manning `n` in `s/m^(1/3)`, the wall `n` above a floor.
    pub(crate) manning: Tensor<R>,
    /// Enclosure of the face entrance loss `ke / (1 - f)^2 + (f / (1 - f))^2`. Both bounds are
    /// `ke` on rows with `f = 0`.
    pub(crate) entrance: Bounds<R>,
    /// Signed barrel slope `S`, positive when the barrel falls toward the outlet.
    pub(crate) slope: Tensor<R>,
    /// Gravity in metres per second squared.
    pub(crate) gravity: Tensor<R>,
    /// `B / D`, the span over the rise, with `D'` for `D` above a floor.
    pub(crate) beta: Bounds<R>,
    /// Bed term of the box perimeter in units of the rise: `rho^(3/2) B / D'` on a floored box,
    /// the same bounds as `beta` elsewhere.
    pub(crate) bed: Bounds<R>,
    /// Arc bands in units of the rise, the open bands above a floor. `None` when no row is an
    /// arc barrel or a floored circle.
    pub(crate) arcs: Option<ArcShape<R>>,
    /// The barrel rows as the march reads them: each floored row's open section, see
    /// [`TensorBarrelContext::march_view`]. `None` when no row has a floor.
    pub(crate) open: Option<Box<TensorBarrelContext<R>>>,
}

impl<R: Runtime<DType = DType>> TensorOutletContext<R> {
    /// Gathers the rows at the I64 indices `rows`, in their order. Indices may repeat.
    pub(crate) fn select(&self, c: &R::Client, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        Ok(Self {
            length: take(&self.length)?,
            manning: take(&self.manning)?,
            entrance: pair(&self.entrance)?,
            slope: take(&self.slope)?,
            gravity: take(&self.gravity)?,
            beta: pair(&self.beta)?,
            bed: pair(&self.bed)?,
            arcs: self
                .arcs
                .as_ref()
                .map(|arcs| arcs.select(c, rows))
                .transpose()?,
            open: self
                .open
                .as_ref()
                .map(|open| open.select(rows).map(Box::new))
                .transpose()?,
        })
    }

    /// The rows of `barrel` as the march reads them: the open sections above its floors, or
    /// `barrel` itself without a floor. `barrel` is the context these rows align with.
    pub(crate) fn view<'s>(
        &'s self,
        barrel: &'s TensorBarrelContext<R>,
    ) -> &'s TensorBarrelContext<R> {
        self.open.as_deref().unwrap_or(barrel)
    }

    /// The shape view of the rows of `barrel`, the march view these rows align with.
    pub(crate) fn outline<'s>(&'s self, barrel: &'s TensorBarrelContext<R>) -> Outline<'s, R> {
        Outline {
            circular: &barrel.circular,
            beta: &self.beta,
            bed: &self.bed,
            arcs: self.arcs.as_ref(),
        }
    }
}
