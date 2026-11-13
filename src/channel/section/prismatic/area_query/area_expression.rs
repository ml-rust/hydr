use super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::{AreaAnchor, AreaQuery, SupportedAreaQuery, admission};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

/// Callers check compiled models and owner rows before resolving an anchor.
pub(super) fn resolve_anchor<R>(
    a: &mut Arithmetic<'_, R>,
    query: &AreaQuery<R>,
    boundaries: &BoundaryAreas<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(match &query.anchor {
        AreaAnchor::Source { area, .. } => {
            a.expansion_bounds(area.high(), area.low(), area.abs_error())?
        }
        AreaAnchor::Zero => a.expansion_point(&Tensor::zeros(
            query.owners.shape(),
            DType::F64,
            a.client().device(),
        )?)?,
        AreaAnchor::BandStart(indices) => boundaries.boundary(a, indices)?,
        AreaAnchor::BandEnd(indices) => {
            let end = a.client().add_scalar(indices, 1.0)?;
            boundaries.boundary(a, &end)?
        }
        AreaAnchor::Cap => boundaries.boundary(a, boundaries.cap_indices())?,
    })
}

impl<R> SupportedAreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn retained_endpoint_area(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        boundaries: &BoundaryAreas<R>,
    ) -> HydraulicResult<CertifiedExpansion<R>> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_model(boundaries.same_context(&bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::matching_owners(a, bands, boundaries.owners())?;
        admission::reject(
            a,
            &a.client()
                .ne(self.query.parameter.lower(), self.query.parameter.upper())?,
            "retained endpoint area requires one physical endpoint. Select a supported physical endpoint",
        )?;
        let anchor = resolve_anchor(a, &self.query, boundaries)?;
        Ok(a.expansion_add(&anchor, &self.query.offsets.lower)?)
    }
}
