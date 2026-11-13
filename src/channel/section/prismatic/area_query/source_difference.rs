use super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::{
    SupportedAreaQuery, admission, area_expression::resolve_anchor, chart::ExpandedBounds,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use crate::channel::section::PrismaticAreaPointRecovery;
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use std::sync::Arc;

/// Retain query-minus-source differences in physical area order.
/// Query, source, and boundary witnesses require the same compiled model.
pub(crate) fn source_difference<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    query: &SupportedAreaQuery<R>,
    source: &PrismaticAreaPointRecovery<R>,
    boundaries: &BoundaryAreas<R>,
) -> HydraulicResult<ExpandedBounds<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    admission::matching_model(Arc::ptr_eq(&query.query.model, &bands.model))?;
    admission::matching_model(source.same_context(&bands.model))?;
    admission::matching_model(boundaries.same_context(&bands.model))?;
    admission::matching_owners(a, bands, &query.query.owners)?;
    admission::matching_owners(a, bands, source.owners())?;
    admission::matching_owners(a, bands, boundaries.owners())?;
    let anchor = resolve_anchor(a, &query.query, boundaries)?;
    let base = a.expansion_sub(&anchor, source.requested_areas_m2())?;
    Ok(ExpandedBounds {
        lower: a.expansion_add(&base, &query.query.offsets.lower)?,
        upper: a.expansion_add(&base, &query.query.offsets.upper)?,
    })
}
