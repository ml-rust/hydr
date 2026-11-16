use super::super::{
    area_point_hydrostatic, area_point_input,
    context::Context,
    hydraulic_geometry::{GatheredBands, query_error},
};
use super::{admission, chart::SupportedAreaQuery, evaluation};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

/// Two-part selection value of the hydrostatic integral at the exact area point of `supported`.
///
/// `area` is the query's exact zero-radius area point.
/// The value comes from the same band model as `PrismaticAreaPointRecovery::hydrostatic_point_m3`.
/// Its radius covers model arithmetic only. Never use it as a certificate.
/// Rows outside the model lift the certified query integral. Dry rows hold exact zero.
pub(in crate::channel::section::prismatic) fn hydrostatic_point<R>(
    a: &mut Arithmetic<'_, R>,
    context: &Arc<Context<R>>,
    bands: &GatheredBands<R>,
    supported: &SupportedAreaQuery<R>,
    area: &CertifiedExpansion<R>,
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
    let c = a.client();
    // The area point must meet the supported query area.
    let bounds = a.expansion_interval(area)?;
    admission::reject_geometry(
        a,
        &supported.query,
        &c.maximum(
            &c.lt(bounds.upper(), supported.area.lower())?,
            &c.gt(bounds.lower(), supported.area.upper())?,
        )?,
        "hydrostatic area point is disjoint from its supported query. Pass the query's own area point",
    )?;
    let (state, coordinates) = evaluation::recover_parts(a, context, bands, supported)?;
    let inputs =
        area_point_input::capture(a, context, &bands.section_owner, area).map_err(query_error)?;
    area_point_hydrostatic::point(a, context, &inputs, &coordinates, &state.integral)
        .map_err(query_error)
}
