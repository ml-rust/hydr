use super::super::{
    SupportedAreaQuery, admission, endpoints::PhysicalEndpoint,
    source_difference::source_difference,
};

use super::{
    secant,
    types::{PressureEndpoint, PressureRange},
};
use crate::channel::section::prismatic::boundary_area::BoundaryAreas;
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::{GatheredBands, HydraulicGeometry, PrismaticAreaPointRecovery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Consume one supported physical endpoint without evaluating the opposite endpoint.
pub(crate) fn pressure_endpoint<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    endpoint: &SupportedAreaQuery<R>,
    source: &PrismaticAreaPointRecovery<R>,
    boundaries: &BoundaryAreas<R>,
    gravity: f64,
) -> HydraulicResult<PressureEndpoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    admission::reject(
        a,
        &a.client().ne(
            endpoint.query.parameter.lower(),
            endpoint.query.parameter.upper(),
        )?,
        "pressure endpoint requires a singleton physical query. Select one supported endpoint",
    )?;
    let delta = source_difference(a, bands, endpoint, source, boundaries)?;
    let target = geometry.recover_query(a, bands, endpoint)?;
    let k = secant::secant(a, geometry, bands, source, &target, &delta.lower, gravity)?;
    finish(a, delta.lower, k)
}
pub(crate) fn pressure_range<R>(
    a: &mut Arithmetic<'_, R>,
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    query: &SupportedAreaQuery<R>,
    source: &PrismaticAreaPointRecovery<R>,
    boundaries: &BoundaryAreas<R>,
    gravity: f64,
) -> HydraulicResult<PressureRange<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let delta = source_difference(a, bands, query, source, boundaries)?;
    let lower_query = query.endpoint(a, bands, PhysicalEndpoint::Lower)?;
    let upper_query = query.endpoint(a, bands, PhysicalEndpoint::Upper)?;
    let lower_geometry = geometry.recover_query(a, bands, &lower_query)?;
    let upper_geometry = geometry.recover_query(a, bands, &upper_query)?;
    let lower_k = secant::secant(
        a,
        geometry,
        bands,
        source,
        &lower_geometry,
        &delta.lower,
        gravity,
    )?;
    let upper_k = secant::secant(
        a,
        geometry,
        bands,
        source,
        &upper_geometry,
        &delta.upper,
        gravity,
    )?;
    let lower = finish(a, delta.lower, lower_k)?;
    let upper = finish(a, delta.upper, upper_k)?;
    Ok(PressureRange { lower, upper })
}

fn finish<R>(
    a: &mut Arithmetic<'_, R>,
    delta_area: solvr::common::arithmetic::CertifiedExpansion<R>,
    secant: solvr::common::arithmetic::Interval<R>,
) -> HydraulicResult<PressureEndpoint<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let factor = a.expansion_from_interval(&secant)?;
    let retained_increment = a.expansion_mul(&delta_area, &factor)?;
    let projected = a.expansion_interval(&retained_increment)?;
    let delta = a.expansion_interval(&delta_area)?;
    let direct = a.mul(&delta, &secant)?;
    let c = a.client();
    let zero = Tensor::zeros(delta_area.high().shape(), DType::F64, c.device())?;
    let nonnegative = c.ge(delta.lower(), &zero)?;
    let nonpositive = c.le(delta.upper(), &zero)?;
    // Monotone pressure fixes the physical sign without shrinking the raw arithmetic ball.
    let lower = c.maximum(projected.lower(), direct.lower())?;
    let upper = c.minimum(projected.upper(), direct.upper())?;
    let lower = c.where_cond(&nonnegative, &c.maximum(&lower, &zero)?, &lower)?;
    let upper = c.where_cond(&nonpositive, &c.minimum(&upper, &zero)?, &upper)?;
    let physical_increment = a.bounds(&lower, &upper)?;
    Ok(PressureEndpoint {
        delta_area,
        retained_increment,
        physical_increment,
        secant,
    })
}
