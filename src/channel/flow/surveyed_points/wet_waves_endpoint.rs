use super::{
    paired_sources::{PairedSources, SelectedRows},
    wave_curve_types::{CurveValue, scatter_value, unavailable},
    wet_star_types::WetStarDomain,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar},
        error::{Arithmetic, HydraulicResult},
    },
    section::PhysicalEndpoint,
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

pub(super) struct EndpointOffsets<R: Runtime<DType = DType>> {
    pub(super) actual: CurveValue<R>,
    pub(super) conditional: CurveValue<R>,
}

pub(super) fn offsets<R>(
    a: &mut Arithmetic<'_, R>,
    star: &WetStarDomain<R>,
    endpoint: usize,
    side: usize,
) -> HydraulicResult<EndpointOffsets<R>>
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
    let n = star.parent_rows.numel();
    let mut actual = unavailable(a, n)?;
    let mut conditional = unavailable(a, n)?;
    if let Some(local) = &star.endpoints[endpoint].waves.local[side] {
        actual = scatter_value(a, actual, &local.rows, &local.endpoint.actual_shock.value)?;
        conditional = scatter_value(
            a,
            conditional,
            &local.rows,
            &local.endpoint.conditional_shock.value,
        )?;
    }
    Ok(EndpointOffsets {
        actual,
        conditional,
    })
}

pub(super) fn sounds<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    selected: &SelectedRows<R>,
    star: &WetStarDomain<R>,
) -> HydraulicResult<[CurveValue<R>; 2]>
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
    let gravity = scalar(a, sources.evaluation.config.gravity_m_per_s2)?;
    let mut make = |endpoint| -> HydraulicResult<CurveValue<R>> {
        let query = star.query.endpoint(a, &selected.bands, endpoint)?;
        let state = sources
            .geometry
            .recover_query(a, &selected.bands, &query)?
            .state;
        let physical = gravity_depth_root(a, &gravity, &state.hydraulic_depth, None)?;
        Ok(CurveValue {
            retained: a.expansion_from_interval(&physical)?,
            physical,
            available: Tensor::ones(star.parent_rows.shape(), a.dtype(), a.client().device())?,
        })
    };
    Ok([
        make(PhysicalEndpoint::Lower)?,
        make(PhysicalEndpoint::Upper)?,
    ])
}

pub(super) fn select<R>(
    a: &mut Arithmetic<'_, R>,
    mask: &Tensor<R>,
    yes: &CertifiedExpansion<R>,
    no: &CertifiedExpansion<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    let c = a.client();
    Ok(a.expansion_bounds(
        &c.where_cond(mask, yes.high(), no.high())?,
        &c.where_cond(mask, yes.low(), no.low())?,
        &c.where_cond(mask, yes.abs_error(), no.abs_error())?,
    )?)
}
