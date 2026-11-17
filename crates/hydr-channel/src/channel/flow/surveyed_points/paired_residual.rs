use super::paired_sources::PairedSources;
use super::source_binding::SourcePanelLevel;
use super::wave_curve_types::{CurveValue, scatter_value, unavailable};
use super::wet_star_levels::WetStarLevels;
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

/// Positions index `levels` for each row. Rows evaluate in groups of one panel level.
pub(super) fn endpoint_at_levels<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
    levels: &WetStarLevels<R>,
    positions: &Tensor<R>,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    if let [index] = levels.policies() {
        let level = parent.panel_level(*index)?;
        return endpoint_at_level(parent, rows, query, endpoint, &level);
    }
    let c = parent.geometry.client();
    let local = levels.selected(c, positions)?;
    let mut result = checked(c, "mixed-level wet residual storage", |a| {
        unavailable(a, rows.numel())
    })?;
    for index in local.policies() {
        let group = local.rows_at(c, *index)?;
        if group.numel() == 0 {
            continue;
        }
        let (parents, target) = checked::<R, _>(c, "mixed-level wet residual operands", |a| {
            Ok((c.index_select(rows, 0, &group)?, query.select(a, &group)?))
        })?;
        let level = parent.panel_level(*index)?;
        let value = endpoint_at_level(parent, &parents, &target, endpoint, &level)?;
        result = checked(c, "mixed-level wet residual completion", |a| {
            scatter_value(a, result, &group, &value)
        })?;
    }
    Ok(result)
}

pub(super) fn endpoint_at_level<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
    level: &SourcePanelLevel,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let terms = super::paired_endpoint::prepare(
        parent,
        rows,
        query,
        endpoint,
        super::paired_endpoint::EndpointEvidenceMode::Discard,
        level,
    )?;
    let super::paired_endpoint::EndpointTerms {
        stage,
        far,
        local,
        far_sources,
        target_value,
        ..
    } = terms;
    let c = parent.geometry.client();
    let n = stage.original_rows.numel();
    checked(c, "paired cancellation before target potentials", |a| {
        let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
        let retained_du = &stage.velocity.difference;
        let mut retained = a.expansion_bounds(
            &c.where_cond(&stage.velocity.both_wet, retained_du.high(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.low(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.abs_error(), &zero)?,
        )?;
        for value in &far_sources {
            retained = a.expansion_sub(&retained, &value.retained)?;
        }
        for value in &local {
            retained = a.expansion_add(&retained, &value.retained)?;
        }
        for mask in &far {
            let term = a.expansion_bounds(
                &c.where_cond(mask, target_value.retained.high(), &zero)?,
                &c.where_cond(mask, target_value.retained.low(), &zero)?,
                &c.where_cond(mask, target_value.retained.abs_error(), &zero)?,
            )?;
            retained = a.expansion_add(&retained, &term)?;
        }
        let physical = a.expansion_interval(&retained)?;
        Ok(CurveValue {
            retained,
            physical,
            available: stage.velocity.both_wet,
        })
    })
}
