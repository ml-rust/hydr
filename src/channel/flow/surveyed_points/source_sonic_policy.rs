use super::{
    paired_sources::PairedSources,
    potential_range::{PotentialContext, PotentialEvaluation},
    source_sonic_numerical::SourceSonicNumericalEvidence,
    source_sonic_types::SonicSide,
    wave_curve_types::{CurveValue, scatter_value, unavailable},
};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{PhysicalEndpoint, SupportedAreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn residual<R>(
    sources: &mut PairedSources<'_, R>,
    parents: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
    side: SonicSide,
    numerical: &SourceSonicNumericalEvidence<R>,
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
    numerical.check_parents(sources, side, parents)?;
    if let [index] = numerical.policies() {
        let level = sources.panel_level(*index)?;
        return super::source_sonic_residual::endpoint_at_level(
            sources,
            parents,
            query,
            endpoint,
            side,
            numerical.epsilon(),
            &level,
        );
    }
    let c = sources.geometry.client();
    let mut result = checked(c, "mixed-policy sonic residual storage", |a| {
        unavailable(a, parents.numel())
    })?;
    for index in numerical.policies() {
        let rows = policy_rows::<R>(c, numerical.levels(), *index)?;
        if rows.numel() == 0 {
            continue;
        }
        let (parent, target, proof) =
            checked::<R, _>(c, "mixed-policy sonic residual operands", |a| {
                Ok((
                    c.index_select(parents, 0, &rows)?,
                    query.select(a, &rows)?,
                    numerical.selected(a, &rows)?,
                ))
            })?;
        let level = sources.panel_level(*index)?;
        let value = super::source_sonic_residual::endpoint_at_level(
            sources,
            &parent,
            &target,
            endpoint,
            side,
            proof.epsilon(),
            &level,
        )?;
        result = checked(c, "mixed-policy sonic residual completion", |a| {
            scatter_value(a, result, &rows, &value)
        })?;
    }
    Ok(result)
}

pub(super) fn invariant<R>(
    sources: &PairedSources<'_, R>,
    parents: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    side: SonicSide,
    numerical: &SourceSonicNumericalEvidence<R>,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
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
    numerical.check_parents(sources, side, parents)?;
    if let [index] = numerical.policies() {
        return invariant_at(sources, parents, query, side, numerical.epsilon(), *index);
    }
    let c = sources.geometry.client();
    let (mut velocity, mut change) = checked(c, "mixed-policy sonic invariant storage", |a| {
        Ok((
            unavailable(a, parents.numel())?,
            unavailable(a, parents.numel())?,
        ))
    })?;
    for index in numerical.policies() {
        let rows = policy_rows::<R>(c, numerical.levels(), *index)?;
        if rows.numel() == 0 {
            continue;
        }
        let (parent, target, proof) =
            checked::<R, _>(c, "mixed-policy sonic invariant operands", |a| {
                Ok((
                    c.index_select(parents, 0, &rows)?,
                    query.select(a, &rows)?,
                    numerical.selected(a, &rows)?,
                ))
            })?;
        let (u, du) = invariant_at(sources, &parent, &target, side, proof.epsilon(), *index)?;
        (velocity, change) = checked(c, "mixed-policy sonic invariant completion", |a| {
            let available = Tensor::ones(rows.shape(), a.dtype(), c.device())?;
            let u = CurveValue {
                physical: a.expansion_interval(&u)?,
                retained: u,
                available: available.clone(),
            };
            let du = CurveValue {
                physical: a.expansion_interval(&du)?,
                retained: du,
                available,
            };
            Ok((
                scatter_value(a, velocity, &rows, &u)?,
                scatter_value(a, change, &rows, &du)?,
            ))
        })?;
    }
    Ok((velocity.retained, change.retained))
}

fn invariant_at<R>(
    sources: &PairedSources<'_, R>,
    parents: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    side: SonicSide,
    epsilon: &CertifiedExpansion<R>,
    index: usize,
) -> HydraulicResult<(CertifiedExpansion<R>, CertifiedExpansion<R>)>
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
    let stage = sources.select_rows(parents, query)?;
    let level = sources.panel_level(index)?;
    let context = PotentialContext::new(
        sources.geometry,
        &stage.bands,
        &stage.sources[side.index()],
        query,
        &stage.boundaries,
        PotentialEvaluation {
            config: sources.evaluation.config,
            panels: level.panels(),
            work: sources.evaluation.work,
        },
    )?;
    super::source_sonic_invariant::endpoint(&context, side, epsilon)
}

pub(super) fn policy_rows<R>(
    c: &R::Client,
    levels: &Tensor<R>,
    index: usize,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R> + RuntimeClient<R>,
{
    let level = Tensor::full_scalar(&[], DType::I64, index as f64, c.device())?;
    Ok(c.nonzero(&c.eq(levels, &level)?)?.flatten()?)
}
