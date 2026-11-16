#[cfg(test)]
pub(super) use super::wet_root_types::WetRootDomains;
use super::{
    chart_admission, paired_sources::PairedSources, wet_root_direct, wet_root_search,
    wet_root_state::Locator,
};
use super::{root_invocation::RootInvocation, wet_root_types::WetRootResults};
#[cfg(test)]
use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedUpperBracketAlgorithms};

/// Locate complete wet/wet root domains without selecting centers or accepting hydraulic fluxes.
#[cfg(test)]
pub(super) fn locate_wet_roots<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    max_points: usize,
) -> HydraulicResult<WetRootDomains<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let config = sources.evaluation.config;
    if max_points > config.root.max_points || (parent_rows.numel() != 0 && max_points == 0) {
        return Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: format!(
                "Wet root max_points={max_points}, rows={}, configured ceiling={}. Supply a positive nonempty invocation budget within the ceiling",
                parent_rows.numel(),
                config.root.max_points,
            ),
        });
    }
    let mut invocation = RootInvocation::new(sources.root_witness()?, max_points)?;
    let result = locate_on_invocation(sources, parent_rows, &mut invocation)?;
    Ok(WetRootDomains {
        invocation,
        invocation_rows: result.invocation_rows,
        status: result.status,
        domains: result.domains,
        evidence: result.evidence,
    })
}

pub(super) fn locate_on_invocation<R>(
    sources: &mut PairedSources<'_, R>,
    parent_rows: &Tensor<R>,
    invocation: &mut RootInvocation,
) -> HydraulicResult<WetRootResults<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    sources.check_root_witness(invocation.source_witness())?;
    let c = sources.geometry.client();
    let selected = checked::<R, _>(c, "wet root invocation admission", |a| {
        let selected = sources.select_sources(a, parent_rows)?;
        let zero = Tensor::zeros(&[], a.dtype(), c.device())?;
        if c.nonzero(&c.le(&selected.velocity.both_wet, &zero)?)?
            .numel()
            != 0
        {
            return Err(chart_admission::unresolved(
                "Wet root locator requires two wet sources. Classify dry sources before locating domains",
            ));
        }
        Ok(selected)
    })?;
    let rows = checked::<R, _>(c, "wet root invocation ownership", |a| {
        let owned = Tensor::empty(selected.rows.shape(), DType::I64, a.client().device())?;
        a.client().copy_into(&owned, &selected.rows)?;
        Ok(owned)
    })?;
    let mut locator = Locator::new(c, rows, invocation)?;
    wet_root_direct::singleton(sources, &mut locator, &selected, false)?;
    wet_root_direct::dry_limit(sources, &mut locator)?;
    wet_root_direct::singleton(sources, &mut locator, &selected, true)?;
    wet_root_search::local(sources, &mut locator, 0)?;
    wet_root_search::local(sources, &mut locator, 1)?;
    wet_root_search::covered(sources, &mut locator)?;
    Ok(locator.output)
}

#[cfg(test)]
#[path = "wet_root_tests.rs"]
mod tests;
