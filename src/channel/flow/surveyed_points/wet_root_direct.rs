use super::{
    paired_sources::{PairedSources, SelectedRows},
    wet_root_charts,
    wet_root_state::{Locator, direct_domain, exact_zero, selected_value},
    wet_root_types::{WetRootEvidence, WetRootEvidenceKind, WetRootStatus},
};
use crate::channel::{
    flow::surveyed::{
        error::{HydraulicError, HydraulicResult, checked},
        work::QueryWork,
    },
    section::PhysicalEndpoint,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(super) fn singleton<R>(
    parent: &mut PairedSources<'_, R>,
    locator: &mut Locator<R>,
    initial: &SelectedRows<R>,
    cap: bool,
) -> HydraulicResult<()>
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
    let c = parent.geometry.client();
    let owner = if cap {
        locator.pending_rows(c)?
    } else {
        c.nonzero(&c.mul(&locator.pending, &initial.velocity.uniform)?)?
            .flatten()?
    };
    if owner.numel() == 0 {
        return Ok(());
    }
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
    let selected = checked::<R, _>(c, "wet root direct source selection", |a| {
        parent.select_sources(a, &parent_rows)
    })?;
    let candidate = wet_root_charts::singleton(parent, &selected, cap)?;
    let Some(query) = candidate.query else {
        if candidate.owner.numel() != 0 {
            return Err(HydraulicError::Numerical {
                context:
                    "Wet root singleton has owners without support. Discard the locator result"
                        .into(),
            });
        }
        return Ok(());
    };
    let owner = c.index_select(&owner, 0, &candidate.owner)?;
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
    if !admit(parent, locator, &owner)? {
        return Ok(());
    }
    let value = parent.endpoint_on_rows(&parent_rows, &query, PhysicalEndpoint::Lower)?;
    let (hit_owner, domain, exceeded) =
        checked::<R, _>(c, "wet root direct classification", |a| {
            let hits = c.nonzero(&exact_zero(a, &value)?)?.flatten()?;
            let hit_owner = c.index_select(&owner, 0, &hits)?;
            let domain = if hits.numel() == 0 {
                None
            } else {
                let rows = c.index_select(&parent_rows, 0, &hits)?;
                let query = query.select(a, &hits)?;
                let exact = selected_value(a, &value, &hits)?;
                Some(direct_domain(a, hit_owner.clone(), rows, query, &exact)?)
            };
            let exceeded = if cap {
                let f0 = selected_value(a, &locator.f0, &owner)?;
                let zero = Tensor::zeros(owner.shape(), a.dtype(), c.device())?;
                let negative = c.mul(&value.available, &c.lt(value.physical.upper(), &zero)?)?;
                let negative_f0 = c.mul(&f0.available, &c.lt(f0.physical.upper(), &zero)?)?;
                let rows = c.nonzero(&c.mul(&negative, &negative_f0)?)?.flatten()?;
                c.index_select(&owner, 0, &rows)?
            } else {
                Tensor::empty(&[0], DType::I64, c.device())?
            };
            Ok((hit_owner, domain, exceeded))
        })?;
    if let Some(domain) = domain {
        locator.output.domains.push(domain);
    }
    locator.set_status(c, &hit_owner, WetRootStatus::Located)?;
    locator.set_status(c, &exceeded, WetRootStatus::CapExceeded)?;
    locator.output.evidence.push(WetRootEvidence {
        invocation_owner: owner,
        parent_rows,
        kind: if cap {
            WetRootEvidenceKind::Cap
        } else {
            WetRootEvidenceKind::Uniform
        },
        value,
    });
    Ok(())
}

pub(super) fn dry_limit<R>(
    parent: &mut PairedSources<'_, R>,
    locator: &mut Locator<R>,
) -> HydraulicResult<()>
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
    let c = parent.geometry.client();
    let owner = locator.pending_rows(c)?;
    if owner.numel() == 0 || !admit(parent, locator, &owner)? {
        return Ok(());
    }
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
    let invariant = parent.source_invariant_on_rows(&parent_rows)?;
    let vacuum = c.nonzero(&invariant.vacuum)?.flatten()?;
    let touching = c.nonzero(&invariant.touching)?.flatten()?;
    locator.set_status(
        c,
        &c.index_select(&owner, 0, &vacuum)?,
        WetRootStatus::Vacuum,
    )?;
    locator.set_status(
        c,
        &c.index_select(&owner, 0, &touching)?,
        WetRootStatus::TouchingDry,
    )?;
    checked::<R, _>(c, "wet root dry-limit retention", |a| {
        locator.retain_f0(a, &owner, &invariant.value)
    })?;
    locator.output.evidence.push(WetRootEvidence {
        invocation_owner: owner,
        parent_rows,
        kind: WetRootEvidenceKind::DryLimit,
        value: invariant.value,
    });
    Ok(())
}

fn admit<R>(
    parent: &PairedSources<'_, R>,
    locator: &mut Locator<R>,
    owner: &Tensor<R>,
) -> HydraulicResult<bool>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if owner.numel() > locator.remaining()? {
        locator.set_status(parent.geometry.client(), owner, WetRootStatus::WorkLimit)?;
        return Ok(false);
    }
    parent
        .evaluation
        .work
        .charge(QueryWork::Residual, owner.numel())?;
    locator.add_points(owner.numel())?;
    Ok(true)
}
