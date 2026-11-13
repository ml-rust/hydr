use super::{
    chart_brackets::bracket_chart,
    paired_sources::PairedSources,
    wet_root_charts, wet_root_seeds,
    wet_root_state::{Locator, selected_value},
    wet_root_types::{WetRootDomain, WetRootStatus},
};
use crate::channel::{
    flow::surveyed::error::{HydraulicError, HydraulicResult, checked},
    section::{SupportedAreaQuery, cover_domain},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedUpperBracketAlgorithms, CertifiedUpperBracketStatus},
};

pub(super) fn local<R>(
    parent: &mut PairedSources<'_, R>,
    locator: &mut Locator<R>,
    side: usize,
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
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let c = parent.geometry.client();
    let owner = locator.pending_rows(c)?;
    if owner.numel() == 0 {
        return Ok(());
    }
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
    let selected = checked::<R, _>(c, "wet root local source selection", |a| {
        parent.select_sources(a, &parent_rows)
    })?;
    let candidate = wet_root_charts::local(parent, &selected, side)?;
    let Some(query) = candidate.query else {
        if candidate.owner.numel() != 0 {
            return Err(HydraulicError::Numerical {
                context:
                    "Wet root local chart has owners without support. Discard the locator result"
                        .into(),
            });
        }
        return Ok(());
    };
    let owner = c.index_select(&owner, 0, &candidate.owner)?;
    let seed = checked::<R, _>(c, "wet root source-local seed", |a| {
        let input = selected.velocity.input();
        let left = a.expansion_interval(&input.left_area)?;
        let right = a.expansion_interval(&input.right_area)?;
        let magnitude = c.maximum(left.lower(), right.lower())?;
        let magnitude = c.index_select(&magnitude, 0, &candidate.owner)?;
        let zero = Tensor::zeros(magnitude.shape(), a.dtype(), c.device())?;
        let trial = c.where_cond(
            &c.gt(&magnitude, &zero)?,
            &magnitude,
            query.query.parameter.upper(),
        )?;
        Ok(c.maximum(
            query.query.parameter.lower(),
            &c.minimum(&trial, query.query.parameter.upper())?,
        )?)
    })?;
    bracket(parent, locator, &owner, query, &seed)
}

pub(super) fn covered<R>(
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
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let c = parent.geometry.client();
    let pending = locator.pending_rows(c)?;
    if pending.numel() == 0 {
        return Ok(());
    }
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &pending)?;
    let selected = checked::<R, _>(c, "wet root cover source selection", |a| {
        parent.select_sources(a, &parent_rows)
    })?;
    let coverage = checked::<R, _>(c, "wet root complete domain cover", |a| {
        cover_domain(
            a,
            &selected.bands,
            &selected.boundaries,
            &selected.original_rows,
        )
    })?;
    let Some(pair) = coverage.pair else {
        if coverage.owner.numel() != 0 {
            return Err(HydraulicError::Numerical {
                context: "Wet root coverage has owners without charts. Discard the locator result"
                    .into(),
            });
        }
        return Ok(());
    };
    let owner = c.index_select(&pending, 0, &coverage.owner)?;
    let parent_rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
    let selected = checked::<R, _>(c, "wet root covered source selection", |a| {
        parent.select_sources(a, &parent_rows)
    })?;
    let f0 = checked::<R, _>(c, "wet root covered dry-limit selection", |a| {
        selected_value(a, &locator.f0, &owner)
    })?;
    let seed = wet_root_seeds::lower_seed(parent, &selected, &pair.lower, &f0)?;
    bracket(parent, locator, &owner, pair.lower, &seed)?;

    let remaining = c
        .nonzero(&c.index_select(&locator.pending, 0, &owner)?)?
        .flatten()?;
    if remaining.numel() == 0 {
        return Ok(());
    }
    let (query, owner, selected) = checked::<R, _>(c, "wet root cap fallback selection", |a| {
        let query = pair.cap.select(a, &remaining)?;
        let owner = c.index_select(&owner, 0, &remaining)?;
        let rows = c.index_select(&locator.output.invocation_rows, 0, &owner)?;
        let selected = parent.select_sources(a, &rows)?;
        Ok((query, owner, selected))
    })?;
    let seed = wet_root_seeds::cap_seed(parent, &selected, &query)?;
    bracket(parent, locator, &owner, query, &seed)
}

fn bracket<R>(
    parent: &mut PairedSources<'_, R>,
    locator: &mut Locator<R>,
    owner: &Tensor<R>,
    query: SupportedAreaQuery<R>,
    seed: &Tensor<R>,
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
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let c = parent.geometry.client();
    let needed = owner
        .numel()
        .checked_mul(2)
        .ok_or_else(|| HydraulicError::Numerical {
            context: "Wet root bracket admission overflows usize. Split the invocation".into(),
        })?;
    if needed > locator.remaining()? {
        locator.set_status(c, owner, WetRootStatus::WorkLimit)?;
        return Ok(());
    }
    let rows = c.index_select(&locator.output.invocation_rows, 0, owner)?;
    let result = bracket_chart(parent, &rows, &query, seed, locator.remaining()?)?;
    locator.add_points(result.result.callback_points())?;
    let limited_code = Tensor::from_slice(
        &[CertifiedUpperBracketStatus::WorkLimit as i64],
        &[],
        c.device(),
    )?;
    let limited = c
        .nonzero(&c.eq(result.result.status(), &limited_code)?)?
        .flatten()?;
    let limited = c.index_select(owner, 0, &limited)?;
    if limited.numel() != 0 {
        // A later supported chart can still locate these rows within the remaining budget.
        locator.output.status = c.index_put(
            &locator.output.status,
            0,
            &limited,
            &Tensor::full_scalar(
                limited.shape(),
                DType::I64,
                WetRootStatus::WorkLimit as i64 as f64,
                c.device(),
            )?,
        )?;
    }
    if result.bracket_owner.numel() == 0 {
        return Ok(());
    }
    let domain = checked::<R, _>(c, "wet root bracket evidence retention", |a| {
        let invocation_owner = c.index_select(owner, 0, &result.bracket_owner)?;
        let lower_residual = a.bounds(
            &c.index_select(
                &result.result.lower_residual().lower,
                0,
                &result.bracket_owner,
            )?,
            &c.index_select(
                &result.result.lower_residual().upper,
                0,
                &result.bracket_owner,
            )?,
        )?;
        let upper_residual = a.bounds(
            &c.index_select(
                &result.result.upper_residual().lower,
                0,
                &result.bracket_owner,
            )?,
            &c.index_select(
                &result.result.upper_residual().upper,
                0,
                &result.bracket_owner,
            )?,
        )?;
        Ok(WetRootDomain {
            invocation_owner,
            parent_rows: result.bracket_rows,
            query: result.query,
            lower_retained: a.expansion_from_interval(&lower_residual)?,
            upper_retained: a.expansion_from_interval(&upper_residual)?,
            lower_residual,
            upper_residual,
        })
    })?;
    locator.set_status(c, &domain.invocation_owner, WetRootStatus::Located)?;
    locator.output.domains.push(domain);
    Ok(())
}
