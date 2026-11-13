use super::{
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SourceSonicBinding,
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult, checked},
    section::AreaDirection,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn owners<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    status: &Tensor<R>,
    not_requested: i64,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
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
    if owners.ndim() != 1 || owners.dtype() != DType::I64 || !owners.device().is_same(c.device()) {
        return Err(invalid(
            "request owners need a one-dimensional I64 tensor on the caller device",
        ));
    }
    let zero = Tensor::zeros(&[], DType::I64, c.device())?;
    let count = Tensor::full_scalar(
        &[],
        DType::I64,
        session.invocation_rows.numel() as f64,
        c.device(),
    )?;
    reject(
        a,
        &c.maximum(&c.lt(owners, &zero)?, &c.ge(owners, &count)?)?,
        "request owner lies outside the captured invocation",
    )?;
    let sorted = c.sort(owners, 0, false)?;
    if owners.numel() > 1 {
        let n = owners.numel() - 1;
        reject(
            a,
            &c.eq(&sorted.narrow(0, 0, n)?, &sorted.narrow(0, 1, n)?)?,
            "request contains duplicate invocation owners",
        )?;
    }
    let selected = c.index_select(status, 0, owners)?;
    let not_requested = Tensor::full_scalar(&[], DType::I64, not_requested as f64, c.device())?;
    reject(
        a,
        &c.ne(&selected, &not_requested)?,
        "request repeats an already-started invocation owner and side",
    )?;
    let captured = Tensor::empty(owners.shape(), DType::I64, c.device())?;
    c.copy_into(&captured, owners)?;
    let rows = c.index_select(&session.invocation_rows, 0, &captured)?;
    Ok((captured, rows))
}

pub(super) fn check<R>(session: &PointRootSession<'_, '_, R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    // Host identity checks precede every tensor selection and physical mapping.
    let mut reported_points = 0usize;
    for request in &session.sonic.requests {
        reported_points = reported_points
            .checked_add(request.endpoint_points)
            .and_then(|n| n.checked_add(request.bracket_points))
            .ok_or_else(|| invalid("source sonic point diagnostics overflow usize"))?;
        if request.bracket_expansions
            > session
                .invocation
                .source_witness()
                .config()
                .max_bracket_expansions
        {
            return Err(invalid(
                "source sonic bracket rounds exceed the original ceiling",
            ));
        }
        request
            .binding
            .check(&session.invocation, request.binding.side, &request.binding)?;
        for domain in &request.domains {
            if domain.iterations
                > session
                    .invocation
                    .source_witness()
                    .config()
                    .root
                    .max_iterations
            {
                return Err(invalid(
                    "source sonic midpoint rounds exceed the original ceiling",
                ));
            }
            reported_points = reported_points
                .checked_add(domain.callback_points)
                .ok_or_else(|| invalid("source sonic native point diagnostics overflow usize"))?;
            if domain.refinement.is_none()
                && (domain.iterations != 0
                    || domain.callback_points != 0
                    || domain.mapped_points != 0)
            {
                return Err(invalid(
                    "source sonic continuation is absent after native work",
                ));
            }
            request
                .binding
                .check(&session.invocation, domain.side, &domain.binding)?;
            if let Some(state) = &domain.refinement {
                request
                    .binding
                    .check(&session.invocation, domain.side, &state.binding)?;
            }
        }
    }
    if reported_points > session.callback_points() {
        return Err(invalid(
            "source sonic point diagnostics exceed the original consumed allowance",
        ));
    }
    checked::<R, _>(
        session.sources.geometry.client(),
        "source sonic bound invocation rows",
        |a| {
            let c = a.client();
            for request in &session.sonic.requests {
                let expected =
                    c.index_select(&session.invocation_rows, 0, &request.binding.owners)?;
                equal(a, &expected, &request.binding.rows)?;
                for domain in &request.domains {
                    rows(
                        a,
                        &domain.binding,
                        &domain.invocation_owner,
                        &domain.parent_rows,
                    )?;
                    let selected = session
                        .sources
                        .select_rows(&domain.parent_rows, &domain.query)?;
                    let source = selected.sources[domain.side.index()].requested_areas_m2();
                    for (actual, expected) in [
                        (domain.query.query.scale.high(), source.high()),
                        (domain.query.query.scale.low(), source.low()),
                        (domain.query.query.scale.abs_error(), source.abs_error()),
                    ] {
                        equal(a, actual, expected)?;
                    }
                    if !matches!(domain.query.query.direction, AreaDirection::Add) {
                        return Err(invalid("source sonic chart direction changed"));
                    }
                    let zero = Tensor::zeros(domain.parent_rows.shape(), a.dtype(), c.device())?;
                    let parameter = a.point(&zero)?;
                    let origin =
                        domain
                            .query
                            .with_parameter_bounds(a, &selected.bands, &parameter)?;
                    let origin =
                        origin.retained_endpoint_area(a, &selected.bands, &selected.boundaries)?;
                    for value in [origin.high(), origin.low(), origin.abs_error()] {
                        equal(a, value, &zero)?;
                    }
                    if let Some(state) = &domain.refinement {
                        rows(
                            a,
                            &state.binding,
                            &domain.invocation_owner,
                            &domain.parent_rows,
                        )?;
                        session
                            .sources
                            .select_rows(&domain.parent_rows, &state.chart)?;
                    }
                }
            }
            Ok(())
        },
    )
}

fn rows<R>(
    a: &mut Arithmetic<'_, R>,
    binding: &SourceSonicBinding<R>,
    owners: &Tensor<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    equal(a, &binding.owners, owners)?;
    equal(a, &binding.rows, rows)
}
fn equal<R>(a: &mut Arithmetic<'_, R>, first: &Tensor<R>, second: &Tensor<R>) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if first.shape() != second.shape() || first.dtype() != second.dtype() {
        return Err(invalid("source sonic bound row metadata changed"));
    }
    reject(
        a,
        &a.client().ne(first, second)?,
        "source sonic side, row, or chart binding changed",
    )
}
fn reject<R>(a: &Arithmetic<'_, R>, mask: &Tensor<R>, message: &str) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if a.client().nonzero(mask)?.numel() != 0 {
        return Err(invalid(message));
    }
    Ok(())
}
