use super::{
    point_quadrature_types::{PointQuadratureOutcome, PointQuadratureProgress},
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::{
    error::{Arithmetic, HydraulicResult, checked},
    work::PrismaticRiemannWork,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

/// Admitted owners start as `NotApplicable` at level -1 and charge no work.
pub(super) fn start<R>(
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    side: Option<SonicSide>,
    context: &'static str,
) -> HydraulicResult<PointQuadratureProgress<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = session.sources.geometry.client();
    checked::<R, _>(c, context, |a| {
        let (owners, parents) = super::dry_vacuum_evidence::owners(a, session, owners)?;
        let count = owners.numel();
        Ok(PointQuadratureProgress {
            invocation_owner: owners,
            parent_rows: parents,
            side,
            outcome: Tensor::full_scalar(
                &[count],
                DType::I64,
                PointQuadratureOutcome::NotApplicable as i64 as f64,
                c.device(),
            )?,
            old_level: Tensor::full_scalar(&[count], DType::I64, -1., c.device())?,
            new_level: Tensor::full_scalar(&[count], DType::I64, -1., c.device())?,
            bracket_improved: Tensor::zeros(&[count], a.dtype(), c.device())?,
            additional_residual_points: 0,
            additional_simpson_value_points: 0,
            additional_simpson_derivative_points: 0,
            additional_mapped_points: 0,
            callback_points: session.callback_points(),
            point_limit: session.point_limit(),
        })
    })
}

/// Invocation-row demand mask and the position of each invocation row in the progress.
pub(super) fn demand<R>(
    session: &PointRootSession<'_, '_, R>,
    progress: &PointQuadratureProgress<R>,
    context: &'static str,
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
    let c = session.sources.geometry.client();
    checked::<R, _>(c, context, |a| {
        let owners = &progress.invocation_owner;
        let demand = c.index_put(
            &Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?,
            0,
            owners,
            &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
        )?;
        let association =
            super::point_flux_association::inverse(a, session.invocation_rows.numel(), owners)?;
        Ok((demand, association))
    })
}

/// Writes the ledger deltas and the final invocation count into the progress.
pub(super) fn finish<R: Runtime<DType = DType>>(
    progress: &mut PointQuadratureProgress<R>,
    before: PrismaticRiemannWork,
    after: PrismaticRiemannWork,
    mapped: [usize; 2],
    callback_points: usize,
) -> HydraulicResult<()> {
    progress.additional_residual_points = delta(before.residual_points, after.residual_points)?;
    progress.additional_simpson_value_points =
        delta(before.simpson_value_points, after.simpson_value_points)?;
    progress.additional_simpson_derivative_points = delta(
        before.simpson_derivative_points,
        after.simpson_derivative_points,
    )?;
    progress.additional_mapped_points = delta(mapped[0], mapped[1])?;
    progress.callback_points = callback_points;
    Ok(())
}

fn delta(before: usize, after: usize) -> HydraulicResult<usize> {
    after
        .checked_sub(before)
        .ok_or_else(|| invalid("quadrature cumulative work decreased"))
}

/// Per-row values of one committed or unchanged group, in invocation-owner order.
pub(super) struct Record<'a, R: Runtime<DType = DType>> {
    pub(super) owners: &'a Tensor<R>,
    pub(super) old: &'a Tensor<R>,
    pub(super) new: &'a Tensor<R>,
    pub(super) outcome: &'a Tensor<R>,
    pub(super) improved: &'a Tensor<R>,
}

pub(super) fn record<R>(
    a: &Arithmetic<'_, R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
    value: Record<'_, R>,
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
    let c = a.client();
    let positions = super::point_flux_association::positions(a, association, value.owners)?;
    progress.old_level = c.index_put(&progress.old_level, 0, &positions, value.old)?;
    progress.new_level = c.index_put(&progress.new_level, 0, &positions, value.new)?;
    progress.outcome = c.index_put(&progress.outcome, 0, &positions, value.outcome)?;
    progress.bracket_improved =
        c.index_put(&progress.bracket_improved, 0, &positions, value.improved)?;
    Ok(())
}

/// Records rows whose level and bracket stay unchanged.
pub(super) fn mark<R>(
    c: &R::Client,
    owners: &Tensor<R>,
    levels: &Tensor<R>,
    outcome: PointQuadratureOutcome,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
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
    checked::<R, _>(c, "quadrature unchanged progress", |a| {
        let outcome = code::<R>(c, owners, outcome)?;
        let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
        record(
            a,
            association,
            progress,
            Record {
                owners,
                old: levels,
                new: levels,
                outcome: &outcome,
                improved: &zero,
            },
        )
    })
}

pub(super) fn code<R: Runtime<DType = DType>>(
    c: &R::Client,
    rows: &Tensor<R>,
    outcome: PointQuadratureOutcome,
) -> HydraulicResult<Tensor<R>> {
    Ok(Tensor::full_scalar(
        rows.shape(),
        DType::I64,
        outcome as i64 as f64,
        c.device(),
    )?)
}

/// Strict nesting of the selected new bracket inside the old one.
pub(super) fn improved<R>(
    c: &R::Client,
    old: [&Tensor<R>; 2],
    new: [&Tensor<R>; 2],
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    let nested = c.mul(&c.ge(new[0], old[0])?, &c.le(new[1], old[1])?)?;
    let strict = c.maximum(&c.gt(new[0], old[0])?, &c.lt(new[1], old[1])?)?;
    Ok(c.mul(&nested, &strict)?)
}

/// Maps each committed row's depth status to its quadrature outcome.
pub(super) fn committed<R>(
    c: &R::Client,
    rows: &Tensor<R>,
    status: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + CompareOps<R>,
{
    use super::wet_depth_types::WetDepthStatus;
    let work = Tensor::full_scalar(
        &[],
        DType::I64,
        WetDepthStatus::WorkLimit as i64 as f64,
        c.device(),
    )?;
    let unresolved = Tensor::full_scalar(
        &[],
        DType::I64,
        WetDepthStatus::Unresolved as i64 as f64,
        c.device(),
    )?;
    Ok(c.where_cond(
        &c.eq(status, &work)?,
        &code::<R>(c, rows, PointQuadratureOutcome::WorkLimit)?,
        &c.where_cond(
            &c.eq(status, &unresolved)?,
            &code::<R>(c, rows, PointQuadratureOutcome::Unresolved)?,
            &code::<R>(c, rows, PointQuadratureOutcome::Advanced)?,
        )?,
    )?)
}
