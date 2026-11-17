use super::{
    point_flux_association,
    point_flux_operands::copy_tensor,
    point_flux_support_metrics::{collect, finite_quarter, improved},
    point_flux_types::{PointPositiveStatus, PointPositiveSupport},
    point_root_session::PointRootSession,
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ReduceOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance, CertifiedUpperBracketAlgorithms,
    },
};

impl<R> PointRootSession<'_, '_, R>
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
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn advance_point_flux_positive_support(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointPositiveSupport<R>> {
        let points_before = self.callback_points();
        let initial = self.certify_point_candidates(owners)?;
        let c = self.sources.geometry.client();
        let (requested, inverse, before, mut attempts, mut unrepresentable, mut exhausted) =
            checked::<R, _>(c, "point positive-support requested snapshots", |a| {
                let requested = copy_tensor(a, &initial.invocation_owner)?;
                let inverse =
                    point_flux_association::inverse(a, self.invocation_rows.numel(), &requested)?;
                let before = collect(a, &initial, &inverse)?;
                let zero = Tensor::zeros(requested.shape(), a.dtype(), c.device())?;
                Ok((
                    requested,
                    inverse,
                    before,
                    Tensor::zeros(initial.invocation_owner.shape(), DType::I64, c.device())?,
                    zero.clone(),
                    zero,
                ))
            })?;
        for side in [SonicSide::Left, SonicSide::Right] {
            let (mut todo, quarter) =
                checked::<R, _>(c, "point positive-support finite runtime target", |a| {
                    let metric = &before.fans[side.index()];
                    let zero = Tensor::zeros(requested.shape(), a.dtype(), c.device())?;
                    let available =
                        c.mul(&c.eq(&metric.work, &zero)?, &c.eq(&metric.floor, &zero)?)?;
                    let todo = c.mul(&before.complete, &c.mul(&metric.blocked, &available)?)?;
                    let (quarter, finite_target) = finite_quarter(a, &metric.normalized_upper)?;
                    let invalid = c.mul(&todo, &c.eq(&finite_target, &zero)?)?;
                    unrepresentable = c.maximum(&unrepresentable, &invalid)?;
                    Ok((c.mul(&todo, &finite_target)?, quarter))
                })?;
            if c.nonzero(&todo)?.numel() == 0 {
                continue;
            }
            if self.invocation.remaining()? == 0 {
                exhausted = c.maximum(&exhausted, &todo)?;
                continue;
            }
            let mut allowance = self.invocation.source_witness().config().root.atol;
            let demand = checked::<R, _>(c, "point positive-support side association", |a| {
                Ok(c.index_put(
                    &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                    0,
                    &requested,
                    &todo,
                )?)
            })?;
            for request in &self.sonic.requests {
                if request.binding.side != side {
                    continue;
                }
                let local = c
                    .nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                    .flatten()?;
                if local.numel() == 0 {
                    continue;
                }
                if !request.tolerance.atol.is_finite() || request.tolerance.atol <= 0. {
                    checked::<R, _>(
                        c,
                        "point positive-support nonpositive current allowance",
                        |a| {
                            let owners = c.index_select(&request.binding.owners, 0, &local)?;
                            let rows = point_flux_association::positions(a, &inverse, &owners)?;
                            todo = c.index_put(
                                &todo,
                                0,
                                &rows,
                                &Tensor::zeros(rows.shape(), a.dtype(), c.device())?,
                            )?;
                            unrepresentable = c.index_put(
                                &unrepresentable,
                                0,
                                &rows,
                                &Tensor::ones(rows.shape(), a.dtype(), c.device())?,
                            )?;
                            Ok(())
                        },
                    )?;
                } else {
                    allowance = allowance.min(request.tolerance.atol);
                }
            }
            let rows = c.nonzero(&todo)?.flatten()?;
            if rows.numel() == 0 {
                continue;
            }
            let quarter = c.index_select(&quarter, 0, &rows)?;
            let atol = c.min(&quarter, &[0], false)?.item::<f64>()?.min(allowance);
            if !atol.is_finite() || atol <= 0. {
                unrepresentable = c.maximum(&unrepresentable, &todo)?;
                continue;
            }
            let owners = c.index_select(&requested, 0, &rows)?;
            let previous = c.index_select(&attempts, 0, &rows)?;
            attempts = c.index_put(
                &attempts,
                0,
                &rows,
                &c.add(
                    &previous,
                    &Tensor::ones(rows.shape(), DType::I64, c.device())?,
                )?,
            )?;
            self.refine_source_sonic_selected(
                &owners,
                side,
                CertifiedRootTolerance { atol, rtol: 0. },
            )?;
        }
        drop(initial);
        let candidates = self.certify_point_candidates(&requested)?;
        let (status, progressed) =
            checked::<R, _>(c, "point positive-support explicit bounded outcomes", |a| {
                let after = collect(a, &candidates, &inverse)?;
                let zero = Tensor::zeros(requested.shape(), a.dtype(), c.device())?;
                let mut blocked = after.other;
                let mut work = exhausted;
                let mut floor = zero.clone();
                let mut progressed = zero.clone();
                for side in 0..2 {
                    blocked = c.maximum(&blocked, &after.fans[side].blocked)?;
                    work = c.maximum(
                        &work,
                        &c.mul(&after.fans[side].work, &after.fans[side].blocked)?,
                    )?;
                    floor = c.maximum(
                        &floor,
                        &c.mul(&after.fans[side].floor, &after.fans[side].blocked)?,
                    )?;
                    progressed = c.maximum(
                        &progressed,
                        &improved(a, &before.fans[side], &after.fans[side])?,
                    )?;
                }
                let code = |value: PointPositiveStatus| -> HydraulicResult<Tensor<R>> {
                    Ok(Tensor::full_scalar(
                        requested.shape(),
                        DType::I64,
                        value as i64 as f64,
                        c.device(),
                    )?)
                };
                let pending = c.where_cond(
                    &progressed,
                    &code(PointPositiveStatus::PendingPositiveArea)?,
                    &code(PointPositiveStatus::NoProgress)?,
                )?;
                let pending = c.where_cond(
                    &unrepresentable,
                    &code(PointPositiveStatus::UnrepresentableTolerance)?,
                    &pending,
                )?;
                let pending =
                    c.where_cond(&floor, &code(PointPositiveStatus::AccuracyFloor)?, &pending)?;
                let pending =
                    c.where_cond(&work, &code(PointPositiveStatus::WorkLimit)?, &pending)?;
                let status =
                    c.where_cond(&blocked, &pending, &code(PointPositiveStatus::Supported)?)?;
                let status = c.where_cond(
                    &after.complete,
                    &status,
                    &code(PointPositiveStatus::PendingCandidates)?,
                )?;
                Ok((status, progressed))
            })?;
        let callback_points = self.callback_points();
        let additional_points = callback_points.checked_sub(points_before).ok_or_else(|| {
            super::point_root_session::invalid(
                "positive-support cumulative callback points decreased",
            )
        })?;
        Ok(PointPositiveSupport {
            candidates,
            status,
            progressed,
            attempts,
            callback_points,
            additional_points,
            point_limit: self.point_limit(),
        })
    }
}
