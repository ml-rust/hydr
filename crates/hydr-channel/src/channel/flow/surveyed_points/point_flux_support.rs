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

#[cfg(test)]
mod tests {
    use super::super::point_flux::fixture::diagnostics;
    use super::super::{
        chart_fixture, source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_point_flux_positive_support_advances_and_stops() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        positive_support::run::<CpuRuntime>(&client)
    }

    mod positive_support {
        use super::super::super::{
            point_candidates::fixture::fixtures,
            point_flux_association, point_flux_support_metrics,
            point_flux_types::{PointFluxStatus, PointPositiveStatus},
            point_root_session::PointRootSession,
            source_sonic_types::SonicSide,
        };
        use super::{chart_fixture, diagnostics, frozen, hydraulic, support};
        use crate::channel::flow::surveyed::error::checked;
        use frozen::Result;
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            quarter::<R>(client)?;
            advance::<R>(client, false)?;
            advance::<R>(client, true)
        }

        fn quarter<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let (value, positive) = hydraulic(checked::<R, _>(
                client,
                "finite positive runtime target admission",
                |a| {
                    let upper =
                        Tensor::from_slice(&[0., f64::from_bits(1), 1.], &[3], client.device())?;
                    point_flux_support_metrics::finite_quarter(a, &upper)
                },
            ))?;
            assert_eq!(value.try_to_vec::<f64>()?, [0., 0., 0.25]);
            assert_eq!(positive.try_to_vec::<f64>()?, [0., 0., 1.]);
            Ok(())
        }

        fn advance<R>(client: &R::Client, exhausted: bool) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let executions = fixtures::executions();
            let case = executions
                .iter()
                .find(|case| case.id == "scalar_rectangle_vacuum_threshold_inside")
                .ok_or("Missing original inside-threshold execution")?;
            let config = fixtures::CONFIGS[case.config].build();
            chart_fixture::with_pair::<R, _>(client, &[case.id], &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let initial = hydraulic(session.select_point_fluxes(&owners))?;
                diagnostics::check(&session, &initial)?;
                assert_eq!(
                    initial.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::PendingPositiveArea as i64]
                );
                assert_eq!(initial.physical_valid.try_to_vec::<f64>()?, [0.]);
                assert_eq!(initial.joint_valid.try_to_vec::<f64>()?, [0.]);
                assert!(initial.selected.is_empty());
                let unchanged_progress = hydraulic(checked::<R, _>(
                    client,
                    "unchanged enclosure never establishes progress",
                    |a| {
                        let inverse = point_flux_association::inverse(
                            a,
                            session.invocation_rows.numel(),
                            &initial.candidates.invocation_owner,
                        )?;
                        let before =
                            point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
                        let after =
                            point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
                        point_flux_support_metrics::improved(
                            a,
                            &before.fans[SonicSide::Right.index()],
                            &after.fans[SonicSide::Right.index()],
                        )
                    },
                ))?;
                assert_eq!(unchanged_progress.try_to_vec::<f64>()?, [0.]);
                if exhausted {
                    let ceiling = session.callback_points();
                    drop(session);
                    let mut limited = hydraulic(PointRootSession::new(pair, &owners, ceiling))?;
                    let raw = hydraulic(limited.select_point_fluxes(&owners))?;
                    assert_eq!(
                        raw.status.try_to_vec::<i64>()?,
                        [PointFluxStatus::PendingPositiveArea as i64]
                    );
                    assert_eq!(limited.callback_points(), ceiling);
                    let before = work.snapshot();
                    let result = hydraulic(limited.advance_point_flux_positive_support(&owners))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [PointPositiveStatus::WorkLimit as i64]
                    );
                    assert_eq!(result.attempts.try_to_vec::<i64>()?, [0]);
                    assert_eq!(result.progressed.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(result.additional_points, 0);
                    assert_eq!(result.point_limit, ceiling);
                    assert_eq!(result.callback_points, ceiling);
                    assert_eq!(work.snapshot(), before);
                    return Ok(());
                }
                let wet = session
                    .wet
                    .domains
                    .iter()
                    .map(|domain| {
                        Ok((
                            domain.query.query.parameter.lower().try_to_vec::<f64>()?,
                            domain.query.query.parameter.upper().try_to_vec::<f64>()?,
                            domain.iterations,
                            domain.callback_points,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let points = session.callback_points();
                let result = hydraulic(session.advance_point_flux_positive_support(&owners))?;
                hydraulic(
                    session
                        .invocation
                        .check(&result.candidates.invocation_witness),
                )?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointPositiveStatus::Supported as i64]
                );
                assert_eq!(result.attempts.try_to_vec::<i64>()?, [1]);
                assert_eq!(result.progressed.try_to_vec::<f64>()?, [1.]);
                assert!(result.additional_points > 0);
                assert_eq!(result.callback_points - points, result.additional_points);
                assert_eq!(result.point_limit, config.root.max_points);
                assert!(
                    session
                        .sonic
                        .requests
                        .iter()
                        .all(|request| request.binding.side == SonicSide::Right)
                );
                let after_wet = session
                    .wet
                    .domains
                    .iter()
                    .map(|domain| {
                        Ok((
                            domain.query.query.parameter.lower().try_to_vec::<f64>()?,
                            domain.query.query.parameter.upper().try_to_vec::<f64>()?,
                            domain.iterations,
                            domain.callback_points,
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(wet, after_wet);
                let accepted = hydraulic(session.select_point_fluxes(&owners))?;
                diagnostics::check(&session, &accepted)?;
                assert_eq!(
                    accepted.status.try_to_vec::<i64>()?,
                    [PointFluxStatus::Accepted as i64]
                );
                assert_eq!(accepted.physical_valid.try_to_vec::<f64>()?, [1.]);
                assert_eq!(accepted.joint_valid.try_to_vec::<f64>()?, [1.]);
                let points = session.callback_points();
                let repeated = hydraulic(session.advance_point_flux_positive_support(&owners))?;
                assert_eq!(
                    repeated.status.try_to_vec::<i64>()?,
                    [PointPositiveStatus::Supported as i64]
                );
                assert_eq!(repeated.attempts.try_to_vec::<i64>()?, [0]);
                assert_eq!(repeated.progressed.try_to_vec::<f64>()?, [0.]);
                assert_eq!(repeated.additional_points, 0);
                assert_eq!(session.callback_points(), points);
                Ok(())
            })
        }
    }
}
