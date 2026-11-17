use super::super::super::{
    point_candidates::tests::fixtures,
    point_flux_association, point_flux_support_metrics,
    point_flux_types::{PointFluxStatus, PointPositiveStatus},
    point_root_session::PointRootSession,
    source_sonic_types::SonicSide,
};
use super::{chart_test_support, diagnostics, frozen, hydraulic, support};
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
            let upper = Tensor::from_slice(&[0., f64::from_bits(1), 1.], &[3], client.device())?;
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
    chart_test_support::with_pair::<R, _>(client, &[case.id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
                let before = point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
                let after = point_flux_support_metrics::collect(a, &initial.candidates, &inverse)?;
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
