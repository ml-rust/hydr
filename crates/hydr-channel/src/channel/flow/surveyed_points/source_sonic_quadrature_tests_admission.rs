use super::*;
use crate::channel::flow::surveyed_points::source_sonic_numerical::SourceSonicNumericalEvidence;
use numr::{ops::BinaryOps, runtime::RuntimeClient, tensor::Tensor};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn provenance<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = config(CASES[0])?;
    let foreign =
        chart_test_support::with_pair::<R, _>(client, &[CASES[1]], &config, |pair, work| {
            let parent = support::rows::<R>(client, &[0])?;
            let empty = support::rows::<R>(client, &[])?;
            let level = hydraulic(pair.panel_level(1))?;
            let before = work.snapshot();
            let values = hydraulic(pair.source_values_at_level(&level, [&parent, &empty]))?;
            assert!(work.snapshot().simpson_value_points > before.simpson_value_points);
            assert_eq!(values.values[0].available.try_to_vec::<f64>()?, [1.]);
            Ok(values)
        })?;
    chart_test_support::with_pair::<R, _>(client, &CASES[..2], &config, |pair, work| {
        let parents = support::rows::<R>(client, &[0, 1])?;
        let first = support::rows::<R>(client, &[0])?;
        let second = support::rows::<R>(client, &[1])?;
        let empty = support::rows::<R>(client, &[])?;
        let mut session = hydraulic(PointRootSession::new(
            pair,
            &parents,
            config.root.max_points,
        ))?;
        hydraulic(session.start_source_sonic(&first, SonicSide::Left))?;
        hydraulic(session.start_source_sonic(&second, SonicSide::Left))?;
        hydraulic(session.start_source_sonic(&first, SonicSide::Right))?;
        let binding = session.sonic.requests[0].binding.clone();
        let other_request = session.sonic.requests[1]
            .domains
            .first()
            .ok_or("Second real sonic request lacks numerical evidence")?
            .numerical
            .clone();
        let other_side = session.sonic.requests[2]
            .domains
            .first()
            .ok_or("Real opposite-side request lacks numerical evidence")?
            .numerical
            .clone();
        let level = hydraulic(session.sources.panel_level(1))?;
        let correct = hydraulic(
            session
                .sources
                .source_values_at_level(&level, [&first, &empty]),
        )?;
        let wrong_parent = hydraulic(
            session
                .sources
                .source_values_at_level(&level, [&second, &empty]),
        )?;
        let wrong_side = hydraulic(
            session
                .sources
                .source_values_at_level(&level, [&empty, &first]),
        )?;
        let before = work.snapshot();
        let snapshot = snapshots::owner(&session, SonicSide::Left, 0)?;
        assert!(
            SourceSonicNumericalEvidence::issue(session.sources, &binding, &wrong_parent).is_err()
        );
        assert!(
            SourceSonicNumericalEvidence::issue(session.sources, &binding, &wrong_side).is_err()
        );
        assert!(SourceSonicNumericalEvidence::issue(session.sources, &binding, &foreign).is_err());
        assert!(
            other_request
                .check(session.sources, &session.invocation, &binding)
                .is_err()
        );
        assert!(
            other_side
                .check(session.sources, &session.invocation, &binding)
                .is_err()
        );
        assert_eq!(work.snapshot(), before);
        assert_eq!(snapshots::owner(&session, SonicSide::Left, 0)?, snapshot);
        let issued = hydraulic(SourceSonicNumericalEvidence::issue(
            session.sources,
            &binding,
            &correct,
        ))?;
        hydraulic(issued.check(session.sources, &session.invocation, &binding))?;
        assert_eq!(issued.levels().try_to_vec::<i64>()?, [1]);
        assert!(
            frozen::components(issued.epsilon())?[0][2] != 0,
            "Provenance control never constructs real numerical source uncertainty"
        );
        client.copy_into(&first, &Tensor::ones(&[1], DType::I64, client.device())?)?;
        assert_eq!(correct.parent_rows[0].try_to_vec::<i64>()?, [0]);
        assert_eq!(binding.rows.try_to_vec::<i64>()?, [0]);
        hydraulic(issued.check(session.sources, &session.invocation, &binding))?;
        let issued_again = hydraulic(SourceSonicNumericalEvidence::issue(
            session.sources,
            &binding,
            &correct,
        ))?;
        assert_eq!(
            frozen::components(issued.epsilon())?,
            frozen::components(issued_again.epsilon())?
        );
        assert_eq!(work.snapshot(), before);
        assert_eq!(snapshots::owner(&session, SonicSide::Left, 0)?, snapshot);
        Ok(())
    })
}

pub(super) fn stale_generation<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = CASES[0];
    let config = config(id)?;
    let side = side(id)?;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        let domain = session
            .sonic
            .requests
            .iter()
            .find(|request| request.binding.side == side)
            .and_then(|request| request.domains.first())
            .ok_or("Stale-generation control lacks a native sonic domain")?;
        assert!(domain.refinement.is_some());
        let old_levels = Tensor::empty(
            domain.numerical.levels().shape(),
            DType::I64,
            client.device(),
        )?;
        client.copy_into(&old_levels, domain.numerical.levels())?;
        assert_eq!(old_levels.try_to_vec::<i64>()?, [0]);
        let progress = hydraulic(session.advance_source_sonic_quadrature(&owners, side))?;
        assert_eq!(progress.outcome.try_to_vec::<i64>()?, [0]);
        assert!(progress.additional_simpson_value_points > 0);
        let domain = session
            .sonic
            .requests
            .iter_mut()
            .find(|request| request.binding.side == side)
            .and_then(|request| request.domains.first_mut())
            .ok_or("Advanced generation loses the native sonic domain")?;
        assert_eq!(domain.numerical.levels().try_to_vec::<i64>()?, [1]);
        let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
        let target = (normalized[1] - normalized[0]) * 0.25;
        assert!(target > 0.);
        domain
            .refinement
            .as_mut()
            .ok_or("Advanced generation loses its native continuation")?
            .levels = old_levels;
        let before = work.snapshot();
        let points = session.callback_points();
        assert!(
            session
                .refine_source_sonic_selected(
                    &owners,
                    side,
                    CertifiedRootTolerance {
                        atol: target,
                        rtol: 0.
                    }
                )
                .is_err()
        );
        assert_eq!(work.snapshot(), before);
        assert_eq!(session.callback_points(), points);
        Ok(())
    })
}
