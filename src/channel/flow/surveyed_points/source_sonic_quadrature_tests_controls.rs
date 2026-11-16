use super::*;
use numr::{runtime::RuntimeClient, tensor::Tensor};

pub(super) fn bounded_levels<R>(client: &R::Client) -> Result
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
        let initial_cache = chart_test_support::cache(session.sources)?;
        let initial_panels = session.sources.evaluation.panels;
        let mut level = 0_i64;
        let mut limited = false;
        for _ in 0..=config.max_refinements {
            let old = session
                .sonic
                .requests
                .iter()
                .flat_map(|request| &request.domains)
                .map(|domain| {
                    (
                        domain.iterations,
                        domain.callback_points,
                        domain.mapped_points,
                    )
                })
                .collect::<Vec<_>>();
            let before = work.snapshot();
            let old_points = session.callback_points();
            let old_mapped = mapped(&session);
            let result = hydraulic(session.advance_source_sonic_quadrature(&owners, side))?;
            work_delta(
                &result,
                before,
                work.snapshot(),
                old_mapped,
                mapped(&session),
            );
            assert_eq!(result.callback_points, session.callback_points());
            assert_eq!(
                result.callback_points.checked_sub(old_points),
                Some(result.additional_residual_points)
            );
            assert_eq!(result.point_limit, config.root.max_points);
            assert!(result.callback_points <= result.point_limit);
            let after = session
                .sonic
                .requests
                .iter()
                .flat_map(|request| &request.domains)
                .map(|domain| {
                    (
                        domain.iterations,
                        domain.callback_points,
                        domain.mapped_points,
                    )
                })
                .collect::<Vec<_>>();
            assert_eq!(old.len(), after.len());
            for (old, after) in old.iter().zip(&after) {
                assert!(after.0 >= old.0 && after.1 >= old.1 && after.2 >= old.2);
                assert!(after.0 <= config.root.max_iterations);
                assert!(after.1 <= config.root.max_points);
            }
            assert_eq!(result.old_level.try_to_vec::<i64>()?, [level]);
            let outcome = result.outcome.item::<i64>()?;
            if outcome == 1 {
                assert_eq!(result.new_level.try_to_vec::<i64>()?, [level]);
                assert_eq!(result.bracket_improved.try_to_vec::<f64>()?, [0.]);
                assert_eq!(result.additional_residual_points, 0);
                assert_eq!(result.additional_simpson_value_points, 0);
                assert_eq!(result.additional_simpson_derivative_points, 0);
                assert_eq!(result.additional_mapped_points, 0);
                assert_eq!(work.snapshot(), before);
                limited = true;
                break;
            }
            assert_eq!(
                outcome,
                0,
                "Original sonic fixture stops before its panel ceiling: level={level}, work={:?}",
                work.snapshot()
            );
            assert!(result.additional_simpson_value_points > 0);
            assert!(result.additional_simpson_derivative_points > 0);
            level += 1;
            assert_eq!(result.new_level.try_to_vec::<i64>()?, [level]);
            assert!(usize::try_from(level)? <= config.max_refinements);
            assert!(work.snapshot().maximum_simpson_panels <= config.max_simpson_panels);
            assert_eq!(snapshots::levels(&session, side)?, [(0, 0, level)]);
        }
        assert!(
            level > 1 && limited,
            "Explicit advances never reach the original panel ceiling"
        );
        let before = work.snapshot();
        let state = snapshots::owner(&session, side, 0)?;
        let repeated = hydraulic(session.advance_source_sonic_quadrature(&owners, side))?;
        assert_eq!(repeated.outcome.try_to_vec::<i64>()?, [1]);
        assert_eq!(repeated.old_level.try_to_vec::<i64>()?, [level]);
        assert_eq!(repeated.new_level.try_to_vec::<i64>()?, [level]);
        assert_eq!(work.snapshot(), before);
        assert_eq!(snapshots::owner(&session, side, 0)?, state);
        let caches = (
            session.sources.panel_cache.len(),
            session.sources.panel_cache.capacity(),
        );
        let empty = Tensor::empty(&[0], DType::I64, client.device())?;
        let progress = hydraulic(session.advance_source_sonic_quadrature(&empty, side))?;
        assert_eq!(progress.invocation_owner.numel(), 0);
        assert_eq!(progress.parent_rows.numel(), 0);
        assert_eq!(progress.outcome.numel(), 0);
        assert_eq!(progress.old_level.numel(), 0);
        assert_eq!(progress.new_level.numel(), 0);
        assert_eq!(progress.additional_residual_points, 0);
        assert_eq!(progress.additional_simpson_value_points, 0);
        assert_eq!(progress.additional_simpson_derivative_points, 0);
        assert_eq!(progress.additional_mapped_points, 0);
        assert_eq!(progress.callback_points, session.callback_points());
        assert_eq!(progress.point_limit, config.root.max_points);
        assert_eq!(work.snapshot(), before);
        assert_eq!(snapshots::owner(&session, side, 0)?, state);
        assert_eq!(
            (
                session.sources.panel_cache.len(),
                session.sources.panel_cache.capacity()
            ),
            caches
        );
        assert_eq!(session.sources.evaluation.panels, initial_panels);
        assert_eq!(chart_test_support::cache(session.sources)?, initial_cache);
        Ok(())
    })
}

fn no_capacity<R>(
    session: &mut PointRootSession<'_, '_, R>,
    work: &crate::channel::flow::surveyed::work::WorkLedger,
    side: SonicSide,
    owners: &Tensor<R>,
    initialized: bool,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let before = work.snapshot();
    let callbacks = session.callback_points();
    let limit = session.point_limit();
    assert_eq!(callbacks, limit);
    let state = snapshots::owner(session, side, 0)?;
    let native = session
        .sonic
        .requests
        .iter()
        .flat_map(|request| &request.domains)
        .map(|domain| {
            (
                domain.iterations,
                domain.callback_points,
                domain.mapped_points,
            )
        })
        .collect::<Vec<_>>();
    assert!(
        session
            .sonic
            .requests
            .iter()
            .flat_map(|request| &request.domains)
            .all(|domain| domain.refinement.is_some() == initialized)
    );
    let cache = chart_test_support::cache(session.sources)?;
    let allocations = (
        session.sources.panel_cache.len(),
        session.sources.panel_cache.capacity(),
    );
    let panels = session.sources.evaluation.panels;
    let global_limit = session.sources.evaluation.config.max_native_query_points;
    let progress = hydraulic(session.advance_source_sonic_quadrature(owners, side))?;
    assert_eq!(progress.outcome.try_to_vec::<i64>()?, [2]);
    assert_eq!(progress.old_level.try_to_vec::<i64>()?, [0]);
    assert_eq!(progress.new_level.try_to_vec::<i64>()?, [0]);
    assert_eq!(progress.bracket_improved.try_to_vec::<f64>()?, [0.]);
    assert_eq!(progress.additional_residual_points, 0);
    assert_eq!(progress.additional_simpson_value_points, 0);
    assert_eq!(progress.additional_simpson_derivative_points, 0);
    assert_eq!(progress.additional_mapped_points, 0);
    assert_eq!(progress.callback_points, callbacks);
    assert_eq!(progress.point_limit, limit);
    assert_eq!(session.callback_points(), callbacks);
    assert_eq!(session.point_limit(), limit);
    assert_eq!(
        session.sources.evaluation.config.max_native_query_points,
        global_limit
    );
    assert_eq!(session.sources.evaluation.panels, panels);
    assert_eq!(chart_test_support::cache(session.sources)?, cache);
    assert_eq!(
        (
            session.sources.panel_cache.len(),
            session.sources.panel_cache.capacity()
        ),
        allocations
    );
    assert_eq!(snapshots::owner(session, side, 0)?, state);
    assert_eq!(
        session
            .sonic
            .requests
            .iter()
            .flat_map(|request| &request.domains)
            .map(|domain| (
                domain.iterations,
                domain.callback_points,
                domain.mapped_points
            ))
            .collect::<Vec<_>>(),
        native
    );
    assert_eq!(work.snapshot(), before);
    Ok(())
}

pub(super) fn exhausted<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = CASES[0];
    let config = config(id)?;
    let side = side(id)?;
    let limit = chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.start_source_sonic(&owners, side))?;
        assert!(
            session
                .sonic
                .requests
                .iter()
                .flat_map(|request| &request.domains)
                .all(|domain| domain.refinement.is_some())
        );
        Ok(session.callback_points())
    })?;
    assert!(limit > 0);
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
        hydraulic(session.start_source_sonic(&owners, side))?;
        no_capacity(&mut session, work, side, &owners, true)
    })?;
    let config = chart_test_support::config();
    let epsilon = 2.0_f64.powi(-50);
    let input = [[1., 0., 1., 2. - epsilon]];
    let limit = support::with_rectangle::<R, _>(client, 1., 4., &input, &config, |pair, _| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.start_source_sonic(&owners, SonicSide::Right))?;
        assert!(
            session
                .sonic
                .requests
                .iter()
                .flat_map(|request| &request.domains)
                .all(|domain| domain.refinement.is_none())
        );
        Ok(session.callback_points())
    })?;
    assert!(limit > 0);
    support::with_rectangle::<R, _>(client, 1., 4., &input, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
        hydraulic(session.start_source_sonic(&owners, SonicSide::Right))?;
        no_capacity(&mut session, work, SonicSide::Right, &owners, false)
    })
}

fn total(work: PrismaticRiemannWork) -> Result<usize> {
    work.residual_points
        .checked_add(work.simpson_value_points)
        .and_then(|points| points.checked_add(work.simpson_derivative_points))
        .ok_or("Quadrature control work sum exceeds usize".into())
}

pub(super) fn charged_error<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    use crate::channel::flow::surveyed::error::{HydraulicError, PrismaticRiemannRejection};
    let id = CASES[0];
    let mut config = config(id)?;
    let side = side(id)?;
    let (seed_work, ceiling) =
        chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let empty = support::rows::<R>(client, &[])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            hydraulic(session.select_point_fluxes(&owners))?;
            assert!(
                session
                    .sonic
                    .requests
                    .iter()
                    .flat_map(|request| &request.domains)
                    .any(|domain| domain.refinement.is_some())
            );
            let seed = total(work.snapshot())?;
            let level = hydraulic(session.sources.panel_level(1))?;
            let rows = if side == SonicSide::Left {
                [&owners, &empty]
            } else {
                [&empty, &owners]
            };
            hydraulic(session.sources.source_values_at_level(&level, rows))?;
            let primed = total(work.snapshot())?;
            assert!(primed > seed);
            let ceiling = primed
                .checked_add(owners.numel())
                .ok_or("Measured quadrature error ceiling exceeds usize")?;
            Ok((seed, ceiling))
        })?;
    config.max_native_query_points = ceiling;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        assert_eq!(total(work.snapshot())?, seed_work);
        let before = work.snapshot();
        let points = session.callback_points();
        let result = session.advance_source_sonic_quadrature(&owners, side);
        assert!(
            matches!(
                result,
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ),
            "Measured partial-work ceiling does not exercise consumed WorkLimit"
        );
        let after = work.snapshot();
        let residual = after
            .residual_points
            .checked_sub(before.residual_points)
            .ok_or("Failed quadrature advance decreases executed residual work")?;
        assert!(
            residual > 0,
            "Error control never consumes the native continuation"
        );
        assert!(after.simpson_value_points > before.simpson_value_points);
        assert!(total(after)? <= ceiling);
        assert_eq!(
            session.callback_points().checked_sub(points),
            Some(residual)
        );
        assert_eq!(session.point_limit(), config.root.max_points);
        assert!(
            session
                .sonic
                .requests
                .iter()
                .any(|request| request.terminal)
        );
        assert!(
            session
                .advance_source_sonic_quadrature(&owners, side)
                .is_err()
        );
        assert_eq!(work.snapshot(), after);
        assert!(session.certify_source_sonic().is_err());
        assert_eq!(work.snapshot(), after);
        assert_eq!(
            session.callback_points().checked_sub(points),
            Some(residual)
        );
        Ok(())
    })
}
