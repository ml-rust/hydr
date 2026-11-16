use super::*;
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection, error::HydraulicError, work::WorkLedger,
};
use numr::{runtime::RuntimeClient, tensor::Tensor};

/// Stored bits of one wet row: bracket, depth, normalized depth, status and level.
#[derive(Debug, PartialEq, Eq)]
struct RowState {
    bracket: [u64; 2],
    depth: [u64; 2],
    normalized: [u64; 2],
    status: i64,
    level: i64,
}

fn row_state<R>(session: &PointRootSession<'_, '_, R>, owner: i64) -> Result<RowState>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for domain in &session.wet.domains {
        let owners = domain.invocation_owner.try_to_vec::<i64>()?;
        let Some(row) = owners.iter().position(|value| *value == owner) else {
            continue;
        };
        let bits = |value: [f64; 2]| value.map(f64::to_bits);
        let parameter = &domain.query.query.parameter;
        return Ok(RowState {
            bracket: [
                parameter.lower().try_to_vec::<f64>()?[row].to_bits(),
                parameter.upper().try_to_vec::<f64>()?[row].to_bits(),
            ],
            depth: bits(frozen::endpoints(&domain.depth)?[row]),
            normalized: bits(frozen::endpoints(&domain.normalized_depth)?[row]),
            status: domain.status.try_to_vec::<i64>()?[row],
            level: domain.levels.levels().try_to_vec::<i64>()?[row],
        });
    }
    Err("Session lacks a wet domain for the owner".into())
}

fn mirror<R, T>(
    client: &R::Client,
    states: &[[f64; 4]],
    config: &PrismaticRiemannConfig,
    operation: impl FnOnce(
        &mut crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
        &WorkLedger,
    ) -> Result<T>,
) -> Result<T>
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let section = trapezoid::<R>(client)?;
    support::with_section::<R, _>(client, &section, states, config, operation)
}

fn total(work: PrismaticRiemannWork) -> Result<usize> {
    work.residual_points
        .checked_add(work.simpson_value_points)
        .and_then(|points| points.checked_add(work.simpson_derivative_points))
        .ok_or("Wet quadrature control work sum exceeds usize".into())
}

/// `max_refinements = 1` admits one level, and the next call charges nothing.
pub(super) fn level_limit<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = strict(PrismaticRiemannConfig::default());
    config.max_refinements = 1;
    mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        let first = hydraulic(session.advance_wet_star_quadrature(&owners))?;
        assert_eq!(
            first.outcome.try_to_vec::<i64>()?,
            [Outcome::Advanced as i64]
        );
        assert_eq!(first.old_level.try_to_vec::<i64>()?, [0]);
        assert_eq!(first.new_level.try_to_vec::<i64>()?, [1]);
        let state = row_state(&session, 0)?;
        let before = work.snapshot();
        let points = session.callback_points();
        let limited = hydraulic(session.advance_wet_star_quadrature(&owners))?;
        assert_eq!(
            limited.outcome.try_to_vec::<i64>()?,
            [Outcome::LevelLimit as i64]
        );
        assert_eq!(limited.old_level.try_to_vec::<i64>()?, [1]);
        assert_eq!(limited.new_level.try_to_vec::<i64>()?, [1]);
        assert_eq!(limited.bracket_improved.try_to_vec::<f64>()?, [0.]);
        assert_eq!(limited.additional_residual_points, 0);
        assert_eq!(limited.additional_simpson_value_points, 0);
        assert_eq!(limited.additional_mapped_points, 0);
        assert_eq!(work.snapshot(), before);
        assert_eq!(session.callback_points(), points);
        assert_eq!(row_state(&session, 0)?, state);
        assert_eq!(session.point_limit(), config.root.max_points);
        Ok(())
    })
}

/// An exhausted or nearly exhausted point limit stops the call, and executed work matches recorded work.
pub(super) fn work_limit<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = strict(PrismaticRiemannConfig::default());
    let used = mirror::<R, _>(client, &MIRROR, &config, |pair, _| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        Ok(session.callback_points())
    })?;
    assert!(used > 0);
    for extra in [0, 1] {
        let limit = used + extra;
        mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
            hydraulic(session.select_point_fluxes(&owners))?;
            assert_eq!(session.callback_points(), used);
            let state = row_state(&session, 0)?;
            let before = work.snapshot();
            let progress = hydraulic(session.advance_wet_star_quadrature(&owners))?;
            let after = work.snapshot();
            work_delta(&progress, before, after);
            assert_eq!(
                session.callback_points().checked_sub(used),
                Some(progress.additional_residual_points)
            );
            assert!(session.callback_points() <= limit);
            assert_eq!(progress.point_limit, limit);
            assert_eq!(session.point_limit(), limit);
            let outcome = progress.outcome.item::<i64>()?;
            eprintln!(
                "WET_STAR_WORK_LIMIT\textra={extra}\toutcome={outcome}\tresidual={}",
                progress.additional_residual_points
            );
            if extra == 0 {
                assert_eq!(outcome, Outcome::WorkLimit as i64);
                assert_eq!(progress.new_level.try_to_vec::<i64>()?, [0]);
                assert_eq!(after, before);
                assert_eq!(row_state(&session, 0)?, state);
            } else {
                assert!(
                    outcome == Outcome::WorkLimit as i64 || outcome == Outcome::Advanced as i64,
                    "Tiny point limit outcome={outcome}"
                );
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// A row without a wet star is untouched, and an undemanded wet row keeps every stored bit.
pub(super) fn not_applicable<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = strict(PrismaticRiemannConfig::default());
    let states = [MIRROR[0], [12.0, 0.0, 0.0, 0.0], [12.0, -30.0, 12.0, 36.0]];
    mirror::<R, _>(client, &states, &config, |pair, _| {
        let all = support::rows::<R>(client, &[0, 1, 2])?;
        let mut session = hydraulic(PointRootSession::new(pair, &all, config.root.max_points))?;
        let base = hydraulic(session.select_point_fluxes(&all))?;
        let dry = [
            frozen::endpoints(&base.joint_water)?[1].map(f64::to_bits),
            frozen::endpoints(&base.joint_momentum)?[1].map(f64::to_bits),
        ];
        assert!(row_state(&session, 1).is_err(), "Dry row owns a wet domain");
        let undemanded = row_state(&session, 2)?;
        let owners = support::rows::<R>(client, &[0, 1])?;
        let progress = hydraulic(session.advance_wet_star_quadrature(&owners))?;
        assert_eq!(progress.invocation_owner.try_to_vec::<i64>()?, [0, 1]);
        assert_eq!(
            progress.outcome.try_to_vec::<i64>()?,
            [Outcome::Advanced as i64, Outcome::NotApplicable as i64]
        );
        assert_eq!(progress.old_level.try_to_vec::<i64>()?, [0, -1]);
        assert_eq!(progress.new_level.try_to_vec::<i64>()?, [1, -1]);
        assert_eq!(row_state(&session, 2)?, undemanded);
        assert_eq!(row_state(&session, 0)?.level, 1);
        let refined = hydraulic(session.select_point_fluxes(&all))?;
        assert_eq!(
            [
                frozen::endpoints(&refined.joint_water)?[1].map(f64::to_bits),
                frozen::endpoints(&refined.joint_momentum)?[1].map(f64::to_bits),
            ],
            dry
        );
        Ok(())
    })
}

/// A ledger rejection inside the refresh consumes the continuation and seals the session.
pub(super) fn sealed<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = strict(PrismaticRiemannConfig::default());
    let seed = mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        total(work.snapshot())
    })?;
    let ceiling = seed
        .checked_add(1)
        .ok_or("Measured wet quadrature error ceiling exceeds usize")?;
    config.max_native_query_points = ceiling;
    mirror::<R, _>(client, &MIRROR, &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        assert_eq!(total(work.snapshot())?, seed);
        let before = work.snapshot();
        let points = session.callback_points();
        let result = session.advance_wet_star_quadrature(&owners);
        assert!(
            matches!(
                result,
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::WorkLimit,
                    ..
                })
            ),
            "Measured ceiling does not exercise a consumed refresh"
        );
        let after = work.snapshot();
        let residual = after
            .residual_points
            .checked_sub(before.residual_points)
            .ok_or("Failed wet quadrature decreases executed residual work")?;
        assert!(
            residual > 0,
            "Error control never reaches the residual callback"
        );
        assert!(total(after)? <= ceiling);
        assert_eq!(
            session.callback_points().checked_sub(points),
            Some(residual)
        );
        assert_eq!(session.point_limit(), config.root.max_points);
        assert!(session.wet.terminal);
        assert!(session.advance_wet_star_quadrature(&owners).is_err());
        assert!(session.select_point_fluxes(&owners).is_err());
        assert_eq!(work.snapshot(), after);
        let empty = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
        assert!(session.advance_wet_star_quadrature(&empty).is_err());
        assert_eq!(work.snapshot(), after);
        Ok(())
    })
}
