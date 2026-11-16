use super::super::super::{
    point_candidates::tests::fixtures,
    point_flux_types::{PointFluxKind, PointFluxStatus, ResolvedPointFluxes},
    point_quadrature_types::PointQuadratureOutcome as Outcome,
    point_root_session::PointRootSession,
    source_sonic_quadrature::tests::CASES,
};
use super::{chart_test_support, diagnostics, frozen, hydraulic, support};
use crate::channel::flow::surveyed::PrismaticRiemannConfig;
use frozen::Result;
use numr::{dtype::DType, runtime::Runtime};
use std::collections::BTreeMap;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    ceiling::<R>(client)?;
    exhausted::<R>(client)?;
    star_terminal::<R>(client)?;
    star_level_limit::<R>(client)?;
    Ok(())
}

/// Checks every selection, and that each face holds its owning selection's values.
pub(super) fn coherent<R>(
    session: &PointRootSession<'_, '_, R>,
    result: &ResolvedPointFluxes<R>,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    assert_eq!(result.callback_points, session.callback_points());
    assert_eq!(result.point_limit, session.point_limit());
    for selection in &result.selections {
        diagnostics::selection(session, selection)?;
    }
    let owners = result.invocation_owner.try_to_vec::<i64>()?;
    let status = result.status.try_to_vec::<i64>()?;
    let index = result.selection.try_to_vec::<i64>()?;
    let face = [
        frozen::endpoints(&result.physical_water)?,
        frozen::endpoints(&result.physical_momentum)?,
        frozen::endpoints(&result.joint_water)?,
        frozen::endpoints(&result.joint_momentum)?,
    ];
    let valid = [
        result.physical_valid.try_to_vec::<f64>()?,
        result.joint_valid.try_to_vec::<f64>()?,
    ];
    let held = result
        .selections
        .iter()
        .map(|selection| {
            Ok((
                selection.candidates.invocation_owner.try_to_vec::<i64>()?,
                [
                    frozen::endpoints(&selection.physical_water)?,
                    frozen::endpoints(&selection.physical_momentum)?,
                    frozen::endpoints(&selection.joint_water)?,
                    frozen::endpoints(&selection.joint_momentum)?,
                ],
                [
                    selection.physical_valid.try_to_vec::<f64>()?,
                    selection.joint_valid.try_to_vec::<f64>()?,
                ],
                selection.status.try_to_vec::<i64>()?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    for (i, owner) in owners.iter().enumerate() {
        let (rows, endpoints, flags, statuses) = held
            .get(usize::try_from(index[i])?)
            .ok_or("Face names an absent selection")?;
        let p = rows
            .iter()
            .position(|row| row == owner)
            .ok_or("Owning selection lacks its face")?;
        for (value, expected) in face.iter().zip(endpoints) {
            assert_eq!(value[i], expected[p], "face {owner} mixes selections");
        }
        assert_eq!(valid[0][i], flags[0][p]);
        assert_eq!(valid[1][i], flags[1][p]);
        let original = statuses[p];
        if original == PointFluxStatus::FluxBudgetExceeded as i64 {
            assert!(
                [
                    PointFluxStatus::FluxBudgetExceeded as i64,
                    PointFluxStatus::WorkLimit as i64,
                    PointFluxStatus::RootBracketUnresolved as i64,
                ]
                .contains(&status[i])
            );
        } else if status[i] == PointFluxStatus::WaveEnvelopeUnresolved as i64 {
            assert_eq!(original, PointFluxStatus::Accepted as i64);
            assert_eq!(result.wave_valid.try_to_vec::<f64>()?[i], 0.);
        } else {
            assert_eq!(status[i], original);
        }
    }
    Ok(())
}

/// Per-face status from the initial selection and its positive-area reselection.
pub(super) fn baseline<R: Runtime<DType = DType>>(
    result: &ResolvedPointFluxes<R>,
) -> Result<Vec<i64>> {
    let owners = result.invocation_owner.try_to_vec::<i64>()?;
    let position = owners
        .iter()
        .enumerate()
        .map(|(i, owner)| (*owner, i))
        .collect::<BTreeMap<_, _>>();
    let count = if result.positive_support.is_some() {
        2
    } else {
        1
    };
    let mut status = vec![-1; owners.len()];
    for selection in result.selections.iter().take(count) {
        let rows = selection.candidates.invocation_owner.try_to_vec::<i64>()?;
        for (owner, value) in rows.iter().zip(selection.status.try_to_vec::<i64>()?) {
            let face = position
                .get(owner)
                .ok_or("Selection owner lies outside the faces")?;
            status[*face] = value;
        }
    }
    Ok(status)
}

fn config(id: &str) -> Result<PrismaticRiemannConfig> {
    let rows = fixtures::executions();
    let row = rows
        .iter()
        .find(|row| row.id == id)
        .ok_or("Missing frozen controller execution")?;
    Ok(fixtures::CONFIGS[row.config].build())
}

/// A one-level ceiling stops one sonic row after one round, still over budget.
fn ceiling<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = CASES[6];
    let mut config = config(id)?;
    config.max_refinements = 1;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.resolve_point_fluxes(&owners))?;
        coherent(&session, &result)?;
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointFluxStatus::FluxBudgetExceeded as i64]
        );
        assert_eq!(result.rounds, 1);
        assert_eq!(result.selections.len(), 2);
        assert_eq!(result.sonic_level.try_to_vec::<i64>()?, [1]);
        // Level 1 is the last admissible level, so the outcome names the level ceiling.
        assert_eq!(
            result.sonic_outcome.try_to_vec::<i64>()?,
            [Outcome::LevelLimit as i64]
        );
        let executed = work.snapshot();
        assert_eq!(executed.refinements, 1);
        assert!(executed.maximum_simpson_panels <= config.initial_simpson_panels * 2);
        assert_eq!(session.callback_points(), executed.residual_points);
        Ok(())
    })
}

/// A point limit just past the initial selection stops the continuation at its work ceiling.
fn exhausted<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = CASES[6];
    let config = config(id)?;
    let initial = chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        Ok(hydraulic(session.select_point_fluxes(&owners))?.callback_points)
    })?;
    let limit = initial + 1;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, limit))?;
        let result = hydraulic(session.resolve_point_fluxes(&owners))?;
        coherent(&session, &result)?;
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointFluxStatus::WorkLimit as i64]
        );
        assert_eq!(
            result.sonic_outcome.try_to_vec::<i64>()?,
            [Outcome::WorkLimit as i64]
        );
        assert_eq!(result.rounds, 0);
        assert_eq!(result.point_limit, limit);
        let executed = work.snapshot();
        assert_eq!(executed.refinements, 0);
        assert_eq!(session.callback_points(), executed.residual_points);
        assert_eq!(result.callback_points, executed.residual_points);
        assert!(result.callback_points <= limit);
        Ok(())
    })
}

/// A zero flux budget refines a star row in every round, up to the refinement ceiling.
/// The row stays over budget, and its sonic continuation never runs.
fn star_terminal<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = "scalar_rectangle_depth_jump";
    let mut config = config(id)?;
    config.water_flux_atol_m3_per_s = 0.;
    config.momentum_flux_atol_m4_per_s2 = 0.;
    config.flux_rtol = 0.;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.resolve_point_fluxes(&owners))?;
        coherent(&session, &result)?;
        let kinds = result.selections[0]
            .selected
            .iter()
            .map(|point| point.kind)
            .collect::<Vec<_>>();
        assert_eq!(kinds, [PointFluxKind::Star], "{id} selects {kinds:?}");
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointFluxStatus::FluxBudgetExceeded as i64]
        );
        let star_level = result.star_level.try_to_vec::<i64>()?;
        eprintln!(
            "POINT_FLUX_STAR_TERMINAL\t{id}\trounds={}\tstar_outcome={:?}\tstar_level={star_level:?}",
            result.rounds,
            result.star_outcome.try_to_vec::<i64>()?
        );
        // The last round reaches the refinement ceiling, so the outcome names the level ceiling.
        assert_eq!(
            result.star_outcome.try_to_vec::<i64>()?,
            [Outcome::LevelLimit as i64]
        );
        assert_eq!(star_level, [i64::try_from(config.max_refinements)?]);
        assert_eq!(result.rounds, config.max_refinements);
        assert_eq!(result.selections.len(), result.rounds + 1);
        assert_eq!(
            result.sonic_outcome.try_to_vec::<i64>()?,
            [Outcome::NotApplicable as i64]
        );
        assert_eq!(result.sonic_level.try_to_vec::<i64>()?, [-1]);
        let executed = work.snapshot();
        assert_eq!(executed.refinements, result.rounds);
        assert!(executed.maximum_simpson_panels <= config.max_simpson_panels);
        assert_eq!(session.callback_points(), executed.residual_points);
        Ok(())
    })
}

/// A panel ceiling below the round ceiling makes the star continuation terminal first.
/// The round after the last level advances nothing, so the loop stops there.
fn star_level_limit<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = "scalar_rectangle_depth_jump";
    let mut config = config(id)?;
    config.water_flux_atol_m3_per_s = 0.;
    config.momentum_flux_atol_m4_per_s2 = 0.;
    config.flux_rtol = 0.;
    config.max_simpson_panels = config.initial_simpson_panels * 4;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let result = hydraulic(session.resolve_point_fluxes(&owners))?;
        coherent(&session, &result)?;
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointFluxStatus::FluxBudgetExceeded as i64]
        );
        assert_eq!(
            result.star_outcome.try_to_vec::<i64>()?,
            [Outcome::LevelLimit as i64]
        );
        assert_eq!(result.star_level.try_to_vec::<i64>()?, [2]);
        assert_eq!(result.rounds, 2);
        assert_eq!(result.selections.len(), 3);
        let executed = work.snapshot();
        assert_eq!(executed.refinements, 2);
        assert_eq!(executed.maximum_simpson_panels, config.max_simpson_panels);
        assert_eq!(session.callback_points(), executed.residual_points);
        Ok(())
    })
}
