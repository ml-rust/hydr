use super::*;
use numr::{ops::BinaryOps, runtime::RuntimeClient, tensor::Tensor};

pub(super) fn shared_parent<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let id = CASES[0];
    let config = config(id)?;
    let side = side(id)?;
    chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
        let parents = support::rows::<R>(client, &[0, 0])?;
        let owners = support::rows::<R>(client, &[0, 1])?;
        let mut session = hydraulic(PointRootSession::new(
            pair,
            &parents,
            config.root.max_points,
        ))?;
        hydraulic(session.select_point_fluxes(&owners))?;
        let untouched = snapshots::owner(&session, side, 0)?;
        let initial_cache = chart_test_support::cache(session.sources)?;
        let before = work.snapshot();
        let old_mapped = mapped(&session);
        let input = support::rows::<R>(client, &[1])?;
        let progress = hydraulic(session.advance_source_sonic_quadrature(&input, side))?;
        assert_eq!(progress.invocation_owner.try_to_vec::<i64>()?, [1]);
        assert_eq!(progress.parent_rows.try_to_vec::<i64>()?, [0]);
        assert_eq!(progress.outcome.try_to_vec::<i64>()?, [0]);
        assert_eq!(progress.old_level.try_to_vec::<i64>()?, [0]);
        assert_eq!(progress.new_level.try_to_vec::<i64>()?, [1]);
        assert!(progress.additional_residual_points > 0);
        assert!(progress.additional_simpson_value_points > 0);
        work_delta(
            &progress,
            before,
            work.snapshot(),
            old_mapped,
            mapped(&session),
        );
        let iterations = work
            .snapshot()
            .root_iterations
            .checked_sub(before.root_iterations)
            .ok_or("Selected quadrature root iterations decreased")?;
        assert_eq!(
            progress.additional_residual_points, iterations,
            "One selected native owner must supply each executed midpoint callback"
        );
        assert_eq!(untouched, snapshots::owner(&session, side, 0)?);
        assert_eq!(snapshots::levels(&session, side)?, [(0, 0, 0), (1, 0, 1)]);
        assert_eq!(chart_test_support::cache(session.sources)?, initial_cache);
        client.copy_into(&input, &Tensor::zeros(&[1], DType::I64, client.device())?)?;
        assert_eq!(progress.invocation_owner.try_to_vec::<i64>()?, [1]);
        assert_eq!(progress.parent_rows.try_to_vec::<i64>()?, [0]);
        assert_eq!(snapshots::levels(&session, side)?, [(0, 0, 0), (1, 0, 1)]);
        let before = work.snapshot();
        let old_mapped = mapped(&session);
        let reverse = support::rows::<R>(client, &[1, 0])?;
        let mixed = hydraulic(session.advance_source_sonic_quadrature(&reverse, side))?;
        assert_eq!(mixed.invocation_owner.try_to_vec::<i64>()?, [1, 0]);
        assert_eq!(mixed.parent_rows.try_to_vec::<i64>()?, [0, 0]);
        assert_eq!(mixed.outcome.try_to_vec::<i64>()?, [0, 0]);
        assert_eq!(mixed.old_level.try_to_vec::<i64>()?, [1, 0]);
        assert_eq!(mixed.new_level.try_to_vec::<i64>()?, [2, 1]);
        assert!(mixed.additional_residual_points > 0);
        assert!(mixed.additional_simpson_value_points > 0);
        work_delta(
            &mixed,
            before,
            work.snapshot(),
            old_mapped,
            mapped(&session),
        );
        assert_eq!(snapshots::levels(&session, side)?, [(0, 0, 1), (1, 0, 2)]);
        let selected = hydraulic(session.select_point_fluxes(&reverse))?;
        let reference = flux_fixtures::REFERENCES
            .iter()
            .find(|row| row.id == id)
            .ok_or("Missing shared-parent independent flux truth")?;
        for actual in frozen::endpoints(&selected.physical_water)? {
            contains(
                actual,
                reference.water.ok_or("Missing shared-parent Q truth")?,
                id,
            );
        }
        for actual in frozen::endpoints(&selected.physical_momentum)? {
            contains(
                actual,
                reference.momentum.ok_or("Missing shared-parent J truth")?,
                id,
            );
        }
        let before = work.snapshot();
        let mut checked_owners = std::collections::BTreeSet::new();
        for (index, candidate) in selected.candidates.candidates.iter().enumerate() {
            use crate::channel::flow::surveyed_points::{
                point_candidate_types::PointCandidate, point_flux_operands,
            };
            if !matches!(candidate, PointCandidate::Fan(_)) {
                continue;
            }
            let operand = hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
                client,
                "mixed-level selected fan operand",
                |a| point_flux_operands::candidate(a, &selected.candidates, index),
            ))?;
            let Some(operand) = operand else { continue };
            let order = (0..operand.owners.numel())
                .rev()
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let rows = support::rows::<R>(client, &order)?;
            let operand = hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
                client,
                "mixed-level selected fan evidence",
                |a| point_flux_operands::select(a, &operand, &rows),
            ))?;
            let evidence = operand
                .numerical
                .as_ref()
                .ok_or("Selected fan operand drops its panel-level evidence")?;
            let owners = operand.owners.try_to_vec::<i64>()?;
            let levels = evidence.levels().try_to_vec::<i64>()?;
            assert_eq!(operand.parent.try_to_vec::<i64>()?, vec![0; owners.len()]);
            for (owner, level) in owners.into_iter().zip(levels) {
                assert_eq!(level, if owner == 0 { 1 } else { 2 });
                checked_owners.insert(owner);
            }
        }
        assert_eq!(checked_owners.into_iter().collect::<Vec<_>>(), [0, 1]);
        assert_eq!(work.snapshot(), before);
        assert_eq!(selected.callback_points, session.callback_points());
        assert_eq!(selected.point_limit, config.root.max_points);
        assert_eq!(session.callback_points(), work.snapshot().residual_points);
        Ok(())
    })
}
