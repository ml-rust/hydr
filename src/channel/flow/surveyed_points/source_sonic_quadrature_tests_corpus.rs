use super::*;

pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let replayed = coverage.select(CASES, |id| {
        side(id).ok().map(|side| side == SonicSide::Left)
    });
    for group in [&CASES[..6], &CASES[6..]] {
        let ids = group
            .iter()
            .copied()
            .filter(|id| replayed.contains(id))
            .collect::<Vec<_>>();
        if ids.is_empty() {
            continue;
        }
        let ids = ids.as_slice();
        let config = config(ids[0])?;
        let executions = fixtures::executions();
        let original = executions
            .iter()
            .find(|row| row.id == ids[0])
            .ok_or("Missing original quadrature config")?;
        for id in ids {
            let row = executions
                .iter()
                .find(|row| row.id == *id)
                .ok_or("Missing grouped quadrature execution")?;
            assert_eq!(row.config, original.config, "{id}: grouped controls differ");
        }
        chart_test_support::with_pair::<R, _>(client, ids, &config, |pair, work| {
            let parents = (0..ids.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let owners = support::rows::<R>(client, &parents)?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let initial = hydraulic(session.select_point_fluxes(&owners))?;
            assert_eq!(
                initial.status.try_to_vec::<i64>()?,
                vec![PointFluxStatus::FluxBudgetExceeded as i64; ids.len()]
            );
            assert_eq!(
                initial.physical_valid.try_to_vec::<f64>()?,
                vec![1.; ids.len()]
            );
            let old_q = frozen::endpoints(&initial.physical_water)?;
            let old_j = frozen::endpoints(&initial.physical_momentum)?;
            let old_cache = chart_test_support::cache(session.sources)?;
            let old_panels = session.sources.evaluation.panels;
            for selected_side in [SonicSide::Left, SonicSide::Right] {
                let mut demand = Vec::new();
                for (owner, id) in ids.iter().enumerate() {
                    if side(id)? == selected_side {
                        demand.push(i64::try_from(owner)?);
                        let mut present = false;
                        for point in &initial.selected {
                            if point.kind == PointFluxKind::Fan(selected_side)
                                && point
                                    .invocation_owner
                                    .try_to_vec::<i64>()?
                                    .contains(&i64::try_from(owner)?)
                            {
                                present = true;
                            }
                        }
                        assert!(present, "{id}: original sampled sonic fan is absent");
                    }
                }
                if demand.is_empty() {
                    continue;
                }
                let owners = support::rows::<R>(client, &demand)?;
                let before = work.snapshot();
                let old_points = session.callback_points();
                let old_mapped = mapped(&session);
                let progress =
                    hydraulic(session.advance_source_sonic_quadrature(&owners, selected_side))?;
                assert_eq!(progress.invocation_owner.try_to_vec::<i64>()?, demand);
                assert_eq!(progress.parent_rows.try_to_vec::<i64>()?, demand);
                assert_eq!(progress.side, Some(selected_side));
                assert_eq!(progress.outcome.try_to_vec::<i64>()?, vec![0; demand.len()]);
                assert_eq!(
                    progress.old_level.try_to_vec::<i64>()?,
                    vec![0; demand.len()]
                );
                assert_eq!(
                    progress.new_level.try_to_vec::<i64>()?,
                    vec![1; demand.len()]
                );
                assert_eq!(
                    progress.bracket_improved.try_to_vec::<f64>()?,
                    vec![1.; demand.len()]
                );
                assert!(
                    progress.additional_residual_points > 0,
                    "No native refresh work"
                );
                assert!(
                    progress.additional_simpson_value_points > 0,
                    "No refined value work"
                );
                assert!(
                    progress.additional_simpson_derivative_points > 0,
                    "No refined derivative work"
                );
                assert!(
                    progress.additional_mapped_points > 0,
                    "No mapped refresh work"
                );
                work_delta(
                    &progress,
                    before,
                    work.snapshot(),
                    old_mapped,
                    mapped(&session),
                );
                assert_eq!(
                    session.callback_points().checked_sub(old_points),
                    Some(progress.additional_residual_points)
                );
                assert_eq!(progress.callback_points, session.callback_points());
                assert_eq!(progress.point_limit, config.root.max_points);
                assert!(progress.callback_points <= progress.point_limit);
                assert_eq!(session.sources.evaluation.panels, old_panels);
                assert_eq!(chart_test_support::cache(session.sources)?, old_cache);
                let refined = hydraulic(session.select_point_fluxes(&owners))?;
                let q = frozen::endpoints(&refined.physical_water)?;
                let j = frozen::endpoints(&refined.physical_momentum)?;
                for (position, owner) in demand.iter().enumerate() {
                    let owner = usize::try_from(*owner)?;
                    let id = ids[owner];
                    let reference = flux_fixtures::REFERENCES
                        .iter()
                        .find(|row| row.id == id)
                        .ok_or("Missing independent sonic full flux truth")?;
                    contains(
                        q[position],
                        reference.water.ok_or("Missing sonic water truth")?,
                        id,
                    );
                    contains(
                        j[position],
                        reference.momentum.ok_or("Missing sonic momentum truth")?,
                        id,
                    );
                    assert!(
                        q[position][1] - q[position][0] < old_q[owner][1] - old_q[owner][0],
                        "{id}: one level does not narrow physical Q: {:?} -> {:?}",
                        old_q[owner],
                        q[position]
                    );
                    assert!(
                        j[position][1] - j[position][0] < old_j[owner][1] - old_j[owner][0],
                        "{id}: one level does not narrow physical J: {:?} -> {:?}",
                        old_j[owner],
                        j[position]
                    );
                    eprintln!(
                        "SONIC_QUADRATURE_LEVEL\t{id}\tQ={:?}->{:?}\tJ={:?}->{:?}",
                        old_q[owner], q[position], old_j[owner], j[position]
                    );
                }
                assert_eq!(
                    refined.physical_valid.try_to_vec::<f64>()?,
                    vec![1.; demand.len()]
                );
                assert_eq!(refined.callback_points, session.callback_points());
            }
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            Ok(())
        })?;
    }
    Ok(())
}
