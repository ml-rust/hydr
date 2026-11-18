use super::{
    point_quadrature_progress,
    point_quadrature_types::PointQuadratureProgress,
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
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
    pub(super) fn advance_source_sonic_quadrature(
        &mut self,
        owners: &Tensor<R>,
        side: SonicSide,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(invalid(
                "sonic quadrature remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let mut progress = point_quadrature_progress::start(
            self,
            owners,
            Some(side),
            "sonic quadrature complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) =
            point_quadrature_progress::demand(self, &progress, "sonic quadrature nonempty demand")?;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = mapped::<R>(&self.sonic.requests)?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            for request in &mut self.sonic.requests {
                if request.binding.side != side {
                    continue;
                }
                if c.nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                    .numel()
                    == 0
                {
                    continue;
                }
                if let Err(error) = super::source_sonic_quadrature_request::advance(
                    self.sources,
                    &mut self.invocation,
                    request,
                    &demand,
                    &association,
                    &mut progress,
                ) {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result?;
        point_quadrature_progress::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [old_mapped, mapped::<R>(&self.sonic.requests)?],
            self.callback_points(),
        )?;
        Ok(progress)
    }
}

/// Mapped-coordinate work summed over every sonic domain.
pub(super) fn mapped<R: Runtime<DType = DType>>(
    requests: &[super::source_sonic_types::SourceSonicRequest<R>],
) -> HydraulicResult<usize> {
    requests
        .iter()
        .flat_map(|request| &request.domains)
        .try_fold(0usize, |sum, domain| {
            sum.checked_add(domain.mapped_points)
                .ok_or_else(|| invalid("sonic mapped work sum overflows usize"))
        })
}

#[cfg(test)]
#[path = "source_sonic_quadrature_fixture.rs"]
mod snapshots;

#[cfg(test)]
pub(in crate::channel::flow::surveyed_points) mod tests {
    use super::super::{
        chart_test_support,
        corpus_coverage::Coverage,
        point_candidates::tests::{fixtures, flux_fixtures},
        point_flux_types::{PointFluxKind, PointFluxStatus},
        point_quadrature_types::PointQuadratureProgress,
        point_root_session::PointRootSession,
        source_sonic_types::SonicSide,
        source_tests::support as frozen,
        wet_depth::tests::support,
    };
    use super::snapshots;
    use crate::channel::flow::surveyed::{PrismaticRiemannConfig, work::PrismaticRiemannWork};
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    pub(in crate::channel::flow::surveyed_points) const CASES: [&str; 7] = [
        "scalar_trapezoid_depth_jump",
        "scalar_three_bands_depth_jump",
        "scalar_three_bands_both_knots_dry_fan",
        "scalar_three_bands_both_knots_dry_fan_reverse",
        "scalar_planned_three_bands_both_knots_dry_fan",
        "scalar_planned_three_bands_both_knots_near_pure_fan",
        "scalar_three_bands_loose_recovery_allowance",
    ];

    fn config(id: &str) -> Result<PrismaticRiemannConfig> {
        let rows = fixtures::executions();
        let row = rows
            .iter()
            .find(|row| row.id == id)
            .ok_or("Missing frozen sonic quadrature execution")?;
        Ok(fixtures::CONFIGS[row.config].build())
    }

    fn side(id: &str) -> Result<SonicSide> {
        let rows = fixtures::executions();
        let row = rows
            .iter()
            .find(|row| row.id == id)
            .ok_or("Missing frozen sonic quadrature execution")?;
        match row.traces.first().map(|trace| trace.label) {
            Some("left-sonic") => Ok(SonicSide::Left),
            Some("right-sonic") => Ok(SonicSide::Right),
            _ => Err("Frozen sonic quadrature execution lacks its sampled sonic trace".into()),
        }
    }

    fn mapped<R: Runtime<DType = DType>>(session: &PointRootSession<'_, '_, R>) -> usize {
        session
            .sonic
            .requests
            .iter()
            .flat_map(|request| &request.domains)
            .map(|domain| domain.mapped_points)
            .sum()
    }

    fn work_delta<R>(
        progress: &PointQuadratureProgress<R>,
        before: PrismaticRiemannWork,
        after: PrismaticRiemannWork,
        old_mapped: usize,
        new_mapped: usize,
    ) where
        R: Runtime<DType = DType>,
    {
        assert_eq!(
            after.residual_points.checked_sub(before.residual_points),
            Some(progress.additional_residual_points)
        );
        assert_eq!(
            after
                .simpson_value_points
                .checked_sub(before.simpson_value_points),
            Some(progress.additional_simpson_value_points)
        );
        assert_eq!(
            after
                .simpson_derivative_points
                .checked_sub(before.simpson_derivative_points),
            Some(progress.additional_simpson_derivative_points)
        );
        assert_eq!(
            new_mapped.checked_sub(old_mapped),
            Some(progress.additional_mapped_points)
        );
    }

    fn contains(actual: [f64; 2], expected: [u64; 2], id: &str) {
        let expected = expected.map(f64::from_bits);
        assert!(
            actual[0] <= expected[0] && expected[1] <= actual[1],
            "{id}: refined sonic flux {actual:?} excludes independent truth {expected:?}"
        );
    }

    /// Ownership, admission, and level controls, checked on the CPU.
    fn session_controls<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        ownership::shared_parent::<R>(client)?;
        admission::provenance::<R>(client)?;
        admission::stale_generation::<R>(client)?;
        controls::bounded_levels::<R>(client)?;
        controls::exhausted::<R>(client)?;
        controls::charged_error::<R>(client)?;
        Ok(())
    }

    #[test]
    fn configured_cpu_source_sonic_quadrature_narrows_frozen_fluxes() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        corpus::run::<CpuRuntime>(&client, Coverage::Full)?;
        session_controls::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_source_sonic_quadrature_narrows_frozen_fluxes() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        corpus::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }

    mod admission {
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
            let foreign = chart_test_support::with_pair::<R, _>(
                client,
                &[CASES[1]],
                &config,
                |pair, work| {
                    let parent = support::rows::<R>(client, &[0])?;
                    let empty = support::rows::<R>(client, &[])?;
                    let level = hydraulic(pair.panel_level(1))?;
                    let before = work.snapshot();
                    let values = hydraulic(pair.source_values_at_level(&level, [&parent, &empty]))?;
                    assert!(work.snapshot().simpson_value_points > before.simpson_value_points);
                    assert_eq!(values.values[0].available.try_to_vec::<f64>()?, [1.]);
                    Ok(values)
                },
            )?;
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
                    SourceSonicNumericalEvidence::issue(session.sources, &binding, &wrong_parent)
                        .is_err()
                );
                assert!(
                    SourceSonicNumericalEvidence::issue(session.sources, &binding, &wrong_side)
                        .is_err()
                );
                assert!(
                    SourceSonicNumericalEvidence::issue(session.sources, &binding, &foreign)
                        .is_err()
                );
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
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
    }

    mod ownership {
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
                    let operand =
                        hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
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
                    let operand =
                        hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
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
    }

    mod controls {
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
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
            let limit =
                chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
            let limit =
                support::with_rectangle::<R, _>(client, 1., 4., &input, &config, |pair, _| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
            use crate::channel::flow::surveyed::error::{
                HydraulicError, PrismaticRiemannRejection,
            };
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
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
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
    }

    mod corpus {
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
                        let progress = hydraulic(
                            session.advance_source_sonic_quadrature(&owners, selected_side),
                        )?;
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
    }
}
