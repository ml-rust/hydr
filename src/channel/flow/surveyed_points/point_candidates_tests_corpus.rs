use super::super::super::{
    point_candidate_types::PointCandidateStatus, wet_root_types::WetRootStatus,
};
use super::{
    PointRootSession, Result, chart_test_support, fixture_model::Execution, fixtures, frozen,
    hydraulic, support,
};
use crate::channel::flow::{
    surveyed::error::{HydraulicError, PrismaticRiemannRejection},
    surveyed_points::corpus_coverage::Coverage,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let corpus = fixtures::executions();
    assert_eq!(corpus.len(), 70);
    let all = coverage.select(corpus, |row| (row.outcome, row.physical));
    let mut executed = 0;
    let mut summary = super::telemetry::Summary::default();
    for (config_index, declared) in fixtures::CONFIGS.iter().enumerate() {
        let rows = all
            .iter()
            .copied()
            .filter(|r| r.config == config_index)
            .collect::<Vec<_>>();
        if rows.is_empty() {
            continue;
        }
        let ids = rows.iter().map(|r| r.id).collect::<Vec<_>>();
        let config = declared.build();
        chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
            let original = source_bits(pair)?;
            let indices = (0..rows.len())
                .map(i64::try_from)
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let parents = support::rows::<R>(client, &indices)?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let result = session.certify_point_candidates(&parents);
            if rows.iter().all(|r| r.outcome == "WorkLimit") {
                assert!(
                    matches!(
                        result,
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        })
                    ),
                    "Declared zero work execution loses its WorkLimit rejection"
                );
                for row in &rows {
                    summary.rejected(row.id);
                }
                assert_eq!(work.snapshot().residual_points, 0);
                return Ok(());
            }
            let result = hydraulic(result)?;
            super::diagnostics::check(&session, &result)?;
            assert_eq!(source_bits(session.sources())?, original);
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            assert!(session.callback_points() <= config.root.max_points);
            let actual = super::observed::actual(&session, &result)?;
            let status = result.status.try_to_vec::<i64>()?;
            let root = result.wet_status.try_to_vec::<i64>()?;
            for (i, row) in rows.iter().enumerate() {
                super::telemetry::record(&mut summary, row, i, &result, &actual)?;
                if row.outcome == "CapExceeded" || row.outcome == "physical_cap_rejection" {
                    assert_eq!(root[i], WetRootStatus::CapExceeded as i64, "{}", row.id);
                    assert_eq!(
                        status[i],
                        PointCandidateStatus::Pending as i64,
                        "{}",
                        row.id
                    );
                    let owner = i64::try_from(i)?;
                    assert!(actual.iter().all(|a| a.owner != owner));
                    continue;
                }
                assert_eq!(
                    status[i],
                    PointCandidateStatus::Complete as i64,
                    "{} loses physical candidate completeness under its declared config",
                    row.id
                );
                check_states(
                    row,
                    i,
                    &actual,
                    status[i] == PointCandidateStatus::Complete as i64,
                )?;
                super::wave_references::check(
                    row,
                    i,
                    &result,
                    status[i] == PointCandidateStatus::Complete as i64,
                )?;
            }
            Ok(())
        })?;
        executed += rows.len();
    }
    assert_eq!(executed, all.len());
    let cap_rows = all
        .iter()
        .filter(|r| r.outcome == "CapExceeded" || r.outcome == "physical_cap_rejection")
        .count();
    let work_rows = all.iter().filter(|r| r.outcome == "WorkLimit").count();
    let applicable_labels = all
        .iter()
        .filter(|r| r.outcome != "WorkLimit")
        .map(|r| r.traces.len())
        .sum();
    if coverage.is_full() {
        assert_eq!(applicable_labels, 76);
    }
    summary.finish(
        [all.len() - cap_rows - work_rows, cap_rows, work_rows],
        applicable_labels,
    );
    Ok(())
}
fn source_bits<R: Runtime<DType = DType>>(
    pair: &super::super::super::paired_sources::PairedSources<'_, R>,
) -> Result<Vec<Vec<[u64; 3]>>> {
    let p = pair.velocity.input();
    [
        &p.left_area,
        &p.left_discharge,
        &p.right_area,
        &p.right_discharge,
    ]
    .iter()
    .map(|p| frozen::components(p))
    .collect()
}
fn check_states(
    row: &Execution,
    owner: usize,
    actual: &[super::observed::State],
    complete: bool,
) -> Result {
    let owner = i64::try_from(owner)?;
    for expected in row.traces {
        let label = super::observed::label(expected.label);
        let matching = actual
            .iter()
            .filter(|a| a.owner == owner && a.label == label)
            .collect::<Vec<_>>();
        if complete {
            assert!(
                !matching.is_empty(),
                "{} lacks frozen sampled label {}",
                row.id,
                expected.label
            );
        }
        for actual in matching {
            compare(actual, expected, row.id);
        }
    }
    if let Some(star) = row.star {
        for actual in actual
            .iter()
            .filter(|a| a.owner == owner && a.label == "star")
        {
            compare(actual, &star, row.id);
        }
    }
    Ok(())
}
fn compare(actual: &super::observed::State, expected: &super::fixture_model::State, id: &str) {
    super::observed::contains(actual.area, expected.area, id);
    super::observed::contains(actual.discharge, expected.discharge, id);
    if let Some(depth) = expected.depth {
        super::observed::contains(actual.depth, depth, id);
    }
    if let (Some(actual), Some(expected)) = (actual.velocity, expected.velocity) {
        super::observed::contains(actual, expected, id);
    }
}
