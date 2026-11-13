use super::{
    frozen::{self, Result, hydraulic},
    support::{self, DepthClient},
};
use crate::channel::flow::surveyed_points::{
    chart_test_support,
    wet_depth::refine_wet_depth,
    wet_depth_types::{WetDepthSession, WetDepthStatus},
    wet_roots::locate_wet_roots,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.atol = 1e-17;
    config.root.rtol = 0.0;
    let states = [
        [1.0, 0.0, 1.0, 0.0],
        [2.0, 0.0, 2.0, 0.0],
        [1.0 / 32.0, 0.0, 1.0 / 32.0, 0.0],
    ];
    support::with_rectangle::<R, _>(client, 3.0, 4.0, &states, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[2, 0, 1])?;
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        assert_eq!(located.domains.len(), 1);
        assert_eq!(
            located.domains[0].parent_rows.try_to_vec::<i64>()?,
            [2, 0, 1]
        );
        let located_points = located.callback_points;
        let cache = chart_test_support::cache(pair)?;
        let before = work.snapshot();
        let session = hydraulic(refine_wet_depth(pair, located))?;
        check_domains(&session)?;
        assert_eq!(session.domains().len(), 2);
        assert_eq!(work.snapshot().root_solves - before.root_solves, 1);
        assert_eq!(work.snapshot().residual_points - before.residual_points, 2);
        assert_eq!(session.callback_points(), located_points + 2);
        let mut initialized_points = None;
        for domain in session.domains() {
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            let normalized = frozen::endpoints(&domain.normalized_depth)?;
            if parents == [2] {
                assert!(domain.is_deferred());
                assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, [0]);
                assert_eq!(
                    domain.status.try_to_vec::<i64>()?,
                    [WetDepthStatus::DepthConverged as i64]
                );
                assert!(normalized[0][1] > normalized[0][0]);
                assert!(normalized[0][1] - normalized[0][0] <= config.root.atol);
                assert_eq!(domain.callback_points, 0);
            } else {
                assert_eq!(parents, [0, 1]);
                assert!(!domain.is_deferred());
                assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, [1, 2]);
                assert_eq!(
                    domain.status.try_to_vec::<i64>()?,
                    [WetDepthStatus::AccuracyFloor as i64; 2]
                );
                assert!(
                    normalized
                        .iter()
                        .all(|[lower, upper]| upper - lower > config.root.atol)
                );
                assert_eq!(domain.callback_points, 2);
                initialized_points = Some(domain.mapped_points);
            }
            assert_eq!(domain.iterations, 0);
        }
        let initialized_points =
            initialized_points.ok_or("Mixed deferred batch lacks its two-row continuation")?;
        let before = work.snapshot();
        let previous_points = session.callback_points();
        let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
            atol: 0.0,
            rtol: 0.0,
        }))?;
        check_domains(&session)?;
        assert_eq!(session.callback_points(), previous_points + 1);
        assert_eq!(work.snapshot().residual_points - before.residual_points, 1);
        assert_eq!(work.snapshot().root_solves - before.root_solves, 2);
        assert_eq!(work.snapshot().root_iterations, before.root_iterations);
        assert_eq!(
            work.snapshot().maximum_root_iterations,
            before.maximum_root_iterations
        );
        assert_eq!(chart_test_support::cache(session.sources())?, cache);
        for domain in session.domains() {
            assert!(!domain.is_deferred());
            assert!(
                domain
                    .status
                    .try_to_vec::<i64>()?
                    .iter()
                    .all(|&status| status == WetDepthStatus::AccuracyFloor as i64)
            );
            if domain.parent_rows.numel() == 2 {
                assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, [0, 1]);
                assert_eq!(domain.callback_points, 2);
                assert!(domain.mapped_points > initialized_points);
            } else {
                assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, [2]);
                assert_eq!(domain.callback_points, 1);
            }
            assert_eq!(domain.iterations, 0);
        }
        assert_eq!(
            session.callback_points(),
            located_points
                + session
                    .domains()
                    .iter()
                    .map(|domain| domain.callback_points)
                    .sum::<usize>()
        );
        Ok(())
    })
}

fn check_domains<R>(session: &WetDepthSession<'_, '_, R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: DepthClient<R>,
{
    assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [2, 0, 1]);
    let areas = [1.0, 2.0, 1.0 / 32.0];
    // The lower values round A/12 downward. Their successors round it upward.
    let normalized_lower = [0x3fb5555555555555, 0x3fc5555555555555, 0x3f65555555555555];
    for domain in session.domains() {
        let parents = domain.parent_rows.try_to_vec::<i64>()?;
        let owners = domain.invocation_owner.try_to_vec::<i64>()?;
        let diagnostics = domain.query.query.original_rows.try_to_vec::<i64>()?;
        let area = support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?;
        let normalized = frozen::endpoints(&domain.normalized_depth)?;
        for (index, &parent) in parents.iter().enumerate() {
            let parent = usize::try_from(parent)?;
            assert!(parent < areas.len());
            assert_eq!(owners[index], [1, 2, 0][parent]);
            assert_eq!(diagnostics[index], 101 + i64::try_from(parent)?);
            assert_eq!(area[index], [areas[parent]; 2]);
            assert!(normalized[index][0] <= f64::from_bits(normalized_lower[parent]));
            assert!(normalized[index][1] >= f64::from_bits(normalized_lower[parent] + 1));
        }
    }
    Ok(())
}
