use super::{Result, chart_test_support, hydraulic, locate_wet_roots, refine_wet_depth, support};
use crate::channel::flow::surveyed_points::{
    wet_depth_refinement, wet_depth_types::DepthRefinementState,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

#[derive(Clone, Copy)]
enum Entry {
    Depth,
    Native,
    Star,
    Waves,
}

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for initialized in [false, true] {
        for entry in [Entry::Depth, Entry::Native, Entry::Star, Entry::Waves] {
            transplant::<R>(client, initialized, false, entry)?;
        }
        // Keep the fresh outer domain identity while transplanting its foreign continuation authority.
        transplant::<R>(client, initialized, true, Entry::Depth)?;
    }
    Ok(())
}

fn transplant<R>(client: &R::Client, initialized: bool, substate_only: bool, entry: Entry) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let mut config = chart_test_support::config();
    config.root.atol = 0.05;
    config.root.rtol = 0.0;
    let state = if initialized {
        [4.0, 0.0, 1.0, 0.0]
    } else {
        [1.0, 0.0, 1.0, 0.0]
    };
    support::with_rectangle::<R, _>(client, 1.0, 8.0, &[state], &config, |pair, work| {
        let rows = support::rows::<R>(client, &[0])?;
        let (foreign_domain, original_identity, consumed) = {
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let mut first = hydraulic(refine_wet_depth(pair, located))?;
            assert_eq!(first.domains().len(), 1);
            assert_eq!(!first.domains()[0].is_deferred(), initialized);
            let identity = first.invocation.witness();
            let consumed = first.callback_points();
            let domain = first
                .wet
                .domains
                .pop()
                .ok_or("Missing original invocation domain")?;
            (domain, identity, consumed)
        };
        let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
        let mut second = hydraulic(refine_wet_depth(pair, located))?;
        assert_eq!(second.callback_points(), consumed);
        assert_eq!(!second.domains()[0].is_deferred(), initialized);
        assert!(second.invocation.check(&original_identity).is_err());
        let before = work.snapshot();
        let cache = chart_test_support::cache(second.sources())?;
        let points = second.callback_points();
        let limit = second.point_limit();
        if substate_only {
            second.wet.domains[0].refinement = foreign_domain.refinement;
            hydraulic(
                second
                    .invocation
                    .check(&second.wet.domains[0].invocation_witness),
            )?;
            match &second.wet.domains[0].refinement {
                DepthRefinementState::Deferred { original } => {
                    assert!(
                        second
                            .invocation
                            .check(&original.invocation_witness)
                            .is_err()
                    );
                }
                DepthRefinementState::Initialized { state } => {
                    assert!(second.invocation.check(&state.invocation_witness).is_err());
                }
            }
        } else {
            second.wet.domains[0] = foreign_domain;
        }
        assert_eq!(second.callback_points(), points);
        assert_eq!(second.point_limit(), limit);
        let tolerance = CertifiedRootTolerance {
            atol: config.root.atol,
            rtol: config.root.rtol,
        };
        // Matching depth tolerance exercises already-converged fastpaths as well as explicit native admission.
        let error = match entry {
            Entry::Depth => second
                .refine_depth_to(tolerance)
                .err()
                .ok_or("Foreign domain enters depth fastpath")?,
            Entry::Native => {
                let domain = second
                    .wet
                    .domains
                    .pop()
                    .ok_or("Missing transplanted native domain")?;
                let error = wet_depth_refinement::advance(
                    second.sources,
                    &mut second.invocation,
                    domain,
                    &tolerance,
                )
                .err()
                .ok_or("Foreign domain enters native continuation")?;
                drop(second);
                error
            }
            Entry::Star => {
                let error = second
                    .certify_wet_star()
                    .err()
                    .ok_or("Foreign domain enters star evaluation")?;
                assert_eq!(second.callback_points(), points);
                drop(second);
                error
            }
            Entry::Waves => {
                let error = second
                    .certify_wet_waves()
                    .err()
                    .ok_or("Foreign domain enters wave evaluation")?;
                assert_eq!(second.callback_points(), points);
                drop(second);
                error
            }
        };
        assert!(format!("{error:?}").contains("another invocation"));
        assert_eq!(work.snapshot(), before);
        assert_eq!(chart_test_support::cache(pair)?, cache);
        Ok(())
    })
}
