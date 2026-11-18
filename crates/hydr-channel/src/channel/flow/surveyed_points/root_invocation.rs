use super::source_binding::RootSourceWitness;
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult};
use std::sync::Arc;

struct RootInvocationIdentity;

/// Own one root invocation's fixed source policy and cumulative callback allowance.
pub(super) struct RootInvocation {
    source: RootSourceWitness,
    point_limit: usize,
    callback_points: usize,
    identity: Arc<RootInvocationIdentity>,
}

/// Identity authorizes domains to use their owner. It carries no callback allowance.
#[derive(Clone)]
pub(super) struct RootInvocationWitness {
    identity: Arc<RootInvocationIdentity>,
}

impl RootInvocation {
    pub(super) fn new(source: RootSourceWitness, point_limit: usize) -> HydraulicResult<Self> {
        if point_limit > source.config().root.max_points {
            return Err(inconsistent(&format!(
                "point_limit={point_limit} exceeds captured root.max_points={}",
                source.config().root.max_points
            )));
        }
        Ok(Self {
            source,
            point_limit,
            callback_points: 0,
            identity: Arc::new(RootInvocationIdentity),
        })
    }

    pub(super) fn source_witness(&self) -> &RootSourceWitness {
        &self.source
    }
    pub(super) fn point_limit(&self) -> usize {
        self.point_limit
    }
    pub(super) fn callback_points(&self) -> usize {
        self.callback_points
    }
    pub(super) fn witness(&self) -> RootInvocationWitness {
        RootInvocationWitness {
            identity: Arc::clone(&self.identity),
        }
    }

    pub(super) fn check(&self, witness: &RootInvocationWitness) -> HydraulicResult<()> {
        if !Arc::ptr_eq(&self.identity, &witness.identity) {
            return Err(inconsistent("domain belongs to another invocation"));
        }
        Ok(())
    }

    pub(super) fn remaining(&self) -> HydraulicResult<usize> {
        self.point_limit
            .checked_sub(self.callback_points)
            .ok_or_else(|| {
                inconsistent(&format!(
                    "callback_points={} exceeds point_limit={}",
                    self.callback_points, self.point_limit
                ))
            })
    }

    pub(super) fn check_add_points(&self, points: usize) -> HydraulicResult<()> {
        self.checked_total(points).map(|_| ())
    }

    pub(super) fn add_points(&mut self, points: usize) -> HydraulicResult<()> {
        self.callback_points = self.checked_total(points)?;
        Ok(())
    }

    fn checked_total(&self, points: usize) -> HydraulicResult<usize> {
        let total = self.callback_points.checked_add(points).ok_or_else(|| {
            inconsistent(&format!(
                "callback_points={} plus points={points} overflows usize with point_limit={}",
                self.callback_points, self.point_limit
            ))
        })?;
        if total > self.point_limit {
            return Err(inconsistent(&format!(
                "callback_points={} plus points={points} gives total={total}, exceeding point_limit={}",
                self.callback_points, self.point_limit
            )));
        }
        Ok(total)
    }
}

fn inconsistent(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!(
            "Root invocation {message}. Preserve the original invocation owner and allowance"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::RootInvocation;
    use crate::channel::flow::surveyed_points::{
        chart_fixture, source_fixture_support as frozen,
        wet_depth::{fixture as support, refine_wet_depth},
        wet_roots::locate_wet_roots,
    };
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        bookkeeping::<R>(client)?;
        located_transplant::<R>(client)?;
        exhausted_transfer::<R>(client)?;
        empty_transfer::<R>(client)?;
        transplants::run::<R>(client)
    }

    fn bookkeeping<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            |pair, work| {
                let before = work.snapshot();
                let source = hydraulic(pair.root_witness())?;
                let mut first = hydraulic(RootInvocation::new(source.clone(), 2))?;
                let second = hydraulic(RootInvocation::new(source, 2))?;
                let witness = first.witness();
                hydraulic(first.check(&witness))?;
                assert!(second.check(&witness).is_err());
                assert_eq!(first.point_limit(), 2);
                assert_eq!(first.callback_points(), 0);
                assert_eq!(hydraulic(first.remaining())?, 2);
                hydraulic(first.add_points(1))?;
                assert!(first.add_points(usize::MAX).is_err());
                assert!(first.add_points(2).is_err());
                assert_eq!(first.callback_points(), 1);
                hydraulic(first.add_points(1))?;
                assert_eq!(hydraulic(first.remaining())?, 0);
                assert!(first.add_points(1).is_err());
                assert_eq!(first.callback_points(), 2);
                assert_eq!(first.point_limit(), 2);
                assert_eq!(second.callback_points(), 0);
                let moved = first;
                hydraulic(moved.check(&witness))?;
                assert_eq!(moved.callback_points(), 2);
                assert_eq!(work.snapshot(), before);
                Ok(())
            },
        )
    }

    fn located_transplant<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let mut first = hydraulic(locate_wet_roots(pair, &rows, 1))?;
                let mut second = hydraulic(locate_wet_roots(pair, &rows, 1))?;
                assert_eq!(first.callback_points(), 1);
                assert_eq!(second.callback_points(), 1);
                assert_eq!(first.point_limit(), 1);
                assert_eq!(second.point_limit(), 1);
                assert!(
                    first
                        .invocation
                        .check(&second.domains[0].invocation_witness)
                        .is_err()
                );
                std::mem::swap(&mut first.domains, &mut second.domains);
                let before = work.snapshot();
                let cache = chart_fixture::cache(pair)?;
                let error = refine_wet_depth(pair, second)
                    .err()
                    .ok_or("Foreign located domain enters depth mapping")?;
                assert!(format!("{error:?}").contains("another invocation"));
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_fixture::cache(pair)?, cache);
                Ok(())
            },
        )
    }

    fn exhausted_transfer<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        use solvr::optimize::CertifiedRootTolerance;
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(
            client,
            3.0,
            1.0,
            &[[1.0, 0.0, 1.0, 0.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[0, 0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, 2))?;
                let identity = located.invocation.witness();
                assert_eq!(located.callback_points(), 2);
                assert_eq!(hydraulic(located.invocation.remaining())?, 0);
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                hydraulic(session.invocation.check(&identity))?;
                assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0, 0]);
                assert_eq!(session.callback_points(), 2);
                assert_eq!(session.point_limit(), 2);
                let before = work.snapshot();
                for _ in 0..2 {
                    session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    assert_eq!(session.callback_points(), 2);
                    assert_eq!(session.point_limit(), 2);
                    assert_eq!(hydraulic(session.invocation.remaining())?, 0);
                    for domain in session.domains() {
                        hydraulic(domain.check_invocation(&session.invocation))?;
                    }
                }
                assert_eq!(work.snapshot(), before);
                Ok(())
            },
        )
    }

    fn empty_transfer<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
            let rows = support::rows::<R>(client, &[])?;
            let first = hydraulic(locate_wet_roots(pair, &rows, 0))?;
            let second = hydraulic(locate_wet_roots(pair, &rows, 0))?;
            let identity = first.invocation.witness();
            assert!(second.invocation.check(&identity).is_err());
            let before = work.snapshot();
            let mut session = hydraulic(refine_wet_depth(pair, first))?;
            hydraulic(session.invocation.check(&identity))?;
            assert_eq!(session.point_limit(), 0);
            assert_eq!(session.callback_points(), 0);
            assert!(hydraulic(session.certify_wet_star())?.is_empty());
            assert!(hydraulic(session.certify_wet_waves())?.is_empty());
            assert_eq!(work.snapshot(), before);
            Ok(())
        })
    }

    #[test]
    fn configured_cpu_root_invocation_preserves_allowance_and_domain_identity() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_root_invocation_preserves_allowance_and_domain_identity() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod transplants {
        use super::{
            Result, chart_fixture, hydraulic, locate_wet_roots, refine_wet_depth, support,
        };
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

        fn transplant<R>(
            client: &R::Client,
            initialized: bool,
            substate_only: bool,
            entry: Entry,
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
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
                let cache = chart_fixture::cache(second.sources())?;
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
                assert_eq!(chart_fixture::cache(pair)?, cache);
                Ok(())
            })
        }
    }
}
