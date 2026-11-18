use super::{
    paired_sources::PairedSources,
    wet_depth_mapping, wet_depth_refinement, wet_depth_selection,
    wet_depth_types::{DepthRefinementState, WetDepthDomain, WetDepthSession, WetDepthStatus},
};
#[cfg(test)]
use super::{wet_depth_types::WetDepthState, wet_root_types::WetRootDomains};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance},
};

/// Check physical depth while preserving root evidence for later, stricter stopping phases.
#[cfg(test)]
pub(super) fn refine_wet_depth<'s, 'a, R>(
    sources: &'s mut PairedSources<'a, R>,
    located: WetRootDomains<R>,
) -> HydraulicResult<WetDepthSession<'s, 'a, R>>
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
        + CertifiedMappedBisectionAlgorithms<R>,
{
    sources.check_root_witness(located.invocation.source_witness())?;
    located.invocation.remaining()?;
    for domain in &located.domains {
        located.invocation.check(&domain.invocation_witness)?;
    }
    let tolerance = CertifiedRootTolerance {
        atol: located.invocation.source_witness().config().root.atol,
        rtol: located.invocation.source_witness().config().root.rtol,
    };
    let domains = capture_domains(sources, &located.invocation, located.domains)?;
    let c = sources.geometry.client();
    let sonic =
        super::source_sonic_types::SourceSonicState::empty(c, located.invocation_rows.numel())?;
    WetDepthSession {
        sources,
        invocation_witness: located.invocation.witness(),
        invocation: located.invocation,
        invocation_rows: located.invocation_rows,
        wet: WetDepthState {
            locator_status: located.status,
            evidence: located.evidence,
            domains,
            tolerance,
            terminal: false,
        },
        sonic,
        terminal: false,
    }
    .refine_depth_to(tolerance)
}

pub(super) fn capture_domains<R>(
    sources: &PairedSources<'_, R>,
    invocation: &super::root_invocation::RootInvocation,
    originals: Vec<super::wet_root_types::WetRootDomain<R>>,
) -> HydraulicResult<Vec<WetDepthDomain<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    for domain in &originals {
        invocation.check(&domain.invocation_witness)?;
    }
    let c = sources.geometry.client();
    let mut domains = Vec::with_capacity(originals.len());
    for original in originals {
        let (depth, normalized_depth) =
            wet_depth_mapping::certificates(sources, &original.parent_rows, &original.query)?;
        let (query, invocation_owner, parent_rows) =
            checked::<R, _>(c, "wet depth original bracket capture", |a| {
                let selected = sources.select_sources(a, &original.parent_rows)?;
                let query = original.query.with_parameter_bounds(
                    a,
                    &selected.bands,
                    &original.query.query.parameter,
                )?;
                let owners =
                    Tensor::empty(original.invocation_owner.shape(), DType::I64, c.device())?;
                c.copy_into(&owners, &original.invocation_owner)?;
                Ok((query, owners, selected.rows))
            })?;
        domains.push(WetDepthDomain {
            invocation_witness: original.invocation_witness.clone(),
            invocation_owner,
            parent_rows,
            query,
            depth,
            normalized_depth,
            status: Tensor::full_scalar(
                original.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?,
            levels: super::wet_star_levels::WetStarLevels::initial(
                c,
                original.parent_rows.numel(),
            )?,
            iterations: 0,
            callback_points: 0,
            mapped_points: 0,
            refinement: DepthRefinementState::Deferred {
                original: Box::new(original),
            },
        });
    }
    Ok(domains)
}

impl<'s, 'a, R> WetDepthSession<'s, 'a, R>
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
        + CertifiedMappedBisectionAlgorithms<R>,
{
    /// Request no looser precision than the original configured normalized-depth allowances.
    #[cfg(test)]
    pub(super) fn refine_depth_to(
        mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<Self> {
        self.refine_wet_depth_in_place(tolerance)?;
        Ok(self)
    }

    pub(super) fn refine_wet_depth_in_place(
        &mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        check_tolerance(&tolerance, &self.invocation.source_witness().config().root)?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet refinement remains sealed after an earlier numerical error",
            ));
        }
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let c = self.sources.geometry.client();
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for mut domain in std::mem::take(&mut self.wet.domains) {
                let accepted = checked::<R, _>(c, "wet depth stopping criterion", |a| {
                    wet_depth_mapping::satisfied(a, &domain.normalized_depth, &tolerance)
                })?;
                let passing = c.nonzero(&accepted)?.flatten()?;
                if passing.numel() == domain.parent_rows.numel() {
                    domain.status = Tensor::full_scalar(
                        domain.parent_rows.shape(),
                        DType::I64,
                        WetDepthStatus::DepthConverged as i64 as f64,
                        c.device(),
                    )?;
                    output.push(domain);
                    continue;
                }
                if domain.is_deferred() && passing.numel() != 0 {
                    let zero = Tensor::zeros(accepted.shape(), accepted.dtype(), c.device())?;
                    let failing = c.nonzero(&c.eq(&accepted, &zero)?)?.flatten()?;
                    let (pass, fail) = checked::<R, _>(c, "wet depth deferred partition", |a| {
                        Ok((
                            wet_depth_selection::deferred(
                                a,
                                &domain,
                                &passing,
                                WetDepthStatus::DepthConverged,
                            )?,
                            wet_depth_selection::deferred(
                                a,
                                &domain,
                                &failing,
                                WetDepthStatus::Unresolved,
                            )?,
                        ))
                    })?;
                    output.push(pass);
                    domain = fail;
                }
                output.push(wet_depth_refinement::advance(
                    self.sources,
                    &mut self.invocation,
                    domain,
                    &tolerance,
                )?);
            }
            self.wet.domains = output;
            self.wet.tolerance = tolerance;
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        if result.is_err() {
            self.wet.terminal = true;
        }
        result
    }
}

pub(super) fn check_tolerance(
    tolerance: &CertifiedRootTolerance,
    original: &solvr::optimize::CertifiedBisectionOptions,
) -> HydraulicResult<()> {
    for (name, value, limit) in [
        ("atol", tolerance.atol, original.atol),
        ("rtol", tolerance.rtol, original.rtol),
    ] {
        if !value.is_finite() || value < 0.0 || value > limit {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Wet depth {name}={value} exceeds original finite nonnegative allowance={limit}. Request an allowance within the original ceiling"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "wet_depth_fixture.rs"]
pub(in crate::channel::flow::surveyed_points) mod fixture;

#[cfg(test)]
mod tests {
    use super::fixture as support;
    use crate::channel::flow::surveyed_points::source_fixture_support as frozen;
    use frozen::Result;
    use numr::{dtype::DType, runtime::Runtime};

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        lifecycle::run::<R>(client)?;
        partition::run::<R>(client)?;
        admission::run::<R>(client)?;
        budget::run::<R>(client)
    }

    #[test]
    fn configured_cpu_wet_depth_preserves_physical_certificates_and_continuation() -> Result {
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
    fn actual_cuda_wet_depth_preserves_physical_certificates_and_continuation() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod admission {
        use super::{
            frozen::{Result, hydraulic},
            support::{self, DepthClient},
        };
        use crate::channel::flow::surveyed::error::HydraulicError;
        use crate::channel::flow::surveyed_points::{
            chart_fixture, wet_depth::refine_wet_depth, wet_roots::locate_wet_roots,
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            source_identity_precedes_early_acceptance::<R>(client)?;
            construction_policy_precedes_location::<R>(client)?;
            location_policy_precedes_early_acceptance::<R>(client)?;
            invalid_tolerance_does_not_evaluate::<R>(client)?;
            empty_session_preserves_zero_work::<R>(client)
        }

        fn construction_policy_precedes_location<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let changed = crate::channel::flow::surveyed::PrismaticRiemannConfig {
                gravity_m_per_s2: 2.0,
                ..config.clone()
            };
            for panels in [false, true] {
                support::with_rectangle_policy::<R, _>(
                    client,
                    3.0,
                    1.0,
                    &[[1.0, 0.0, 1.0, 0.0]],
                    &config,
                    &changed,
                    |pair, replacement, work| {
                        let rows = support::rows::<R>(client, &[0])?;
                        hydraulic(pair.source_values([&rows, &rows]))?;
                        let cache = chart_fixture::cache(pair)?;
                        assert!(cache.iter().all(|entry| entry.ready == [1.0]));
                        let before = work.snapshot();
                        if panels {
                            pair.evaluation.panels += 2;
                        } else {
                            pair.evaluation.config = replacement;
                        }
                        let error = locate_wet_roots(pair, &rows, config.root.max_points)
                            .err()
                            .ok_or("Changed construction policy permits root location")?;
                        assert!(
                            matches!(&error, HydraulicError::Numerical { .. }),
                            "{error:?}"
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(pair)?, cache);
                        let error = pair
                            .source_values([&rows, &rows])
                            .err()
                            .ok_or("Changed construction policy permits cached source reuse")?;
                        assert!(
                            matches!(&error, HydraulicError::Numerical { .. }),
                            "{error:?}"
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(pair)?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }

        fn source_identity_precedes_early_acceptance<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let section = chart_fixture::rectangle::<R>(client, 3.0, 1.0)?;
            let located = support::with_section::<R, _>(
                client,
                &section,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, _| {
                    let rows = support::rows::<R>(client, &[0])?;
                    hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))
                },
            )?;
            support::with_section::<R, _>(
                client,
                &section,
                &[[1.0, 0.0, 1.0, 1.0]],
                &config,
                |other, work| {
                    assert!(
                        other
                            .select_rows(&located.domains[0].parent_rows, &located.domains[0].query)
                            .is_ok()
                    );
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(other)?;
                    assert!(refine_wet_depth(other, located).is_err());
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(other)?, cache);
                    Ok(())
                },
            )
        }

        fn location_policy_precedes_early_acceptance<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let changed = crate::channel::flow::surveyed::PrismaticRiemannConfig {
                gravity_m_per_s2: 2.0,
                ..config.clone()
            };
            for panels in [false, true] {
                support::with_rectangle_policy::<R, _>(
                    client,
                    3.0,
                    1.0,
                    &[[1.0, 0.0, 1.0, 0.0]],
                    &config,
                    &changed,
                    |pair, replacement, work| {
                        let rows = support::rows::<R>(client, &[0])?;
                        let located =
                            hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(pair)?;
                        if panels {
                            pair.evaluation.panels += 2;
                        } else {
                            pair.evaluation.config = replacement;
                        }
                        assert!(refine_wet_depth(pair, located).is_err());
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(pair)?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }

        fn invalid_tolerance_does_not_evaluate<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                4.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    for [atol, rtol] in [
                        [-1.0, 0.0],
                        [f64::NAN, 0.0],
                        [f64::INFINITY, 0.0],
                        [0.0, -1.0],
                        [0.0, f64::NAN],
                        [0.0, f64::INFINITY],
                        [2.0 * config.root.atol, config.root.rtol],
                        [config.root.atol, 2.0 * config.root.rtol],
                    ] {
                        let rows = support::rows::<R>(client, &[0])?;
                        let located =
                            hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                        let session = hydraulic(refine_wet_depth(pair, located))?;
                        let before = work.snapshot();
                        assert!(
                            session
                                .refine_depth_to(CertifiedRootTolerance { atol, rtol })
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                    }
                    Ok(())
                },
            )
        }

        fn empty_session_preserves_zero_work<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(client, 1.0, 4.0, &[], &config, |pair, work| {
                let rows = support::rows::<R>(client, &[])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(session.point_limit(), 0);
                assert_eq!(session.callback_points(), 0);
                assert_eq!(session.invocation_rows().numel(), 0);
                assert_eq!(session.locator_status().numel(), 0);
                assert!(session.domains().is_empty());
                let before = work.snapshot();
                let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
                assert_eq!(session.callback_points(), 0);
                assert_eq!(work.snapshot(), before);
                Ok(())
            })
        }
    }

    mod budget {
        use super::{
            frozen::{Result, expansion, hydraulic},
            support::{self, DepthClient},
        };
        use crate::channel::flow::{
            surveyed::error::{HydraulicError, PrismaticRiemannRejection},
            surveyed_points::{
                chart_fixture, wet_depth::refine_wet_depth, wet_depth_types::WetDepthStatus,
                wet_root_types::WetRootStatus, wet_roots::locate_wet_roots,
            },
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            locator_allowance_stays_narrow::<R>(client)?;
            independent_iteration_ceilings::<R>(client)?;
            callback_work_error_remains_typed::<R>(client)
        }

        fn locator_allowance_stays_narrow<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                3.0,
                1.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, 1))?;
                    assert_eq!(located.point_limit(), 1);
                    assert_eq!(located.callback_points(), 1);
                    assert_eq!(
                        located.status.try_to_vec::<i64>()?,
                        [WetRootStatus::Located as i64]
                    );
                    let session = hydraulic(refine_wet_depth(pair, located))?;
                    assert_eq!(session.point_limit(), 1);
                    assert!(session.domains()[0].is_deferred());
                    let before = work.snapshot();
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    assert_eq!(session.point_limit(), 1);
                    assert_eq!(session.callback_points(), 1);
                    assert_eq!(
                        session.domains()[0].status.try_to_vec::<i64>()?,
                        [WetDepthStatus::WorkLimit as i64]
                    );
                    assert!(session.domains()[0].is_deferred());
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }

        fn independent_iteration_ceilings<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.root.atol = 0.0;
            config.root.rtol = 0.0;
            config.root.max_iterations = 1;
            let section = chart_fixture::rectangle::<R>(client, 1.0, 4.0)?;
            let owners = support::rows::<R>(client, &[0, 0])?;
            let epsilon = 3.0 * 2.0_f64.powi(-80);
            let fields = [
                [[4.0_f64.to_bits(), 0, 0], [1.0_f64.to_bits(), 0, 0]],
                [[0, 0, 0], [(-2.0_f64).to_bits(), 0, 0]],
                [[1.0_f64.to_bits(), 0, 0], [1.0_f64.to_bits(), 0, 0]],
                [[0, 0, 0], [2.0_f64.to_bits(), (-epsilon).to_bits(), 0]],
            ];
            let inputs = fields
                .iter()
                .map(|field| expansion::<R>(client, field, true))
                .collect::<Result<Vec<_>>>()?;
            chart_fixture::with_inputs(client, section, owners, inputs, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0, 1])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                assert_eq!(
                    located.status.try_to_vec::<i64>()?,
                    [WetRootStatus::Located as i64; 2]
                );
                assert_eq!(located.domains.len(), 2);
                let points = located.callback_points();
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(session.domains().len(), 2);
                assert!(
                    session
                        .domains()
                        .iter()
                        .all(|domain| domain.iterations == 1)
                );
                assert_eq!(work.snapshot().root_iterations, 2);
                assert_eq!(work.snapshot().maximum_root_iterations, 1);
                for domain in session.domains() {
                    let parents = domain.parent_rows.try_to_vec::<i64>()?;
                    let areas = support::area_bounds(
                        session.sources(),
                        &domain.parent_rows,
                        &domain.query,
                    )?;
                    for (parent, [lower, upper]) in parents.into_iter().zip(areas) {
                        if parent == 1 {
                            // F(A)=4*sqrt(A)-3*2^-80 has the exact root 9*2^-164.
                            let star = 9.0 * 2.0_f64.powi(-164);
                            assert!(lower <= star && star <= upper);
                        }
                    }
                }
                assert_eq!(
                    session.callback_points(),
                    points
                        + session
                            .domains()
                            .iter()
                            .map(|domain| domain.callback_points)
                            .sum::<usize>()
                );
                let before = work.snapshot();
                let points = session.callback_points();
                let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
                assert!(
                    session
                        .domains()
                        .iter()
                        .all(|domain| domain.iterations == 1)
                );
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot().residual_points, before.residual_points);
                assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                assert_eq!(work.snapshot().maximum_root_iterations, 1);
                Ok(())
            })
        }

        fn callback_work_error_remains_typed<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = crate::channel::flow::surveyed::PrismaticRiemannConfig {
                max_native_query_points: 1,
                ..chart_fixture::config()
            };
            support::with_rectangle::<R, _>(
                client,
                3.0,
                1.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let session = hydraulic(refine_wet_depth(pair, located))?;
                    assert_eq!(work.snapshot().residual_points, 1);
                    let result = session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    });
                    assert!(matches!(
                        result,
                        Err(HydraulicError::Rejected {
                            reason: PrismaticRiemannRejection::WorkLimit,
                            ..
                        })
                    ));
                    assert_eq!(work.snapshot().residual_points, 1);
                    assert_eq!(work.snapshot().root_iterations, 0);
                    Ok(())
                },
            )
        }
    }

    mod lifecycle {
        use super::{
            frozen::{self, Result, hydraulic},
            support::{self, DepthClient},
        };
        use crate::channel::flow::{
            surveyed::error,
            surveyed_points::{
                chart_fixture,
                potential_range::select_expansion,
                wet_depth::refine_wet_depth,
                wet_depth_types::{DepthRefinementState, WetDepthStatus},
                wet_root_types::WetRootStatus,
                wet_roots::locate_wet_roots,
            },
        };
        use numr::{
            dtype::DType,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            deferred_certificates::<R>(client)?;
            pointed_dry_adjacent_depth::<R>(client)?;
            exact_parameter_keeps_depth_uncertainty::<R>(client)?;
            mapped_refinement::<R>(client)?;
            selected_caches_stay_fixed::<R>(client)?;
            mixed_classifications::<R>(client)
        }

        fn pointed_dry_adjacent_depth<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = crate::channel::flow::surveyed::PrismaticRiemannConfig {
                gravity_m_per_s2: 2.0,
                ..chart_fixture::config()
            };
            support::with_pointed_film::<R, _>(client, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let before = work.snapshot();
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(work.snapshot(), before);
                assert_eq!(session.domains().len(), 1);
                let domain = &session.domains()[0];
                assert!(domain.is_deferred());
                assert_eq!(
                    domain.status.try_to_vec::<i64>()?,
                    [WetDepthStatus::DepthConverged as i64]
                );
                // A=h² and F(A)=8*A^(1/4)-2^-80 give h*=2^-166 at capacity depth4.
                let area =
                    support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?[0];
                let star = 2.0_f64.powi(-332);
                assert!(area[0] <= star && star <= area[1]);
                let depth = frozen::endpoints(&domain.depth)?[0];
                let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
                assert!(depth[0] <= 2.0_f64.powi(-166) && depth[1] >= 2.0_f64.powi(-166));
                assert!(normalized[0] <= 2.0_f64.powi(-168) && normalized[1] >= 2.0_f64.powi(-168));
                Ok(())
            })
        }

        fn deferred_certificates<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let ids = [
                "scalar_rectangle_uniform_rest",
                "retained_vacuum_inside_film",
                "uniform_cap_retained_deficit_dyadic_velocity",
            ];
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0, 1, 2])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                assert_eq!(
                    located.status.try_to_vec::<i64>()?,
                    [WetRootStatus::Located as i64; 3]
                );
                let original = located
                    .domains
                    .iter()
                    .map(|domain| support::snapshot(pair, domain))
                    .collect::<Result<Vec<_>>>()?;
                let points = located.callback_points();
                let evidence_count = located.evidence.len();
                let before = work.snapshot();
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(session.evidence().len(), evidence_count);
                assert_eq!(
                    session.tolerance().atol.to_bits(),
                    config.root.atol.to_bits()
                );
                assert_eq!(
                    session.tolerance().rtol.to_bits(),
                    config.root.rtol.to_bits()
                );
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot(), before);
                assert_eq!(session.domains().len(), original.len());
                let mut film = false;
                let mut cap = false;
                for (domain, expected) in session.domains().iter().zip(original) {
                    assert!(domain.is_deferred());
                    assert!(
                        domain
                            .status
                            .try_to_vec::<i64>()?
                            .iter()
                            .all(|&status| status == WetDepthStatus::DepthConverged as i64)
                    );
                    assert_eq!(domain.iterations, 0);
                    assert_eq!(domain.callback_points, 0);
                    let DepthRefinementState::Deferred { original } = &domain.refinement else {
                        return Err(
                            "Depth-converged locator bracket unexpectedly initialized".into()
                        );
                    };
                    assert_eq!(support::snapshot(session.sources(), original)?, expected);
                    let parents = domain.parent_rows.try_to_vec::<i64>()?;
                    let areas = support::area_bounds(
                        session.sources(),
                        &domain.parent_rows,
                        &domain.query,
                    )?;
                    let normalized = frozen::endpoints(&domain.normalized_depth)?;
                    for (index, parent) in parents.into_iter().enumerate() {
                        if parent == 1 {
                            let star = 2.0_f64.powi(-164);
                            assert!(areas[index][0] <= star && star <= areas[index][1]);
                            assert!(areas[index][1] < 1e-40);
                            let depth = star / 4.0;
                            assert!(normalized[index][0] <= depth && depth <= normalized[index][1]);
                            film = true;
                        }
                        if parent == 2 {
                            assert!(normalized[index][0] <= 1.0 && normalized[index][1] >= 1.0);
                            cap = true;
                        }
                    }
                    hydraulic(error::checked::<R, _>(
                        client,
                        "retained depth cap identity",
                        |a| {
                            let parents = domain.parent_rows.try_to_vec::<i64>()?;
                            if let Some(index) = parents.iter().position(|&parent| parent == 2) {
                                let local = Tensor::from_slice(
                                    &[i64::try_from(index).map_err(|_| {
                                        error::HydraulicError::Numerical {
                                            context: "Depth test index exceeds I64".into(),
                                        }
                                    })?],
                                    &[1],
                                    client.device(),
                                )?;
                                let query = domain.query.select(a, &local)?;
                                let parent = Tensor::from_slice(&[2_i64], &[1], client.device())?;
                                let truth = select_expansion(
                                    a,
                                    &session.sources().velocity.input().left_area,
                                    &parent,
                                )?;
                                assert_eq!(truth.high().item::<f64>()?, 14.0);
                                assert_eq!(truth.low().item::<f64>()?, -2.0_f64.powi(-80));
                                let bands = session.sources().bands.select(a, &parent)?;
                                let boundaries = session.sources().boundaries.select(a, &parent)?;
                                use crate::channel::section::PhysicalEndpoint;
                                for (endpoint, lower) in [
                                    (PhysicalEndpoint::Lower, true),
                                    (PhysicalEndpoint::Upper, false),
                                ] {
                                    let area = query
                                        .endpoint(a, &bands, endpoint)?
                                        .retained_endpoint_area(a, &bands, &boundaries)?;
                                    let difference = a.expansion_sub(&area, &truth)?;
                                    let difference = a.expansion_interval(&difference)?;
                                    if lower {
                                        assert!(difference.lower().item::<f64>()? <= 0.0);
                                    } else {
                                        assert!(difference.upper().item::<f64>()? >= 0.0);
                                    }
                                }
                            }
                            Ok(())
                        },
                    ))?;
                }
                assert!(film && cap);
                Ok(())
            })
        }

        fn exact_parameter_keeps_depth_uncertainty<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                3.0,
                1.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let session = hydraulic(refine_wet_depth(pair, located))?;
                    assert!(session.domains()[0].is_deferred());
                    let before = work.snapshot();
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    let domain = &session.domains()[0];
                    assert_eq!(
                        domain.status.try_to_vec::<i64>()?,
                        [WetDepthStatus::AccuracyFloor as i64]
                    );
                    assert!(!domain.is_deferred());
                    assert_eq!(domain.query.query.parameter.lower().item::<f64>()?, 0.0);
                    assert_eq!(domain.query.query.parameter.upper().item::<f64>()?, 0.0);
                    let depth = frozen::endpoints(&domain.normalized_depth)?[0];
                    assert!(depth[0] < depth[1]);
                    assert!(depth[0] <= 1.0 / 3.0 && depth[1] >= 1.0 / 3.0);
                    assert_eq!(domain.iterations, 0);
                    assert_eq!(
                        work.snapshot().residual_points - before.residual_points,
                        domain.callback_points
                    );
                    let before = work.snapshot();
                    let used = session.callback_points();
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    assert_eq!(session.callback_points(), used);
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                    Ok(())
                },
            )
        }

        fn mapped_refinement<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.root.atol = 0.05;
            config.root.rtol = 0.0;
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_depth_jump"],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let points = located.callback_points();
                    let session = hydraulic(refine_wet_depth(pair, located))?;
                    assert_eq!(session.domains().len(), 1);
                    let first = session.domains()[0].iterations;
                    assert!(first > 0 && !session.domains()[0].is_deferred());
                    let before = work.snapshot();
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 1e-5,
                        rtol: 0.0,
                    }))?;
                    let second = session.domains()[0].iterations;
                    assert!(second > first);
                    assert_eq!(
                        work.snapshot().residual_points - before.residual_points,
                        second - first
                    );
                    assert_eq!(
                        work.snapshot().root_iterations - before.root_iterations,
                        second - first
                    );
                    assert_eq!(work.snapshot().root_solves - before.root_solves, 1);
                    let before = work.snapshot();
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 1e-10,
                        rtol: 0.0,
                    }))?;
                    let domain = &session.domains()[0];
                    assert!(domain.iterations > second);
                    assert_eq!(
                        work.snapshot().residual_points - before.residual_points,
                        domain.iterations - second
                    );
                    assert_eq!(
                        work.snapshot().root_iterations - before.root_iterations,
                        domain.iterations - second
                    );
                    assert_eq!(work.snapshot().root_solves - before.root_solves, 1);
                    assert!(domain.iterations <= config.root.max_iterations);
                    assert_eq!(
                        domain.status.try_to_vec::<i64>()?,
                        [WetDepthStatus::DepthConverged as i64]
                    );
                    assert_eq!(session.callback_points(), points + domain.callback_points);
                    assert_eq!(work.snapshot().residual_points, session.callback_points());
                    assert_eq!(work.snapshot().root_iterations, domain.iterations);
                    assert_eq!(work.snapshot().maximum_root_iterations, domain.iterations);
                    let normalized = frozen::endpoints(&domain.normalized_depth)?[0];
                    assert!(normalized[1] - normalized[0] <= 1e-10);
                    Ok(())
                },
            )
        }

        fn mixed_classifications<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let ids = [
                "scalar_rectangle_uniform_rest",
                "scalar_rectangle_vacuum",
                "scalar_rectangle_vacuum_threshold",
                "scalar_three_bands_cap_compression",
            ];
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[2, 0, 1, 0, 3])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let statuses = located.status.try_to_vec::<i64>()?;
                let before = work.snapshot();
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(
                    session.invocation_rows().try_to_vec::<i64>()?,
                    [2, 0, 1, 0, 3]
                );
                assert_eq!(session.locator_status().try_to_vec::<i64>()?, statuses);
                assert_eq!(work.snapshot(), before);
                let owners = session
                    .domains()
                    .iter()
                    .map(|domain| domain.invocation_owner.try_to_vec::<i64>())
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                assert_eq!(owners.concat(), [1, 3]);
                Ok(())
            })
        }

        fn selected_caches_stay_fixed<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: DepthClient<R>,
        {
            let config = chart_fixture::config();
            let ids = ["retained_vacuum_inside_film", "scalar_rectangle_two_shocks"];
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let fixed = chart_fixture::cache(pair)?;
                assert!(fixed.iter().all(|entry| entry.ready[1] == 0.0));
                let before = work.snapshot();
                let session = hydraulic(refine_wet_depth(pair, located))?;
                assert_eq!(work.snapshot(), before);
                let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
                assert_eq!(chart_fixture::cache(session.sources())?, fixed);
                assert!(session.domains().iter().all(|domain| !domain.is_deferred()));
                Ok(())
            })
        }
    }

    mod partition {
        use super::{
            frozen::{self, Result, hydraulic},
            support::{self, DepthClient},
        };
        use crate::channel::flow::surveyed_points::{
            chart_fixture,
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
            let mut config = chart_fixture::config();
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
                let located_points = located.callback_points();
                let cache = chart_fixture::cache(pair)?;
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
                let initialized_points = initialized_points
                    .ok_or("Mixed deferred batch lacks its two-row continuation")?;
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
                assert_eq!(chart_fixture::cache(session.sources())?, cache);
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
                hydraulic(domain.check_invocation(&session.invocation))?;
                let parents = domain.parent_rows.try_to_vec::<i64>()?;
                let owners = domain.invocation_owner.try_to_vec::<i64>()?;
                let diagnostics = domain.query.query.original_rows.try_to_vec::<i64>()?;
                let area =
                    support::area_bounds(session.sources(), &domain.parent_rows, &domain.query)?;
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
    }
}
