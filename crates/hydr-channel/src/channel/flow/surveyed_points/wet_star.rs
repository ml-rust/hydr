use super::{
    paired_sources::PairedSources,
    source_binding::SourcePanelLevel,
    wet_depth_types::{WetDepthDomain, WetDepthSession},
    wet_star_types::WetStarDomain,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::Interval, integrate::CertifiedSimpsonAlgorithms};

impl<R> WetDepthSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    #[cfg(test)]
    pub(super) fn certify_wet_star(&mut self) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.wet_star_demand(None)
    }
    pub(super) fn certify_wet_star_selected(
        &mut self,
        demand: &Tensor<R>,
    ) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.wet_star_demand(Some(demand))
    }
    fn wet_star_demand(
        &mut self,
        demand: Option<&Tensor<R>>,
    ) -> HydraulicResult<Vec<WetStarDomain<R>>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet certificates remain sealed after an earlier numerical error",
            ));
        }
        self.check_domains()?;
        let c = self.sources.geometry.client();
        if let Some(mask) = demand {
            super::point_candidates_admission::check(self)?;
            checked::<R, _>(c, "wet star invocation demand admission", |a| {
                super::point_candidates_demand::admit(a, self, mask)
            })?;
        }
        let mut output = if demand.is_some() {
            Vec::new()
        } else {
            Vec::with_capacity(self.wet.domains.len())
        };
        for domain in &self.wet.domains {
            let mut selection = None;
            if let Some(mask) = demand {
                let active = c.index_select(mask, 0, &domain.invocation_owner)?;
                let rows = c.nonzero(&active)?;
                let rows = rows.reshape(&[rows.numel()])?;
                if rows.numel() == 0 {
                    continue;
                }
                if rows.numel() != domain.invocation_owner.numel() {
                    selection = Some(rows);
                }
            }
            if let [index] = domain.levels.policies() {
                let level = self.sources.panel_level(*index)?;
                output.push(star_group(
                    self.sources,
                    domain,
                    selection.as_ref(),
                    &level,
                )?);
                continue;
            }
            // Each output domain holds the rows of one panel level.
            let levels = match &selection {
                Some(rows) => domain.levels.selected(c, rows)?,
                None => domain.levels.clone(),
            };
            for index in levels.policies() {
                let group = levels.rows_at(c, *index)?;
                if group.numel() == 0 {
                    continue;
                }
                let rows = match &selection {
                    Some(rows) => c.index_select(rows, 0, &group)?,
                    None => group,
                };
                let level = self.sources.panel_level(*index)?;
                output.push(star_group(self.sources, domain, Some(&rows), &level)?);
            }
        }
        Ok(output)
    }
}

fn star_group<R>(
    sources: &mut PairedSources<'_, R>,
    domain: &WetDepthDomain<R>,
    rows: Option<&Tensor<R>>,
    level: &SourcePanelLevel,
) -> HydraulicResult<WetStarDomain<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let Some(rows) = rows else {
        return super::wet_candidates_star::evaluate(
            sources,
            &domain.invocation_owner,
            &domain.parent_rows,
            &domain.query,
            level,
        );
    };
    let c = sources.geometry.client();
    let (owners, parent, query) =
        checked::<R, _>(c, "wet star immutable selected operands", |a| {
            Ok((
                c.index_select(&domain.invocation_owner, 0, rows)?,
                c.index_select(&domain.parent_rows, 0, rows)?,
                domain.query.select(a, rows)?,
            ))
        })?;
    super::wet_candidates_star::evaluate(sources, &owners, &parent, &query, level)
}

pub(super) fn take_pair<T>(values: Vec<T>) -> HydraulicResult<[T; 2]> {
    values.try_into().map_err(|_: Vec<T>| {
        crate::channel::flow::surveyed::error::HydraulicError::Numerical {
            context: "Wet star side count differs from two. Rebuild the paired certificate".into(),
        }
    })
}

pub(super) fn ordered_bounds<R>(
    a: &mut Arithmetic<'_, R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    available: &Tensor<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let valid = c.le(lower, upper)?;
    let active = c.mul(available, &valid)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&active, lower, &zero)?,
            &c.where_cond(&active, upper, &zero)?,
        )?,
        valid,
    ))
}

pub(super) fn intersection<R>(
    a: &mut Arithmetic<'_, R>,
    first: &Interval<R>,
    second: &Interval<R>,
) -> HydraulicResult<(Interval<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let lower = c.maximum(first.lower(), second.lower())?;
    let upper = c.minimum(first.upper(), second.upper())?;
    let valid = c.le(&lower, &upper)?;
    let zero = Tensor::zeros(lower.shape(), a.dtype(), c.device())?;
    Ok((
        a.bounds(
            &c.where_cond(&valid, &lower, &zero)?,
            &c.where_cond(&valid, &upper, &zero)?,
        )?,
        valid,
    ))
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
    use crate::channel::flow::surveyed_points::{
        chart_fixture, source_fixture_support as frozen, wet_depth::refine_wet_depth,
        wet_roots::locate_wet_roots, wet_star_types::WetStarStatus,
    };
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};
    use solvr::optimize::CertifiedRootTolerance;

    use crate::channel::flow::surveyed_points::wet_depth::fixture as support;
    use diagnostics::check_diagnostics;

    fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        exact_rectangular_star::<R>(client)?;
        ranges::full_domain_discharge::<R>(client)?;
        ranges::empty_intersections::<R>(client)?;
        frozen_star_states::<R>(client, coverage)?;
        admission::run::<R>(client)
    }

    fn exact_rectangular_star<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_fixture::config();
        // Phi(A)=2*sqrt(A). Sources (4,-4),(4,12) give A*=1,u*=1,Q*=1.
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[4.0, -4.0, 4.0, 12.0], [1.0, 0.125, 1.0, 0.125]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[1, 0, 1, 0])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let points = session.callback_points();
                let statuses = session.locator_status().try_to_vec::<i64>()?;
                let mut identities = Vec::new();
                for domain in session.domains() {
                    identities.push((
                        domain.invocation_owner.try_to_vec::<i64>()?,
                        domain.parent_rows.try_to_vec::<i64>()?,
                        domain.query.query.original_rows.try_to_vec::<i64>()?,
                        frozen::endpoints(&domain.query.query.parameter)?,
                    ));
                }
                let before = work.snapshot();
                let certificates = hydraulic(session.certify_wet_star())?;
                let after = work.snapshot();
                assert_eq!(after.residual_points, before.residual_points);
                assert_eq!(after.root_iterations, before.root_iterations);
                assert_eq!(after.root_solves, before.root_solves);
                assert_eq!(session.callback_points(), points);
                assert_eq!(session.locator_status().try_to_vec::<i64>()?, statuses);
                assert_eq!(certificates.len(), identities.len());
                for (domain, (owners, parents, original_rows, parameter)) in
                    certificates.iter().zip(identities)
                {
                    check_diagnostics(session.sources(), domain)?;
                    assert_eq!(domain.invocation_owner.try_to_vec::<i64>()?, owners);
                    assert_eq!(domain.parent_rows.try_to_vec::<i64>()?, parents);
                    assert_eq!(
                        domain.query.query.original_rows.try_to_vec::<i64>()?,
                        original_rows
                    );
                    assert_eq!(frozen::endpoints(&domain.query.query.parameter)?, parameter);
                    assert!(
                        domain
                            .status
                            .try_to_vec::<i64>()?
                            .iter()
                            .all(|&x| x == WetStarStatus::Certified as i64)
                    );
                    let velocity = frozen::endpoints(&domain.velocity)?;
                    let discharge = frozen::endpoints(&domain.discharge)?;
                    for (index, parent) in parents.into_iter().enumerate() {
                        let expected = if parent == 0 { 1.0 } else { 0.125 };
                        assert!(velocity[index][0] <= expected && expected <= velocity[index][1]);
                        assert!(discharge[index][0] <= expected && expected <= discharge[index][1]);
                        if parent == 1 {
                            assert_eq!(discharge[index], [0.125; 2]);
                        }
                    }
                }
                let cache = chart_fixture::cache(session.sources())?;
                let repeated = hydraulic(session.certify_wet_star())?;
                assert_eq!(repeated.len(), certificates.len());
                assert_eq!(chart_fixture::cache(session.sources())?, cache);
                assert_eq!(work.snapshot(), after);
                // Certificate construction leaves continuation available for stricter physical stopping.
                let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                    atol: 0.0,
                    rtol: 0.0,
                }))?;
                assert!(session.callback_points() >= points);
                Ok(())
            },
        )
    }

    fn frozen_star_states<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        // references.json supplies new_independent_truth.star and weak-cap reused_truths_verbatim.truth.star.
        // Weak-cap rational star bounds project outward to these binary64 bounds. Film sampled flux is sonic.
        let truths = [
            (
                "retained_vacuum_inside_film",
                [0xbae0000000000000, 0xbae0000000000000],
                [0xb0a0000000000000, 0xb0a0000000000000],
            ),
            (
                "uniform_cap_retained_deficit_dyadic_velocity",
                [0x3fc0000000000000, 0x3fc0000000000000],
                [0x3ffbffffffffffff, 0x3ffc000000000000],
            ),
            (
                "trace_origin_after_retained_history_face_193",
                [0x3c8b706c91b2f8d9, 0x3c8b706c91b2f8da],
                [0x3cc8025eff7c99be, 0x3cc8025eff7c99bf],
            ),
            (
                "retained_knot_2_crossing",
                [0xbada20bd700c2c3e, 0xbada20bd700c2c3d],
                [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
            ),
        ];
        let config = chart_fixture::config();
        chart_fixture::with_pair::<R, _>(client, &truths.map(|t| t.0), &config, |pair, work| {
            // The device replays the knot jump and the film, one star of each kind.
            let parents: &[i64] = if coverage.is_full() {
                &[3, 0, 1, 3, 2]
            } else {
                &[3, 0]
            };
            let rows = support::rows::<R>(client, parents)?;
            let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
            let session = hydraulic(refine_wet_depth(pair, located))?;
            let mut session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                atol: 0.0,
                rtol: 0.0,
            }))?;
            let before = work.snapshot();
            let certificates = hydraulic(session.certify_wet_star())?;
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            let mut count = 0;
            for domain in &certificates {
                check_diagnostics(session.sources(), domain)?;
                diagnostics::check_frozen_differences(session.sources(), domain)?;
                let parents = domain.parent_rows.try_to_vec::<i64>()?;
                let velocity = frozen::endpoints(&domain.velocity)?;
                let discharge = frozen::endpoints(&domain.discharge)?;
                assert!(
                    domain
                        .status
                        .try_to_vec::<i64>()?
                        .iter()
                        .all(|&s| s == WetStarStatus::Certified as i64)
                );
                for (index, parent) in parents.into_iter().enumerate() {
                    let truth = truths[usize::try_from(parent)?];
                    for (actual, expected) in
                        [(velocity[index], truth.1), (discharge[index], truth.2)]
                    {
                        let expected = expected.map(f64::from_bits);
                        assert!(
                            actual[0] <= expected[0] && actual[1] >= expected[1],
                            "{}: {actual:?}, {expected:?}",
                            truth.0
                        );
                        if parent == 0 {
                            assert!(
                                actual[1] - actual[0] < expected[0].abs() * 1e-10,
                                "film precision: {actual:?}"
                            );
                        }
                    }
                    count += 1;
                }
            }
            assert_eq!(count, parents.len());
            Ok(())
        })
    }

    #[test]
    fn configured_cpu_wet_star_certificates_preserve_root_state_and_anchors() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client, Coverage::Full)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_wet_star_certificates_preserve_root_state_and_anchors() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }

    mod admission {
        use super::{
            frozen::{Result, hydraulic},
            support,
        };
        use crate::channel::flow::surveyed_points::{
            chart_fixture, wet_depth::refine_wet_depth, wet_roots::locate_wet_roots,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            unchanged_retained_source::<R>(client)?;
            let foreign = support::with_rectangle::<R, _>(
                client,
                1.0,
                8.0,
                &[[1.0, 0.0, 1.0, 0.0]],
                &config,
                |pair, _| hydraulic(pair.root_witness()),
            )?;
            for stale in [false, true] {
                support::with_rectangle::<R, _>(
                    client,
                    1.0,
                    8.0,
                    &[[1.0, 0.0, 1.0, 0.0]],
                    &config,
                    |pair, work| {
                        let rows = support::rows::<R>(client, &[0])?;
                        let located =
                            hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                        let mut session = hydraulic(refine_wet_depth(pair, located))?;
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(session.sources())?;
                        if stale {
                            session.sources.evaluation.panels *= 2;
                        } else {
                            session.invocation = hydraulic(
                                crate::channel::flow::surveyed_points::root_invocation::RootInvocation::new(
                                    foreign.clone(),
                                    session.point_limit(),
                                ),
                            )?;
                        }
                        assert!(session.certify_wet_star().is_err());
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                        Ok(())
                    },
                )?;
            }
            support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
                let rows = support::rows::<R>(client, &[])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                assert!(hydraulic(session.certify_wet_star())?.is_empty());
                assert_eq!(work.snapshot(), before);
                Ok(())
            })
        }

        fn unchanged_retained_source<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            use super::frozen::components;
            use crate::channel::{
                flow::surveyed_points::{potential_range::source_query, wet_star_endpoint},
                section::PhysicalEndpoint,
            };
            let config = chart_fixture::config();
            chart_fixture::with_pair::<R, _>(
                client,
                &["uniform_cap_retained_deficit_dyadic_velocity"],
                &config,
                |pair, _| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let query = hydraulic(source_query(
                        pair.geometry,
                        pair.bands,
                        pair.sources[0],
                        &pair.original_rows,
                    ))?;
                    let level = hydraulic(pair.panel_level(0))?;
                    let endpoint = hydraulic(wet_star_endpoint::evaluate(
                        pair,
                        &rows,
                        &query,
                        PhysicalEndpoint::Lower,
                        &level,
                    ))?;
                    for side in &endpoint.sides {
                        assert_eq!(side.unchanged.try_to_vec::<f64>()?, [1.0]);
                        assert_eq!(
                            components(&side.discharge)?,
                            components(&pair.velocity.input().left_discharge)?
                        );
                        for point in components(&side.area_difference)?
                            .into_iter()
                            .chain(components(&side.velocity_change)?)
                        {
                            assert!(point.map(f64::from_bits).iter().all(|&x| x == 0.0));
                        }
                    }
                    Ok(())
                },
            )
        }
    }

    mod diagnostics {
        use super::{frozen, support};
        use frozen::{Result, hydraulic};
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn check_diagnostics<R>(
            pair: &crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
            domain: &crate::channel::flow::surveyed_points::wet_star_types::WetStarDomain<R>,
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            use crate::channel::flow::surveyed::error::checked;
            hydraulic(checked(
                pair.geometry.client(),
                "wet star diagnostic identities",
                |a| {
                    let stage = pair.select_rows(&domain.parent_rows, &domain.query)?;
                    for endpoint in domain.endpoints.iter() {
                        let area = a.expansion_interval(&endpoint.area)?;
                        for (lower, upper) in area
                            .lower()
                            .try_to_vec::<f64>()?
                            .into_iter()
                            .zip(area.upper().try_to_vec::<f64>()?)
                        {
                            assert!(lower <= upper);
                        }
                        for (side, diagnostic) in endpoint.sides.iter().enumerate() {
                            let source_velocity = if side == 0 {
                                &stage.velocity.left
                            } else {
                                &stage.velocity.right
                            };
                            let source_discharge = if side == 0 {
                                &stage.velocity.input().left_discharge
                            } else {
                                &stage.velocity.input().right_discharge
                            };
                            let source_area = if side == 0 {
                                &stage.velocity.input().left_area
                            } else {
                                &stage.velocity.input().right_area
                            };
                            for (actual, expected) in [
                                (&diagnostic.source_area, source_area),
                                (&diagnostic.source_velocity, source_velocity),
                                (&diagnostic.source_discharge, source_discharge),
                            ] {
                                assert_eq!(
                                    actual.high().try_to_vec::<f64>()?,
                                    expected.high().try_to_vec::<f64>()?
                                );
                                assert_eq!(
                                    actual.low().try_to_vec::<f64>()?,
                                    expected.low().try_to_vec::<f64>()?
                                );
                                assert_eq!(
                                    actual.abs_error().try_to_vec::<f64>()?,
                                    expected.abs_error().try_to_vec::<f64>()?
                                );
                            }
                            for value in [
                                &diagnostic.area_difference,
                                &diagnostic.velocity_change,
                                &diagnostic.discharge_change,
                            ] {
                                assert!(
                                    value
                                        .abs_error()
                                        .try_to_vec::<f64>()?
                                        .iter()
                                        .all(|radius| radius.is_finite() && *radius >= 0.0)
                                );
                            }
                            let unchanged = diagnostic.unchanged.try_to_vec::<f64>()?;
                            let actual = diagnostic.discharge.high().try_to_vec::<f64>()?;
                            let captured =
                                diagnostic.source_discharge.high().try_to_vec::<f64>()?;
                            for (index, unchanged) in unchanged.into_iter().enumerate() {
                                if unchanged != 0.0 {
                                    assert_eq!(actual[index], captured[index]);
                                }
                            }
                        }
                    }
                    Ok(())
                },
            ))?;
            let velocity = frozen::endpoints(&domain.velocity)?;
            let discharge = frozen::endpoints(&domain.discharge)?;
            for side in 0..2 {
                let side_velocity = frozen::endpoints(&domain.side_velocity[side])?;
                let side_discharge = frozen::endpoints(&domain.side_discharge[side])?;
                for index in 0..velocity.len() {
                    assert!(
                        side_velocity[index][0] <= velocity[index][0]
                            && velocity[index][1] <= side_velocity[index][1]
                    );
                    assert!(
                        side_discharge[index][0] <= discharge[index][0]
                            && discharge[index][1] <= side_discharge[index][1]
                    );
                }
                let range = &domain.side_operands[side];
                for value in [
                    &range.area_difference,
                    &range.velocity_change,
                    &range.discharge_change,
                    &range.discharge,
                ] {
                    assert!(
                        value
                            .abs_error()
                            .try_to_vec::<f64>()?
                            .iter()
                            .all(|x| x.is_finite() && *x >= 0.0)
                    );
                }
                assert_eq!(
                    frozen::endpoints(&frozen::bounds(pair.geometry.client(), &range.discharge)?)?,
                    side_discharge
                );
            }
            Ok(())
        }

        pub(super) fn check_frozen_differences<R>(
            pair: &crate::channel::flow::surveyed_points::paired_sources::PairedSources<'_, R>,
            domain: &crate::channel::flow::surveyed_points::wet_star_types::WetStarDomain<R>,
        ) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            // Independent rational star bounds minus exact captured source high+low, projected outward.
            // Source rationals match references.json physical_input_verbatim. Parent order matches frozen_star_states.
            let differences: [[[[u64; 2]; 2]; 2]; 4] = [
                [
                    [
                        [0xbff0000000000000, 0xbfefffffffffffff],
                        [0x3fffffffffffffff, 0x4000000000000000],
                    ],
                    [
                        [0xbff0000000000000, 0xbfefffffffffffff],
                        [0xc000000000000000, 0xbfffffffffffffff],
                    ],
                ],
                [[[0, 0], [0, 0]], [[0, 0], [0, 0]]],
                [
                    [
                        [0xbcc2263845ea350e, 0xbcc2263845ea350d],
                        [0x3cc8025eff7c99be, 0x3cc8025eff7c99bf],
                    ],
                    [
                        [0xbc8207bfde3a47e7, 0xbc8207bfde3a47e6],
                        [0xbc87da100836641c, 0xbc87da100836641b],
                    ],
                ],
                [
                    [
                        [0x3aefffffffffffff, 0x3af0000000000000],
                        [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
                    ],
                    [
                        [0xbaf0000000000001, 0xbaf0000000000000],
                        [0xbaea20bd700c2c3e, 0xbaea20bd700c2c3d],
                    ],
                ],
            ];
            let parents = domain.parent_rows.try_to_vec::<i64>()?;
            for (side, operands) in domain.side_operands.iter().enumerate() {
                let area = frozen::endpoints(&frozen::bounds(
                    pair.geometry.client(),
                    &operands.area_difference,
                )?)?;
                let discharge = frozen::endpoints(&frozen::bounds(
                    pair.geometry.client(),
                    &operands.discharge_change,
                )?)?;
                for (row, &parent) in parents.iter().enumerate() {
                    let truth = differences[usize::try_from(parent)?][side];
                    for (field, (actual, expected)) in
                        [(area[row], truth[0]), (discharge[row], truth[1])]
                            .into_iter()
                            .enumerate()
                    {
                        let expected = expected.map(f64::from_bits);
                        assert!(
                            actual[0] <= expected[0] && expected[1] <= actual[1],
                            "parent={parent}, side={side}, difference={field}: {actual:?} excludes {expected:?}"
                        );
                    }
                }
            }
            Ok(())
        }
    }

    mod ranges {
        use super::{
            Result, WetStarStatus, chart_fixture, frozen, hydraulic, locate_wet_roots,
            refine_wet_depth, support,
        };
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn full_domain_discharge<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            use crate::channel::flow::surveyed_points::wet_depth_types::WetDepthStatus;
            use numr::{runtime::RuntimeClient, tensor::Tensor};
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                8.0,
                &[[4.0, -4.0, 4.0, 12.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    // Widen the supported bracket around the independently known root A*=1.
                    let query = chart_fixture::chart(
                        session.sources(),
                        chart_fixture::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.25, 2.25]],
                    )?;
                    session.wet.domains[0].query = query;
                    session.wet.domains[0].status = Tensor::full_scalar(
                        &[1],
                        DType::I64,
                        WetDepthStatus::WorkLimit as i64 as f64,
                        client.device(),
                    )?;
                    let before = work.snapshot();
                    let certificates = hydraulic(session.certify_wet_star())?;
                    let star = &certificates[0];
                    assert_eq!(
                        star.status.try_to_vec::<i64>()?,
                        [WetStarStatus::Certified as i64]
                    );
                    let range = frozen::endpoints(&star.side_discharge[0])?[0];
                    assert!(range[0] <= 1.0 && range[1] >= 1.0);
                    // QL(A)=A*(3-2*sqrt(A)) peaks at the root, above both endpoint values.
                    let lower = frozen::endpoints(&frozen::bounds(
                        client,
                        &star.endpoints[0].sides[0].discharge,
                    )?)?[0];
                    let upper = frozen::endpoints(&frozen::bounds(
                        client,
                        &star.endpoints[1].sides[0].discharge,
                    )?)?[0];
                    assert!(lower[1] < 1.0 && upper[1] < 1.0);
                    assert_eq!(
                        session.domains()[0].status.try_to_vec::<i64>()?,
                        [WetDepthStatus::WorkLimit as i64]
                    );
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }

        pub(super) fn empty_intersections<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            use crate::channel::flow::surveyed::error::checked;
            use numr::{runtime::RuntimeClient, tensor::Tensor};
            hydraulic(checked::<R, _>(
                client,
                "wet star unresolved and unavailable bounds",
                |a| {
                    let zero = Tensor::zeros(&[1], a.dtype(), client.device())?;
                    let one = Tensor::ones(&[1], a.dtype(), client.device())?;
                    let first = a.point(&zero)?;
                    let second = a.point(&one)?;
                    let (value, valid) = super::super::intersection(a, &first, &second)?;
                    assert_eq!(valid.try_to_vec::<f64>()?, [0.0]);
                    assert_eq!(value.lower().try_to_vec::<f64>()?, [0.0]);
                    let (value, valid) = super::super::ordered_bounds(a, &one, &zero, &zero)?;
                    assert_eq!(valid.try_to_vec::<f64>()?, [0.0]);
                    assert_eq!(value.upper().try_to_vec::<f64>()?, [0.0]);
                    Ok(())
                },
            ))
        }
    }
}
