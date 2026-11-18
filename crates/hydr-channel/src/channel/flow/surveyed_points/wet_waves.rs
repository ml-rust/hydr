use super::{
    wet_depth_types::WetDepthSession,
    wet_star_types::WetStarStatus,
    wet_wave_types::{WetWaveDomain, WetWaveStatus},
    wet_waves_endpoint, wet_waves_regions, wet_waves_side,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

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
    /// Retain possible wet-wave positions without selecting a trace or accepting a stationary discontinuity.
    #[cfg(test)]
    pub(super) fn certify_wet_waves(&mut self) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet certificates remain sealed after an earlier numerical error",
            ));
        }
        self.check_domains()?;
        let stars = self.certify_wet_star()?;
        self.wet_waves_from(stars)
    }

    pub(super) fn certify_wet_waves_selected(
        &mut self,
        demand: &Tensor<R>,
    ) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        let stars = self.certify_wet_star_selected(demand)?;
        self.wet_waves_from(stars)
    }

    fn wet_waves_from(
        &mut self,
        stars: Vec<super::wet_star_types::WetStarDomain<R>>,
    ) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        let c = self.sources.geometry.client();
        let mut result = Vec::with_capacity(stars.len());
        for star in stars {
            let domain = checked(c, "wet wave positions over established root domain", |a| {
                let selected = self.sources.select_rows(&star.parent_rows, &star.query)?;
                let endpoint_sound = wet_waves_endpoint::sounds(a, self.sources, &selected, &star)?;
                let left = wet_waves_side::evaluate(
                    a,
                    &selected,
                    &star,
                    &endpoint_sound,
                    self.sources.evaluation.config.gravity_m_per_s2,
                    0,
                )?;
                let right = wet_waves_side::evaluate(
                    a,
                    &selected,
                    &star,
                    &endpoint_sound,
                    self.sources.evaluation.config.gravity_m_per_s2,
                    1,
                )?;
                let sides = [left, right];
                let mut regions = wet_waves_regions::evaluate(a, &sides)?;
                let covered = c.maximum(
                    &regions.left_exterior,
                    &c.maximum(
                        &regions.left_fan,
                        &c.maximum(
                            &regions.star,
                            &c.maximum(&regions.right_fan, &regions.right_exterior)?,
                        )?,
                    )?,
                )?;
                let complete = c.mul(&covered, &c.mul(&sides[0].valid, &sides[1].valid)?)?;
                let zero = Tensor::zeros(complete.shape(), a.dtype(), c.device())?;
                let status_value = |value: i64| {
                    Tensor::full_scalar(complete.shape(), DType::I64, value as f64, c.device())
                };
                let unavailable = c.eq(
                    &star.status,
                    &status_value(WetStarStatus::Unavailable as i64)?,
                )?;
                let status = c.where_cond(
                    &unavailable,
                    &status_value(WetWaveStatus::Unavailable as i64)?,
                    &c.where_cond(
                        &complete,
                        &status_value(WetWaveStatus::Certified as i64)?,
                        &status_value(WetWaveStatus::Unresolved as i64)?,
                    )?,
                )?;
                for mask in [
                    &mut regions.left_exterior,
                    &mut regions.left_fan,
                    &mut regions.star,
                    &mut regions.right_fan,
                    &mut regions.right_exterior,
                ] {
                    *mask = c.where_cond(&complete, mask, &zero)?;
                }
                Ok(WetWaveDomain {
                    star,
                    endpoint_sound,
                    sides,
                    regions,
                    status,
                })
            })?;
            result.push(domain);
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
    use crate::channel::flow::surveyed_points::{
        chart_fixture, source_fixture_support as frozen,
        wet_depth::fixture as support,
        wet_depth::refine_wet_depth,
        wet_roots::locate_wet_roots,
        wet_wave_types::{WetWaveDomain, WetWaveStatus},
    };
    use frozen::{Result, hydraulic};
    use numr::{dtype::DType, runtime::Runtime};

    fn matrix<R>(client: &R::Client, coverage: Coverage) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        references::run::<R>(client, coverage)?;
        controls::run::<R>(client)?;
        lifecycle::run::<R>(client)?;
        support::with_rectangle::<R, _>(
            client,
            1.0,
            8.0,
            &[[1.0, 1.0, 1.0, 1.0]],
            &chart_fixture::config(),
            |pair, work| {
                let rows = support::rows::<R>(client, &[0])?;
                let located = hydraulic(locate_wet_roots(
                    pair,
                    &rows,
                    pair.evaluation.config.root.max_points,
                ))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                let waves = hydraulic(session.certify_wet_waves())?;
                assert_eq!(work.snapshot(), before);
                let wave = &waves[0];
                assert_eq!(
                    wave.status.try_to_vec::<i64>()?,
                    [WetWaveStatus::Certified as i64]
                );
                assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [0.0]);
                assert_eq!(wave.regions.right_fan.try_to_vec::<f64>()?, [0.0]);
                Ok(())
            },
        )
    }

    #[test]
    fn configured_cpu_wet_wave_certificates_preserve_positions_and_continuation() -> Result {
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
    fn actual_cuda_wet_wave_certificates_preserve_positions_and_continuation() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?, Coverage::Sample)
    }

    mod controls {
        use super::{
            Result, chart_fixture, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
        };
        use crate::channel::{
            flow::surveyed::error::checked,
            section::{AreaAnchor, AreaDirection, AreaQuery},
        };
        use numr::{
            dtype::DType,
            ops::BinaryOps,
            runtime::{Runtime, RuntimeClient},
            tensor::Tensor,
        };

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            broad_uniform::<R>(client)?;
            source_crossing::<R>(client, false)?;
            source_crossing::<R>(client, true)
        }

        fn broad_uniform<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.0,
                8.0,
                &[[1.0, 1.0, 1.0, 1.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    // Exact uniform identity proves A*=1 independently of this wider supported bracket.
                    session.wet.domains[0].query = chart_fixture::chart(
                        session.sources(),
                        chart_fixture::Anchor::Zero,
                        [1.0_f64.to_bits(), 0],
                        &[[0.25, 2.25]],
                    )?;
                    let before = work.snapshot();
                    let waves = hydraulic(session.certify_wet_waves())?;
                    let wave = &waves[0];
                    super::diagnostics::check(wave)?;
                    assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [0.0]);
                    assert_eq!(wave.regions.right_fan.try_to_vec::<f64>()?, [0.0]);
                    // The independent exact characteristic is zero. Certified uncertainty preserves both adjacent candidates.
                    let head = frozen::endpoints(&wave.sides[0].head.physical)?[0];
                    assert!(head[0] <= 0.0 && head[1] >= 0.0);
                    for side in &wave.sides {
                        assert_eq!(side.equality_possible.try_to_vec::<f64>()?, [1.0]);
                        assert_eq!(side.shock_possible.try_to_vec::<f64>()?, [0.0]);
                        assert_eq!(side.rarefaction_possible.try_to_vec::<f64>()?, [0.0]);
                    }
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }

        fn source_crossing<R>(client: &R::Client, uncertain_endpoint: bool) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            // Rectangle Phi(A)=2*sqrt(A). These distinct sources give the exact root A*=1,u*=1.
            support::with_rectangle::<R, _>(
                client,
                1.0,
                8.0,
                &[[1.0, 1.0, 4.0, 12.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    let query = if uncertain_endpoint {
                        hydraulic(checked::<R, _>(
                            client,
                            "supported uncertain source-crossing wave control",
                            |a| {
                                let pair = session.sources();
                                let zero = Tensor::zeros(&[1], a.dtype(), client.device())?;
                                let two =
                                    Tensor::full_scalar(&[1], a.dtype(), 2.0, client.device())?;
                                let low = Tensor::full_scalar(
                                    &[1],
                                    a.dtype(),
                                    -f64::from_bits(1),
                                    client.device(),
                                )?;
                                let half =
                                    Tensor::full_scalar(&[1], a.dtype(), 0.5, client.device())?;
                                let upper =
                                    Tensor::full_scalar(&[1], a.dtype(), 1.125, client.device())?;
                                // Exact scale times one half gives 1-2^-1075. Only the endpoint transform acquires uncertainty.
                                let scale = a.expansion_bounds(&two, &low, &zero)?;
                                let parameter = a.bounds(&half, &upper)?;
                                let query = AreaQuery::band_affine(
                                    a,
                                    pair.bands,
                                    AreaAnchor::Zero,
                                    AreaDirection::Add,
                                    &scale,
                                    &parameter,
                                )?;
                                client.copy_into(&query.original_rows, &pair.original_rows)?;
                                query.prove_support(a, pair.bands)
                            },
                        ))?
                    } else {
                        chart_fixture::chart(
                            session.sources(),
                            chart_fixture::Anchor::Zero,
                            [1.0_f64.to_bits(), 0],
                            &[[0.25, 2.25]],
                        )?
                    };
                    session.wet.domains[0].query = query;
                    let before = work.snapshot();
                    let waves = hydraulic(session.certify_wet_waves())?;
                    let wave = &waves[0];
                    super::diagnostics::check(wave)?;
                    let left = &wave.sides[0];
                    assert_eq!(left.shock_possible.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(left.shock_actual.try_to_vec::<f64>()?, [0.0]);
                    assert_eq!(left.shock_conditional.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(left.rarefaction_possible.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(left.equality_possible.try_to_vec::<f64>()?, [1.0]);
                    // Independent source sound equals one. The retained source certificate supplies this lower limit.
                    let source_sound = frozen::endpoints(&left.source_sound.physical)?[0];
                    assert!(source_sound[0] <= 1.0 && source_sound[1] >= 1.0);
                    let retained_sound =
                        frozen::endpoints(&frozen::bounds(client, &left.source_sound.retained)?)?
                            [0];
                    assert!(retained_sound[0] <= 1.0 && retained_sound[1] >= 1.0);
                    let offset = frozen::endpoints(&left.shock_offset.physical)?[0];
                    assert_eq!(offset[0], retained_sound[0]);
                    assert!(offset[0] <= 1.0 && offset[1] > 1.0);
                    let shock = frozen::endpoints(&left.shock_speed.physical)?[0];
                    assert!(shock[0] < 0.0 && shock[1] >= 0.0);
                    assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
                    if uncertain_endpoint {
                        assert_eq!(
                            wave.star
                                .query
                                .query
                                .scale
                                .abs_error()
                                .try_to_vec::<f64>()?,
                            [0.0]
                        );
                        assert!(
                            wave.star.endpoints[0]
                                .area
                                .abs_error()
                                .try_to_vec::<f64>()?[0]
                                > 0.0
                        );
                        let evidence = wave.star.endpoints[0].waves.local[0]
                            .as_ref()
                            .ok_or("Missing conditional lower endpoint")?;
                        assert_eq!(
                            evidence
                                .endpoint
                                .actual_shock
                                .value
                                .available
                                .try_to_vec::<f64>()?,
                            [0.0]
                        );
                        assert_eq!(
                            evidence
                                .endpoint
                                .conditional_shock
                                .value
                                .available
                                .try_to_vec::<f64>()?,
                            [1.0]
                        );
                    }
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                    Ok(())
                },
            )
        }
    }

    mod diagnostics {
        use super::{Result, WetWaveDomain, frozen, support};
        use numr::{dtype::DType, runtime::Runtime};

        pub(super) fn check<R>(domain: &WetWaveDomain<R>) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let n = domain.star.parent_rows.numel();
            assert_eq!(domain.star.invocation_owner.numel(), n);
            assert_eq!(domain.star.query.query.original_rows.numel(), n);
            for mask in [
                &domain.regions.left_exterior,
                &domain.regions.left_fan,
                &domain.regions.star,
                &domain.regions.right_fan,
                &domain.regions.right_exterior,
            ] {
                assert_eq!(mask.numel(), n);
                assert!(
                    mask.try_to_vec::<f64>()?
                        .iter()
                        .all(|&x| x == 0.0 || x == 1.0)
                );
            }
            for sound in &domain.endpoint_sound {
                for bounds in frozen::endpoints(&sound.physical)? {
                    assert!(0.0 <= bounds[0] && bounds[0] <= bounds[1]);
                }
                assert!(
                    sound
                        .retained
                        .abs_error()
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&x| x.is_finite() && x >= 0.0)
                );
            }
            for side in &domain.sides {
                for mask in [
                    &side.shock_possible,
                    &side.rarefaction_possible,
                    &side.equality_possible,
                    &side.shock_actual,
                    &side.shock_conditional,
                    &side.shock_stationary,
                    &side.equality_stationary,
                    &side.valid,
                ] {
                    assert!(
                        mask.try_to_vec::<f64>()?
                            .iter()
                            .all(|&x| x == 0.0 || x == 1.0)
                    );
                }
                let actual = side.shock_actual.try_to_vec::<f64>()?;
                let conditional = side.shock_conditional.try_to_vec::<f64>()?;
                let possible = side.shock_possible.try_to_vec::<f64>()?;
                for row in 0..n {
                    assert!(actual[row] + conditional[row] <= possible[row]);
                }
                for value in [
                    &side.source_sound,
                    &side.shock_offset,
                    &side.shock_speed,
                    &side.head,
                    &side.tail,
                ] {
                    let available = value.available.try_to_vec::<f64>()?;
                    let bounds = frozen::endpoints(&value.physical)?;
                    for (row, bounds) in bounds.into_iter().enumerate() {
                        assert!(
                            bounds[0].is_finite()
                                && bounds[1].is_finite()
                                && bounds[0] <= bounds[1]
                        );
                        assert!(available[row] == 0.0 || available[row] == 1.0);
                    }
                    assert!(
                        value
                            .retained
                            .abs_error()
                            .try_to_vec::<f64>()?
                            .iter()
                            .all(|&x| x.is_finite() && x >= 0.0)
                    );
                }
                for (mask, speed) in [
                    (&side.shock_stationary, &side.shock_speed),
                    (&side.equality_stationary, &side.head),
                ] {
                    let stationary = mask.try_to_vec::<f64>()?;
                    let available = speed.available.try_to_vec::<f64>()?;
                    let bounds = frozen::endpoints(&speed.physical)?;
                    let retained = frozen::components(&speed.retained)?;
                    for row in 0..n {
                        if stationary[row] != 0.0 {
                            assert_eq!(available[row], 1.0);
                            assert_eq!(bounds[row], [0.0, 0.0]);
                            assert!(retained[row].map(f64::from_bits).iter().all(|&x| x == 0.0));
                        }
                    }
                }
            }
            for endpoint in &domain.star.endpoints {
                for side in 0..2 {
                    let far = endpoint.waves.far_rarefaction[side].try_to_vec::<f64>()?;
                    if let Some(local) = &endpoint.waves.local[side] {
                        for &row in &local.rows.try_to_vec::<i64>()? {
                            assert_eq!(far[usize::try_from(row)?], 0.0);
                        }
                        let classification = &local.endpoint.classification;
                        assert!(
                            classification
                                .dry_source
                                .try_to_vec::<f64>()?
                                .iter()
                                .all(|&v| v == 0.0)
                        );
                        for mask in [
                            &classification.shock,
                            &classification.rarefaction,
                            &classification.equality,
                            &classification.uncertain,
                        ] {
                            assert_eq!(mask.numel(), local.rows.numel());
                        }
                        assert_eq!(
                            local.endpoint.equality_limit.available.numel(),
                            local.rows.numel()
                        );
                    }
                }
            }
            Ok(())
        }
    }

    mod references {
        use super::{
            Result, chart_fixture, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
        };
        use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
        use crate::channel::flow::surveyed_points::{
            wave_curve_types::CurveValue,
            wet_wave_types::{WetWaveDomain, WetWaveStatus},
        };
        use numr::{dtype::DType, runtime::Runtime};

        #[derive(Clone, Copy)]
        enum Edge {
            Shock(usize),
            Head(usize),
            Tail(usize),
        }

        type EdgeReference = (Edge, [u64; 2]);
        type CaseReference<'a> = (&'a str, &'a [EdgeReference]);

        pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            // references.json new_independent_truth.wave_edges, ordered by reference_sampling.py append semantics.
            let references: &[CaseReference<'_>] = &[
                (
                    "retained_vacuum_inside_film",
                    &[
                        (Edge::Head(0), [0xc008000000000000; 2]),
                        (Edge::Tail(0), [0xbae8000000000000; 2]),
                        (Edge::Head(1), [0x4007ffffffffffff, 0x4008000000000000]),
                        (Edge::Tail(1), [0xbad0000000000000; 2]),
                    ],
                ),
                (
                    "retained_knot_2_crossing",
                    &[
                        (Edge::Shock(0), [0xbfea20bd700c2c3e, 0xbfea20bd700c2c3d]),
                        (Edge::Head(1), [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e]),
                        (Edge::Tail(1), [0x3fea20bd700c2c3d, 0x3fea20bd700c2c3e]),
                    ],
                ),
                (
                    "uniform_cap_retained_deficit_dyadic_velocity",
                    &[
                        (Edge::Head(0), [0xbff32a7fa9d2f8ea, 0xbff32a7fa9d2f8e9]),
                        (Edge::Head(1), [0x3ff72a7fa9d2f8e9, 0x3ff72a7fa9d2f8ea]),
                    ],
                ),
                (
                    "tiny_rectangle_compression",
                    &[
                        (Edge::Shock(0), [0x4011ffffffffffff, 0x4012000000000001]),
                        (Edge::Shock(1), [0x4011ffffffffffff, 0x4012000000000001]),
                    ],
                ),
                // reused_truths_verbatim.truth.wave_edges contains independent weak-cap rational intervals.
                (
                    "trace_origin_after_retained_history_face_193",
                    &[
                        (Edge::Head(0), [0xbff52a7fa9d2f8ea, 0xbff52a7fa9d2f8e9]),
                        (Edge::Tail(0), [0xbff52a7fa9d2f8ea, 0xbff52a7fa9d2f8e9]),
                        (Edge::Head(1), [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea]),
                        (Edge::Tail(1), [0x3ff52a7fa9d2f8e9, 0x3ff52a7fa9d2f8ea]),
                    ],
                ),
            ];
            let config = chart_fixture::config();
            let ids = references.iter().map(|r| r.0).collect::<Vec<_>>();
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                // The device skips the two-shock compression. The knot crossing reaches its shock path.
                let parents: &[i64] = if coverage.is_full() {
                    &[3, 0, 4, 2, 1, 0]
                } else {
                    &[0, 4, 2, 1, 0]
                };
                let rows = support::rows::<R>(client, parents)?;
                let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let callbacks = session.callback_points();
                let before = work.snapshot();
                let waves = hydraulic(session.certify_wet_waves())?;
                assert_eq!(work.snapshot().residual_points, before.residual_points);
                assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                assert_eq!(work.snapshot().root_solves, before.root_solves);
                assert_eq!(session.callback_points(), callbacks);
                let mut count = 0;
                for domain in &waves {
                    super::diagnostics::check(domain)?;
                    let parents = domain.star.parent_rows.try_to_vec::<i64>()?;
                    assert!(
                        domain
                            .status
                            .try_to_vec::<i64>()?
                            .iter()
                            .all(|&v| v == WetWaveStatus::Certified as i64)
                    );
                    for (index, parent) in parents.into_iter().enumerate() {
                        let parent = usize::try_from(parent)?;
                        for &(edge, reference) in references[parent].1 {
                            let value = edge_value(domain, edge);
                            assert_eq!(value.available.try_to_vec::<f64>()?[index], 1.0);
                            let actual = frozen::endpoints(&value.physical)?[index];
                            let expected = reference.map(f64::from_bits);
                            assert!(
                                actual[0] <= expected[0] && expected[1] <= actual[1],
                                "{}: {actual:?} excludes {expected:?}",
                                references[parent].0
                            );
                        }
                        match parent {
                            0 => assert_eq!(
                                domain.regions.right_fan.try_to_vec::<f64>()?[index],
                                1.0
                            ),
                            1 | 2 | 4 => {
                                assert_eq!(domain.regions.star.try_to_vec::<f64>()?[index], 1.0)
                            }
                            3 => assert_eq!(
                                domain.regions.left_exterior.try_to_vec::<f64>()?[index],
                                1.0
                            ),
                            _ => return Err("Unexpected frozen wave parent".into()),
                        }
                        match parent {
                            1 => {
                                assert!(
                                    frozen::endpoints(&domain.sides[0].shock_speed.physical)?
                                        [index][1]
                                        < 0.0
                                );
                                assert!(
                                    frozen::endpoints(&domain.sides[1].head.physical)?[index][0]
                                        > 0.0
                                );
                                assert!(
                                    frozen::endpoints(&domain.sides[1].tail.physical)?[index][0]
                                        > 0.0
                                );
                                unique_region(domain, index, 2)?;
                            }
                            2 => {
                                assert!(
                                    frozen::endpoints(&domain.sides[0].head.physical)?[index][1]
                                        < 0.0
                                );
                                assert!(
                                    frozen::endpoints(&domain.sides[1].head.physical)?[index][0]
                                        > 0.0
                                );
                                unique_region(domain, index, 2)?;
                            }
                            3 => {
                                for side in &domain.sides {
                                    assert!(
                                        frozen::endpoints(&side.shock_speed.physical)?[index][0]
                                            > 0.0
                                    );
                                }
                                unique_region(domain, index, 0)?;
                            }
                            _ => {}
                        }
                        if parent == 2 {
                            for side in &domain.sides {
                                assert_eq!(side.equality_possible.try_to_vec::<f64>()?[index], 1.0);
                                assert_eq!(
                                    side.rarefaction_possible.try_to_vec::<f64>()?[index],
                                    0.0
                                );
                                assert_eq!(side.shock_possible.try_to_vec::<f64>()?[index], 0.0);
                            }
                            assert_eq!(domain.regions.left_fan.try_to_vec::<f64>()?[index], 0.0);
                            assert_eq!(domain.regions.right_fan.try_to_vec::<f64>()?[index], 0.0);
                        }
                        count += 1;
                    }
                }
                assert_eq!(count, parents.len());
                Ok(())
            })?;
            stationary::<R>(client)?;
            separated_film::<R>(client)
        }

        fn edge_value<R: Runtime<DType = DType>>(
            domain: &WetWaveDomain<R>,
            edge: Edge,
        ) -> &CurveValue<R> {
            match edge {
                Edge::Shock(side) => &domain.sides[side].shock_speed,
                Edge::Head(side) => &domain.sides[side].head,
                Edge::Tail(side) => &domain.sides[side].tail,
            }
        }

        fn stationary<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.gravity_m_per_s2 = 3.0;
            chart_fixture::with_pair::<R, _>(
                client,
                &["scalar_rectangle_stationary_shock"],
                &config,
                |pair, _| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    let waves = hydraulic(session.certify_wet_waves())?;
                    let wave = &waves[0];
                    super::diagnostics::check(wave)?;
                    // reused_truths_verbatim.truth.wave_edges_m_per_s: exact left shock zero, then right equality characteristic.
                    let shock = frozen::endpoints(&wave.sides[0].shock_speed.physical)?[0];
                    assert!(shock[0] <= 0.0 && shock[1] >= 0.0);
                    let head = frozen::endpoints(&wave.sides[1].head.physical)?[0];
                    assert!(
                        head[0] <= f64::from_bits(0x400f988e1409212e)
                            && head[1] >= f64::from_bits(0x400f988e1409212f)
                    );
                    assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.0]);
                    assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [0.0]);
                    Ok(())
                },
            )
        }

        fn unique_region<R: Runtime<DType = DType>>(
            domain: &WetWaveDomain<R>,
            row: usize,
            expected: usize,
        ) -> Result {
            for (region, mask) in [
                &domain.regions.left_exterior,
                &domain.regions.left_fan,
                &domain.regions.star,
                &domain.regions.right_fan,
                &domain.regions.right_exterior,
            ]
            .into_iter()
            .enumerate()
            {
                assert_eq!(
                    mask.try_to_vec::<f64>()?[row],
                    if region == expected { 1.0 } else { 0.0 }
                );
            }
            Ok(())
        }

        fn separated_film<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            use solvr::optimize::CertifiedRootTolerance;
            let config = chart_fixture::config();
            chart_fixture::with_pair::<R, _>(
                client,
                &["retained_vacuum_inside_film"],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let session = hydraulic(refine_wet_depth(pair, located))?;
                    // A finite positive depth allowance resolves the independent film's right-fan sign margins.
                    let mut session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 2.0_f64.powi(-200),
                        rtol: 0.0,
                    }))?;
                    let before = work.snapshot();
                    let waves = hydraulic(session.certify_wet_waves())?;
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                    let wave = &waves[0];
                    super::diagnostics::check(wave)?;
                    assert_eq!(
                        wave.status.try_to_vec::<i64>()?,
                        [WetWaveStatus::Certified as i64]
                    );
                    let edges = [
                        (&wave.sides[0].head, 0xc008000000000000),
                        (&wave.sides[0].tail, 0xbae8000000000000),
                        (&wave.sides[1].tail, 0xbad0000000000000),
                    ];
                    for (value, bits) in edges {
                        let bounds = frozen::endpoints(&value.physical)?[0];
                        let expected = f64::from_bits(bits);
                        assert!(bounds[0] <= expected && expected <= bounds[1]);
                        assert!(bounds[1] < 0.0);
                    }
                    let head = frozen::endpoints(&wave.sides[1].head.physical)?[0];
                    assert!(
                        head[0] <= f64::from_bits(0x4007ffffffffffff)
                            && f64::from_bits(0x4008000000000000) <= head[1]
                    );
                    assert!(head[0] > 0.0);
                    unique_region(wave, 0, 3)
                },
            )
        }
    }

    mod lifecycle {
        use super::{
            Result, chart_fixture, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
        };
        use numr::{dtype::DType, runtime::Runtime};
        use solvr::optimize::CertifiedRootTolerance;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            witness_precedes_work::<R>(client)?;
            identities_and_work::<R>(client)?;
            original_budget::<R>(client)?;
            empty::<R>(client)
        }

        fn witness_precedes_work<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let foreign = support::with_rectangle::<R, _>(
                client,
                1.0,
                8.0,
                &[[1.0, 1.0, 1.0, 1.0]],
                &config,
                |pair, _| hydraulic(pair.root_witness()),
            )?;
            for stale in [false, true] {
                support::with_rectangle::<R, _>(
                    client,
                    1.0,
                    8.0,
                    &[[1.0, 1.0, 1.0, 1.0]],
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
                        assert!(session.certify_wet_waves().is_err());
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }

        fn identities_and_work<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.root.atol = 1e-4;
            config.root.rtol = 0.0;
            chart_fixture::with_pair::<R, _>(
                client,
                &[
                    "scalar_rectangle_depth_jump",
                    "retained_vacuum_inside_film",
                    "uniform_cap_retained_deficit_dyadic_velocity",
                ],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[2, 0, 2, 1])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, config.root.max_points))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    let locator_status = session.locator_status().try_to_vec::<i64>()?;
                    let points = session.callback_points();
                    let limit = session.point_limit();
                    let identities = session
                        .domains()
                        .iter()
                        .map(|d| {
                            Ok((
                                d.invocation_owner.try_to_vec::<i64>()?,
                                d.parent_rows.try_to_vec::<i64>()?,
                                d.query.query.original_rows.try_to_vec::<i64>()?,
                                frozen::endpoints(&d.query.query.parameter)?,
                                d.iterations,
                                d.callback_points,
                                d.mapped_points,
                            ))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    // Warm source caches, then measure exactly one star evaluation as the hydraulic-work baseline.
                    hydraulic(session.certify_wet_star())?;
                    let before_star = work.snapshot();
                    hydraulic(session.certify_wet_star())?;
                    let after_star = work.snapshot();
                    let waves = hydraulic(session.certify_wet_waves())?;
                    let after_wave = work.snapshot();
                    assert_eq!(
                        after_wave.simpson_value_points - after_star.simpson_value_points,
                        after_star.simpson_value_points - before_star.simpson_value_points
                    );
                    assert_eq!(
                        after_wave.simpson_derivative_points - after_star.simpson_derivative_points,
                        after_star.simpson_derivative_points
                            - before_star.simpson_derivative_points
                    );
                    assert_eq!(after_wave.residual_points, after_star.residual_points);
                    assert_eq!(after_wave.root_iterations, after_star.root_iterations);
                    assert_eq!(after_wave.root_solves, after_star.root_solves);
                    assert_eq!(session.callback_points(), points);
                    assert_eq!(session.point_limit(), limit);
                    assert_eq!(
                        session.locator_status().try_to_vec::<i64>()?,
                        locator_status
                    );
                    assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [2, 0, 2, 1]);
                    for ((wave, domain), identity) in
                        waves.iter().zip(session.domains()).zip(identities)
                    {
                        assert_eq!(wave.star.invocation_owner.try_to_vec::<i64>()?, identity.0);
                        assert_eq!(wave.star.parent_rows.try_to_vec::<i64>()?, identity.1);
                        assert_eq!(
                            wave.star.query.query.original_rows.try_to_vec::<i64>()?,
                            identity.2
                        );
                        assert_eq!(
                            frozen::endpoints(&wave.star.query.query.parameter)?,
                            identity.3
                        );
                        assert_eq!(
                            (
                                domain.iterations,
                                domain.callback_points,
                                domain.mapped_points
                            ),
                            (identity.4, identity.5, identity.6)
                        );
                    }
                    let cache = chart_fixture::cache(session.sources())?;
                    let repeated = hydraulic(session.certify_wet_waves())?;
                    assert_eq!(repeated.len(), waves.len());
                    assert_eq!(chart_fixture::cache(session.sources())?, cache);
                    let session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 1e-7,
                        rtol: 0.0,
                    }))?;
                    assert!(session.callback_points() >= points);
                    assert_eq!(session.point_limit(), limit);
                    Ok(())
                },
            )
        }

        fn original_budget<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let mut config = chart_fixture::config();
            config.max_native_query_points = 1;
            support::with_rectangle::<R, _>(
                client,
                3.0,
                1.0,
                &[[1.0, 1.0, 1.0, 1.0]],
                &config,
                |pair, work| {
                    let rows = support::rows::<R>(client, &[0])?;
                    let located = hydraulic(locate_wet_roots(pair, &rows, 1))?;
                    let mut session = hydraulic(refine_wet_depth(pair, located))?;
                    let before = work.snapshot();
                    hydraulic(session.certify_wet_waves())?;
                    hydraulic(session.certify_wet_waves())?;
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(session.point_limit(), 1);
                    assert_eq!(session.callback_points(), 1);
                    // Explicit exact-floor coverage keeps the exhausted original point ceiling.
                    let mut session = hydraulic(session.refine_depth_to(CertifiedRootTolerance {
                        atol: 0.0,
                        rtol: 0.0,
                    }))?;
                    hydraulic(session.certify_wet_waves())?;
                    assert_eq!(session.callback_points(), 1);
                    assert_eq!(session.point_limit(), 1);
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }

        fn empty<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(client, 1.0, 8.0, &[], &config, |pair, work| {
                let rows = support::rows::<R>(client, &[])?;
                let located = hydraulic(locate_wet_roots(pair, &rows, 0))?;
                let mut session = hydraulic(refine_wet_depth(pair, located))?;
                let before = work.snapshot();
                assert!(hydraulic(session.certify_wet_waves())?.is_empty());
                assert_eq!(work.snapshot(), before);
                Ok(())
            })
        }
    }
}
