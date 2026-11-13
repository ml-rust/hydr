use super::{
    Result, chart_test_support, frozen, hydraulic, locate_wet_roots, refine_wet_depth, support,
};
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

pub(super) fn run<R>(client: &R::Client) -> Result
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
    let config = chart_test_support::config();
    let ids = references.iter().map(|r| r.0).collect::<Vec<_>>();
    chart_test_support::with_pair::<R, _>(client, &ids, &config, |pair, work| {
        let rows = support::rows::<R>(client, &[3, 0, 4, 2, 1, 0])?;
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
                    0 => assert_eq!(domain.regions.right_fan.try_to_vec::<f64>()?[index], 1.0),
                    1 | 2 | 4 => assert_eq!(domain.regions.star.try_to_vec::<f64>()?[index], 1.0),
                    3 => assert_eq!(
                        domain.regions.left_exterior.try_to_vec::<f64>()?[index],
                        1.0
                    ),
                    _ => return Err("Unexpected frozen wave parent".into()),
                }
                match parent {
                    1 => {
                        assert!(
                            frozen::endpoints(&domain.sides[0].shock_speed.physical)?[index][1]
                                < 0.0
                        );
                        assert!(frozen::endpoints(&domain.sides[1].head.physical)?[index][0] > 0.0);
                        assert!(frozen::endpoints(&domain.sides[1].tail.physical)?[index][0] > 0.0);
                        unique_region(domain, index, 2)?;
                    }
                    2 => {
                        assert!(frozen::endpoints(&domain.sides[0].head.physical)?[index][1] < 0.0);
                        assert!(frozen::endpoints(&domain.sides[1].head.physical)?[index][0] > 0.0);
                        unique_region(domain, index, 2)?;
                    }
                    3 => {
                        for side in &domain.sides {
                            assert!(frozen::endpoints(&side.shock_speed.physical)?[index][0] > 0.0);
                        }
                        unique_region(domain, index, 0)?;
                    }
                    _ => {}
                }
                if parent == 2 {
                    for side in &domain.sides {
                        assert_eq!(side.equality_possible.try_to_vec::<f64>()?[index], 1.0);
                        assert_eq!(side.rarefaction_possible.try_to_vec::<f64>()?[index], 0.0);
                        assert_eq!(side.shock_possible.try_to_vec::<f64>()?[index], 0.0);
                    }
                    assert_eq!(domain.regions.left_fan.try_to_vec::<f64>()?[index], 0.0);
                    assert_eq!(domain.regions.right_fan.try_to_vec::<f64>()?[index], 0.0);
                }
                count += 1;
            }
        }
        assert_eq!(count, 6);
        Ok(())
    })?;
    stationary::<R>(client)?;
    separated_film::<R>(client)
}

fn edge_value<R: Runtime<DType = DType>>(domain: &WetWaveDomain<R>, edge: Edge) -> &CurveValue<R> {
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
    let mut config = chart_test_support::config();
    config.gravity_m_per_s2 = 3.0;
    chart_test_support::with_pair::<R, _>(
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
    let config = chart_test_support::config();
    chart_test_support::with_pair::<R, _>(
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
