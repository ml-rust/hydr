use super::super::super::source_sonic_types::{SonicFanMembership, SonicSide, SourceSonicStatus};
use super::{
    DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus, PointRootSession, Result,
    chart_test_support, frozen, hydraulic, support,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

struct Reference {
    id: &'static str,
    side: SonicSide,
    area: [u64; 2],
    depth: [u64; 2],
    velocity: [u64; 2],
    discharge: [u64; 2],
}
const REFERENCES: &[Reference] = &[
    Reference {
        id: "tiny_rectangle_right_dry_fan",
        side: SonicSide::Left,
        area: [0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d],
        depth: [0x1a5c71c71c71c71c, 0x1a5c71c71c71c71d],
        velocity: [0x2d25555555555555, 0x2d25555555555556],
        discharge: [0x0792f684bda12f68, 0x0792f684bda12f69],
    },
    Reference {
        id: "scalar_rectangle_right_dry_sonic",
        side: SonicSide::Left,
        area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
        depth: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
        velocity: [0x3fe5555555555555, 0x3fe5555555555556],
        discharge: [0x3fd2f684bda12f68, 0x3fd2f684bda12f69],
    },
    Reference {
        id: "scalar_rectangle_left_dry_sonic",
        side: SonicSide::Right,
        area: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
        depth: [0x3fdc71c71c71c71c, 0x3fdc71c71c71c71d],
        velocity: [0xbfe5555555555556, 0xbfe5555555555555],
        discharge: [0xbfd2f684bda12f69, 0xbfd2f684bda12f68],
    },
    Reference {
        id: "scalar_triangle_right_dry_sonic",
        side: SonicSide::Left,
        area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
        depth: [0x3ff47ae147ae147a, 0x3ff47ae147ae147b],
        velocity: [0x3fe9999999999999, 0x3fe999999999999a],
        discharge: [0x3ff4f8b588e368f0, 0x3ff4f8b588e368f1],
    },
    Reference {
        id: "scalar_triangle_left_dry_sonic",
        side: SonicSide::Right,
        area: [0x3ffa36e2eb1c432c, 0x3ffa36e2eb1c432d],
        depth: [0x3ff47ae147ae147a, 0x3ff47ae147ae147b],
        velocity: [0xbfe999999999999a, 0xbfe9999999999999],
        discharge: [0xbff4f8b588e368f1, 0xbff4f8b588e368f0],
    },
    Reference {
        id: "scalar_three_bands_both_knots_dry_fan",
        side: SonicSide::Left,
        area: [0x4013f08ac385e532, 0x4013f08ac385e533],
        depth: [0x3ffaefa8a3a38101, 0x3ffaefa8a3a38102],
        velocity: [0x3fedd625462a7e0a, 0x3fedd625462a7e0b],
        discharge: [0x4012976d9b060300, 0x4012976d9b060301],
    },
    Reference {
        id: "scalar_three_bands_both_knots_dry_fan_reverse",
        side: SonicSide::Right,
        area: [0x4013f08ac385e532, 0x4013f08ac385e533],
        depth: [0x3ffaefa8a3a38101, 0x3ffaefa8a3a38102],
        velocity: [0xbfedd625462a7e0b, 0xbfedd625462a7e0a],
        discharge: [0xc012976d9b060301, 0xc012976d9b060300],
    },
];

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for reference in REFERENCES {
        let config = chart_test_support::config();
        chart_test_support::with_pair::<R, _>(client, &[reference.id], &config, |pair, work| {
            let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let initial = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(
                initial.regime.try_to_vec::<i64>()?,
                [if reference.side == SonicSide::Left {
                    DryVacuumRegime::LeftWet
                } else {
                    DryVacuumRegime::RightWet
                } as i64]
            );
            if f64::from_bits(reference.depth[0]) < 1e-30 {
                let atol = (f64::from_bits(reference.depth[0]) / cap) * 1e-12;
                assert!(atol > 0. && atol < config.root.atol);
                hydraulic(session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }))?;
            }
            let before = work.snapshot();
            let points = session.callback_points();
            let result = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [DryVacuumStatus::Complete as i64],
                "{}",
                reference.id
            );
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            assert_eq!(
                result.sonic_status[1 - reference.side.index()].try_to_vec::<i64>()?,
                [SourceSonicStatus::NotRequested as i64]
            );
            let cache = chart_test_support::cache(session.sources())?;
            assert!(
                cache[1 - reference.side.index()]
                    .ready
                    .iter()
                    .all(|&v| v == 0.)
            );
            let fan = result
                .candidates
                .iter()
                .find_map(|v| {
                    if let DryVacuumCandidate::Fan(fan) = v {
                        Some(fan)
                    } else {
                        None
                    }
                })
                .ok_or("Missing frozen sonic candidate")?;
            assert_eq!(fan.side, reference.side);
            assert_eq!(fan.fan_membership, SonicFanMembership::NotChecked);
            assert_eq!(fan.status.try_to_vec::<i64>()?, [0]);
            for (actual, expected) in [
                (&fan.query.area, reference.area),
                (&fan.depth, reference.depth),
                (&fan.velocity, reference.velocity),
                (&fan.discharge, reference.discharge),
            ] {
                let actual = frozen::endpoints(actual)?[0];
                let expected = expected.map(f64::from_bits);
                assert!(
                    actual[0] <= expected[0] && expected[1] <= actual[1],
                    "{} {actual:?} excludes {expected:?}",
                    reference.id
                );
                if f64::from_bits(reference.depth[0]) < 1e-30 {
                    assert!(
                        (actual[1] - actual[0]) / expected[0].abs() < 1e-10,
                        "{} {actual:?}",
                        reference.id
                    );
                }
            }
            let edges = &result.edges[reference.side.index()];
            let head = frozen::endpoints(&edges.head.physical)?[0];
            let front = frozen::endpoints(&edges.front.physical)?[0];
            if reference.side == SonicSide::Left {
                assert!(head[1] < 0. && front[0] > 0.);
            } else {
                assert!(head[0] > 0. && front[1] < 0.);
            }
            Ok(())
        })?;
    }
    regimes::<R>(client)
}
fn regimes<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for (id, regime) in [
        ("scalar_three_bands_both_dry", DryVacuumRegime::BothDry),
        ("scalar_rectangle_vacuum", DryVacuumRegime::Vacuum),
        (
            "scalar_rectangle_vacuum_threshold",
            DryVacuumRegime::Touching,
        ),
        (
            "scalar_rectangle_vacuum_threshold_outside",
            DryVacuumRegime::Vacuum,
        ),
        (
            "scalar_rectangle_vacuum_threshold_inside",
            DryVacuumRegime::NotApplicable,
        ),
        (
            "retained_vacuum_inside_film",
            DryVacuumRegime::NotApplicable,
        ),
    ] {
        let config = chart_test_support::config();
        chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let before = work.snapshot();
            let result = hydraulic(session.certify_dry_vacuum(&owners))?;
            assert_eq!(result.regime.try_to_vec::<i64>()?, [regime as i64], "{id}");
            if regime == DryVacuumRegime::NotApplicable {
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::NotApplicable as i64]
                );
                assert!(result.candidates.is_empty());
                assert_eq!(session.callback_points(), 0);
                for side in &result.sonic_status {
                    assert_eq!(
                        side.try_to_vec::<i64>()?,
                        [SourceSonicStatus::NotRequested as i64]
                    );
                }
            } else {
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::Complete as i64],
                    "{id}"
                );
                let dry = result
                    .candidates
                    .iter()
                    .find_map(|v| {
                        if let DryVacuumCandidate::Dry(dry) = v {
                            Some(dry)
                        } else {
                            None
                        }
                    })
                    .ok_or("Missing frozen dry candidate")?;
                assert_eq!(frozen::components(&dry.area)?, [[0; 3]]);
                assert_eq!(frozen::components(&dry.discharge)?, [[0; 3]]);
                assert_eq!(frozen::endpoints(&dry.depth)?, [[0., 0.]]);
                if regime == DryVacuumRegime::BothDry {
                    assert_eq!(session.callback_points(), 0);
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                }
                if regime == DryVacuumRegime::Vacuum {
                    let left = frozen::endpoints(&result.edges[0].front.physical)?[0];
                    let right = frozen::endpoints(&result.edges[1].front.physical)?[0];
                    if id == "scalar_rectangle_vacuum" {
                        assert!(left[1] < 0. && right[0] > 0.);
                        assert!(left[0] <= -1. && -1. <= left[1]);
                        assert!(right[0] <= 1. && 1. <= right[1]);
                    } else {
                        // Frozen outside-threshold edges retain a left contact and right front 2^-51.
                        assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [1.]);
                        assert_eq!(
                            frozen::components(&result.edges[0].front.retained)?,
                            [[0; 3]]
                        );
                        assert!(left[0] <= 0. && 0. <= left[1]);
                        let reference = f64::from_bits(0x3cc0000000000000);
                        assert!(right[0] > 0.);
                        assert!(right[0] <= reference && reference <= right[1]);
                        assert_eq!(
                            frozen::components(&result.edges[1].front.retained)?,
                            [[0x3cc0000000000000, 0, 0]]
                        );
                        assert_eq!(
                            result.sonic_status[0].try_to_vec::<i64>()?,
                            [SourceSonicStatus::DryContact as i64]
                        );
                        assert_eq!(
                            result.sonic_status[1].try_to_vec::<i64>()?,
                            [SourceSonicStatus::NoRootDry as i64]
                        );
                    }
                }
                if regime == DryVacuumRegime::Touching {
                    assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [1.]);
                    assert_eq!(result.edges[1].front_exact_zero.try_to_vec::<f64>()?, [1.]);
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}
