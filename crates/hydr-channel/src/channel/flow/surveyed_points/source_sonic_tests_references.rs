use super::{
    PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen, hydraulic,
    support,
};
use crate::channel::flow::surveyed_points::corpus_coverage::Coverage;
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
        id: "retained_vacuum_inside_film",
        side: SonicSide::Right,
        area: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
        depth: [0x35bc71c71c71c71c, 0x35bc71c71c71c71d],
        velocity: [0xbad5555555555556, 0xbad5555555555555],
        discharge: [0xb0a2f684bda12f69, 0xb0a2f684bda12f68],
    },
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

pub(super) fn run<R>(client: &R::Client, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let replayed = coverage.select(REFERENCES, |reference| {
        (
            reference.side == SonicSide::Left,
            reference.id.starts_with("scalar_"),
        )
    });
    // Only explicitly labelled sonic candidates supply these truths. Legacy velocities use independent rational Q/A bounds.
    for reference in replayed {
        let config = chart_test_support::config();
        chart_test_support::with_pair::<R, _>(client, &[reference.id], &config, |pair, work| {
            let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
            let parents = support::rows::<R>(client, &[0])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let owners = support::rows::<R>(client, &[0])?;
            hydraulic(session.start_source_sonic(&owners, reference.side))?;
            if f64::from_bits(reference.depth[0]) < 1e-30 {
                let atol = (f64::from_bits(reference.depth[0]) / cap) * 1e-12;
                assert!(atol > 0.0 && atol < config.root.atol);
                hydraulic(session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0.0 }))?;
            }
            let cache = chart_test_support::cache(session.sources())?;
            assert!(
                cache[1 - reference.side.index()]
                    .ready
                    .iter()
                    .all(|&x| x == 0.0)
            );
            let before = work.snapshot();
            let result = hydraulic(session.certify_source_sonic())?;
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            assert_eq!(
                result.status[reference.side.index()].try_to_vec::<i64>()?,
                [SourceSonicStatus::PositiveRoot as i64]
            );
            super::diagnostics::evidence(&result)?;
            assert_eq!(result.domains.len(), 1);
            let domain = &result.domains[0];
            super::diagnostics::check(domain)?;
            assert_eq!(domain.status.try_to_vec::<i64>()?, [0], "{}", reference.id);
            for (actual, expected) in [
                (&domain.query.area, reference.area),
                (&domain.depth, reference.depth),
                (&domain.velocity, reference.velocity),
                (&domain.discharge, reference.discharge),
            ] {
                let actual = frozen::endpoints(actual)?[0];
                let expected = expected.map(f64::from_bits);
                assert!(
                    actual[0] <= expected[0] && expected[1] <= actual[1],
                    "{}: {actual:?} excludes {expected:?}",
                    reference.id
                );
                if f64::from_bits(reference.depth[0]) < 1e-30 {
                    assert!(
                        (actual[1] - actual[0]) / expected[0].abs() < 1e-10,
                        "{} precision: {actual:?}",
                        reference.id
                    );
                }
            }
            // The retained film's selected sonic root is distinct from its wet-star root 2^-164.
            if reference.id == "retained_vacuum_inside_film" {
                assert!(frozen::endpoints(&domain.query.area)?[0][0] > 2.0_f64.powi(-164));
            }
            Ok(())
        })?;
    }
    Ok(())
}
