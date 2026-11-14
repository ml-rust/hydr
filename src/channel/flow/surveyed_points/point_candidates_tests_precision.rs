use super::super::super::{
    point_candidate_types::{PointCandidateStatus, PointRegime},
    source_sonic_types::SonicSide,
};
use super::{
    PointCandidate, PointFanAuthority, PointRootSession, Result, chart_test_support, fixtures,
    frozen, hydraulic, support,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    film::<R>(client)?;
    exterior_and_capacity::<R>(client)
}
fn film<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let reference = fixtures::executions()
        .into_iter()
        .find(|r| r.id == "retained_vacuum_inside_film")
        .ok_or("Missing independent film reference")?;
    let config = fixtures::CONFIGS[reference.config].build();
    chart_test_support::with_pair::<R, _>(client, &[reference.id], &config, |pair, work| {
        let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
        let owners = support::rows::<R>(client, &[0])?;
        let mut session = hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
        let initial = hydraulic(session.certify_point_candidates(&owners))?;
        assert_eq!(
            initial.regime.try_to_vec::<i64>()?,
            [PointRegime::Wet as i64]
        );
        assert_eq!(
            initial.status.try_to_vec::<i64>()?,
            [PointCandidateStatus::Complete as i64]
        );
        let overlap = initial
            .fan_evidence
            .iter()
            .find(|e| e.side == SonicSide::Right)
            .ok_or("Film lacks initial right-fan evidence")?;
        assert_eq!(overlap.membership.pending.try_to_vec::<f64>()?, [0.]);
        assert_eq!(overlap.membership.proved.try_to_vec::<f64>()?, [1.]);
        let separation = frozen::endpoints(&overlap.membership.separation.physical)?[0];
        let tail = frozen::endpoints(&initial.waves[0].sides[1].tail.physical)?[0];
        assert!(tail[1] <= 0. || separation[1] <= 0.);
        eprintln!("POINT_FILM_INITIAL\tright_tail={tail:?}\tstar_minus_sonic={separation:?}");
        let truth = reference.star.ok_or("Film lacks independent star truth")?;
        let atol = f64::from_bits(truth.area[0]) / cap * 1e-12;
        assert!(atol > 0. && atol < config.root.atol);
        hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance { atol, rtol: 0. }))?;
        hydraulic(session.refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }))?;
        let before = work.snapshot();
        let points = session.callback_points();
        let result = hydraulic(session.certify_point_candidates(&owners))?;
        assert_eq!(session.callback_points(), points);
        assert_eq!(work.snapshot().residual_points, before.residual_points);
        assert_eq!(
            result.status.try_to_vec::<i64>()?,
            [PointCandidateStatus::Complete as i64]
        );
        super::diagnostics::check(&session, &result)?;
        let actual = super::observed::actual(&session, &result)?;
        assert!(actual.iter().all(|s| s.label == "right-sonic"));
        assert_eq!(actual.len(), 1);
        let expected = &reference.traces[0];
        let state = &actual[0];
        assert!(state.area[0] > f64::from_bits(truth.area[1]));
        for (actual, expected) in [
            (state.area, expected.area),
            (
                state.depth,
                expected.depth.ok_or("Film missing depth truth")?,
            ),
            (state.discharge, expected.discharge),
            (
                state.velocity.ok_or("Film missing sonic velocity")?,
                expected
                    .velocity
                    .ok_or("Film missing independent velocity")?,
            ),
        ] {
            super::observed::contains(actual, expected, reference.id);
            let scale = f64::from_bits(expected[0]).abs();
            assert!(
                (actual[1] - actual[0]) / scale < 1e-10,
                "Film precision {actual:?}"
            );
        }
        let fan = result
            .candidates
            .iter()
            .find_map(|c| {
                if let PointCandidate::Fan(f) = c {
                    Some(f)
                } else {
                    None
                }
            })
            .ok_or("Missing complete film sonic candidate")?;
        let PointFanAuthority::Wet(index) = fan.authority else {
            return Err("Film fan lacks wet membership".into());
        };
        assert_eq!(
            result.fan_evidence[index]
                .membership
                .proved
                .try_to_vec::<f64>()?,
            [1.]
        );
        assert_eq!(session.point_limit(), config.root.max_points);
        assert!(session.callback_points() <= session.point_limit());
        Ok(())
    })
}
fn exterior_and_capacity<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for id in [
        "tiny_rectangle_compression",
        "uniform_cap_retained_deficit_dyadic_velocity",
    ] {
        let row = fixtures::executions()
            .into_iter()
            .find(|r| r.id == id)
            .ok_or("Missing independent source-scale reference")?;
        let config = fixtures::CONFIGS[row.config].build();
        chart_test_support::with_pair::<R, _>(client, &[id], &config, |pair, _| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let result = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(
                result.status.try_to_vec::<i64>()?,
                [PointCandidateStatus::Complete as i64]
            );
            super::diagnostics::check(&session, &result)?;
            let actual = super::observed::actual(&session, &result)?;
            let expected = &row.traces[0];
            assert!(!actual.is_empty());
            for state in actual {
                assert_eq!(state.label, expected.label);
                for (actual, expected) in [
                    (state.area, expected.area),
                    (state.discharge, expected.discharge),
                ] {
                    super::observed::contains(actual, expected, id);
                    assert!(
                        (actual[1] - actual[0]) / f64::from_bits(expected[0]).abs() < 1e-10,
                        "{id} source-scale gate {actual:?}"
                    );
                }
            }
            if id == "uniform_cap_retained_deficit_dyadic_velocity" {
                let source = frozen::cases()
                    .iter()
                    .find(|r| r.id == id)
                    .ok_or("Missing captured source")?;
                for wave in &result.waves {
                    for endpoint in &wave.star.endpoints {
                        for side in &endpoint.sides {
                            assert_eq!(
                                frozen::components(&side.source_discharge)?,
                                [source.inputs[1]]
                            );
                        }
                    }
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}
