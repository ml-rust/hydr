use super::super::super::{
    source_sonic_types::SonicSide, wet_candidates, wet_candidates_star, wet_fan_membership,
    wet_star_types::WetStarStatus, wet_wave_types::WetWaveDomain, wet_waves_endpoint,
    wet_waves_regions, wet_waves_side,
};
use super::{
    PointCandidate, PointRootSession, Result, chart_test_support, frozen, hydraulic, support,
};
use crate::channel::flow::surveyed::error::checked;
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    // Exact rectangle: left invariant=3, G=3*sqrt(A)-3, and the right source is A=Q=1.
    support::with_rectangle::<R, _>(
        client,
        1.,
        8.,
        &[[4., -4., 1., 1.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            hydraulic(session.start_wet(&owners))?;
            hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
            let original_star = frozen::endpoints(&session.wet.domains[0].query.area)?[0];
            let original_sonic =
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.area)?[0];
            assert!(0.5 <= original_star[0] && original_star[1] <= 1.5);
            assert!(0.75 <= original_sonic[0] && original_sonic[1] <= 1.25);
            let star = chart_test_support::chart(
                session.sources(),
                chart_test_support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.5, 1.5]],
            )?;
            let sonic = chart_test_support::chart(
                session.sources(),
                chart_test_support::Anchor::Zero,
                [1.0_f64.to_bits(), 0],
                &[[0.75, 1.25]],
            )?;
            let before = work.snapshot();
            let points = session.callback_points();
            let level = hydraulic(session.sources.panel_level(0))?;
            let star = hydraulic(wet_candidates_star::evaluate(
                session.sources,
                &owners,
                &owners,
                &star,
                &level,
            ))?;
            assert_eq!(
                star.status.try_to_vec::<i64>()?,
                [WetStarStatus::Certified as i64]
            );
            let wave = hydraulic(checked::<R, _>(
                client,
                "independent broad root wave fixture",
                |a| {
                    let selected = session.sources.select_rows(&owners, &star.query)?;
                    let sound = wet_waves_endpoint::sounds(a, session.sources, &selected, &star)?;
                    let sides = [
                        wet_waves_side::evaluate(a, &selected, &star, &sound, 1., 0)?,
                        wet_waves_side::evaluate(a, &selected, &star, &sound, 1., 1)?,
                    ];
                    let regions = wet_waves_regions::evaluate(a, &sides)?;
                    let covered = client.maximum(
                        &regions.left_exterior,
                        &client.maximum(
                            &regions.left_fan,
                            &client.maximum(
                                &regions.star,
                                &client.maximum(&regions.right_fan, &regions.right_exterior)?,
                            )?,
                        )?,
                    )?;
                    assert_eq!(covered.try_to_vec::<f64>()?, [1.]);
                    for side in &sides {
                        assert_eq!(side.valid.try_to_vec::<f64>()?, [1.]);
                    }
                    Ok(WetWaveDomain {
                        star,
                        endpoint_sound: sound,
                        sides,
                        regions,
                        status: Tensor::zeros(&[1], DType::I64, client.device())?,
                    })
                },
            ))?;
            let tail = frozen::endpoints(&wave.sides[0].tail.physical)?[0];
            assert!(tail[0] < 0. && tail[1] > 0.);
            assert_eq!(wave.regions.left_fan.try_to_vec::<f64>()?, [1.]);
            assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.]);
            let proof = hydraulic(checked::<R, _>(
                client,
                "overlapping established roots retain uncertainty",
                |a| {
                    wet_fan_membership::evaluate(
                        a,
                        session.sources(),
                        &wave,
                        &owners,
                        &sonic,
                        SonicSide::Left,
                    )
                },
            ))?;
            assert_eq!(proof.proved.try_to_vec::<f64>()?, [0.]);
            assert_eq!(proof.excluded.try_to_vec::<f64>()?, [0.]);
            assert_eq!(proof.pending.try_to_vec::<f64>()?, [1.]);
            assert_eq!(proof.coincidence.try_to_vec::<f64>()?, [0.]);
            let difference = frozen::endpoints(&proof.separation.physical)?[0];
            assert!(difference[0] < 0. && difference[1] > 0.);
            let waves = [wave];
            let pending = hydraulic(wet_candidates::evaluate(&mut session, &waves))?;
            assert_eq!(pending.complete.try_to_vec::<f64>()?, [0.]);
            assert_eq!(pending.covered.try_to_vec::<f64>()?, [1.]);
            assert!(pending.candidates.iter().any(|c|matches!(c,PointCandidate::Pending(p) if p.region==super::super::super::point_candidate_types::PointRegion::LeftFan)));
            assert!(
                !pending
                    .candidates
                    .iter()
                    .any(|c| matches!(c, PointCandidate::Fan(_)))
            );
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            assert_eq!(work.snapshot().root_iterations, before.root_iterations);
            assert_eq!(
                frozen::endpoints(&session.wet.domains[0].query.area)?[0],
                original_star
            );
            assert_eq!(
                frozen::endpoints(&session.sonic.requests[0].domains[0].query.area)?[0],
                original_sonic
            );
            hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                atol: 1e-17,
                rtol: 0.,
            }))?;
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 1e-17,
                rtol: 0.,
            }))?;
            let resumed = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(resumed.status.try_to_vec::<i64>()?, [0]);
            super::diagnostics::check(&session, &resumed)?;
            Ok(())
        },
    )
}
