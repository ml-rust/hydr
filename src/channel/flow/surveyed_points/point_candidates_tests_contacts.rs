use super::super::super::{
    point_candidate_types::PointCandidateStatus,
    source_sonic_types::{SonicSide, SourceSonicStatus},
};
use super::{
    PointCandidate, PointFanAuthority, PointRootSession, Result, chart_test_support, frozen,
    hydraulic, support,
};
use numr::{dtype::DType, runtime::Runtime};
use solvr::optimize::CertifiedRootTolerance;

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    // Independent rectangle controls: source contact, critical equality, and star-sonic contact at A=1.
    support::with_rectangle::<R, _>(
        client,
        1.,
        8.,
        &[[1., 1., 0.25, 0.5], [1., 1., 1., 1.], [4., -4., 1., 1.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 1, 2])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            for owner in 0..3 {
                let owners = support::rows::<R>(client, &[owner])?;
                let mut result = hydraulic(session.certify_point_candidates(&owners))?;
                if owner == 2 {
                    hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    result = hydraulic(session.certify_point_candidates(&owners))?;
                }
                super::diagnostics::check(&session, &result)?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [PointCandidateStatus::Complete as i64],
                    "contact {owner}"
                );
                let actual = super::observed::actual(&session, &result)?;
                if owner == 0 {
                    assert_eq!(
                        result.sonic_status[0].try_to_vec::<i64>()?,
                        [SourceSonicStatus::SourceContact as i64]
                    );
                    assert!(actual.iter().all(|s| s.label == "left-exterior"));
                    assert!(!actual.is_empty());
                    for state in actual {
                        assert_eq!(state.area, [1., 1.]);
                        assert_eq!(state.discharge, [1., 1.]);
                    }
                    assert!(result.fan_evidence.iter().any(|e| {
                        e.side == SonicSide::Left
                            && e.membership
                                .proved
                                .try_to_vec::<f64>()
                                .is_ok_and(|v| v == [1.])
                    }));
                } else if owner == 1 {
                    assert!(actual.iter().any(|s| s.label == "left-exterior"));
                    assert!(actual.iter().any(|s| s.label == "star"));
                    assert!(
                        !result
                            .candidates
                            .iter()
                            .any(|s| matches!(s, PointCandidate::Fan(_)))
                    );
                    for state in &result.sonic_status {
                        assert_eq!(
                            state.try_to_vec::<i64>()?,
                            [SourceSonicStatus::NotRequested as i64]
                        );
                    }
                } else {
                    assert!(actual.iter().any(|s| s.label == "star"));
                    assert!(actual.iter().any(|s| s.label == "left-sonic"));
                    for state in actual {
                        super::observed::contains(
                            state.area,
                            [1.0_f64.to_bits(); 2],
                            "star-sonic contact area",
                        );
                        super::observed::contains(
                            state.discharge,
                            [1.0_f64.to_bits(); 2],
                            "star-sonic contact Q",
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
                        .ok_or("Missing established star-sonic candidate")?;
                    let PointFanAuthority::Wet(index) = fan.authority else {
                        return Err("Wet contact loses scoped membership".into());
                    };
                    let proof = &result.fan_evidence[index].membership;
                    assert_eq!(proof.coincidence.try_to_vec::<f64>()?, [1.]);
                    assert_eq!(frozen::components(&proof.lower_difference)?, [[0; 3]]);
                    assert_eq!(frozen::components(&proof.upper_difference)?, [[0; 3]]);
                }
            }
            assert_eq!(session.callback_points(), work.snapshot().residual_points);
            Ok(())
        },
    )
}
