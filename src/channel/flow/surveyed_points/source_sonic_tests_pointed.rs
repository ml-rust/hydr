use super::{
    PointRootSession, Result, SonicSide, SourceSonicStatus, chart_test_support, frozen, hydraulic,
    support,
};
use crate::channel::{
    flow::surveyed_points::source_sonic_charts,
    section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint},
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
    R::Client: support::DepthClient<R>,
{
    // Independent control: A=h²,g=2,c=A^(1/4),Phi=4*c. Epsilon=5*2^-83 gives Asonic=2^-332.
    for upper_exponent in [-331, -330] {
        let mut config = chart_test_support::config();
        config.gravity_m_per_s2 = 2.0;
        let section = CrossSectionBatch::<R>::new(
            client.clone(),
            vec![CrossSectionDefinition {
                id: "pointed_source_sonic_control".into(),
                vertical_datum: "Exact pointed datum".into(),
                points: [(-4.0, 4.0), (0.0, 0.0), (4.0, 4.0)]
                    .into_iter()
                    .map(|(station_m, elevation_m)| CrossSectionPoint {
                        station_m,
                        elevation_m,
                    })
                    .collect(),
                maximum_stage_m: 4.0,
                profile_provenance: "Independent A=h² identity".into(),
            }],
        )?
        .certify_prismatic()?;
        let owners = support::rows::<R>(client, &[0, 0])?;
        let epsilon = 5.0 * 2.0_f64.powi(-83);
        let input = [
            [[0, 0, 0], [1.0_f64.to_bits(), 0, 0]],
            [[0, 0, 0], [(-4.0_f64).to_bits(), epsilon.to_bits(), 0]],
            [[1.0_f64.to_bits(), 0, 0], [0, 0, 0]],
            [[4.0_f64.to_bits(), (-epsilon).to_bits(), 0], [0, 0, 0]],
        ]
        .iter()
        .map(|values| frozen::expansion::<R>(client, values, true))
        .collect::<Result<Vec<_>>>()?;
        chart_test_support::with_inputs(client, section, owners, input, &config, |pair, work| {
            let rows = support::rows::<R>(client, &[0, 1])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[1])?, SonicSide::Left),
            )?;
            hydraulic(
                session.start_source_sonic(&support::rows::<R>(client, &[0])?, SonicSide::Right),
            )?;
            for request in &mut session.sonic.requests {
                assert_eq!(request.domains.len(), 1);
                assert!(request.domains[0].refinement.is_none());
                assert_eq!(
                    request.domains[0].classification,
                    SourceSonicStatus::PositiveRoot
                );
                let domain = &mut request.domains[0];
                let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                let upper = Tensor::full_scalar(
                    &[1],
                    DType::F64,
                    2.0_f64.powi(upper_exponent),
                    client.device(),
                )?;
                // Enlarge only a deferred supported bracket around the independently known root.
                domain.query = hydraulic(source_sonic_charts::chart(
                    session.sources,
                    &domain.parent_rows,
                    domain.side,
                    &zero,
                    &upper,
                ))?;
                let (depth, normalized_depth) = hydraulic(
                    crate::channel::flow::surveyed_points::point_root_depth::certificates(
                        session.sources,
                        &domain.parent_rows,
                        &domain.query,
                    ),
                )?;
                domain.depth = depth;
                domain.normalized_depth = normalized_depth;
            }
            let points_before = session.callback_points();
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 2.0_f64.powi(-210),
                rtol: 0.0,
            }))?;
            let result = hydraulic(session.certify_source_sonic())?;
            assert_eq!(result.domains.len(), 2);
            for domain in &result.domains {
                super::diagnostics::check(domain)?;
                assert_eq!(domain.status.try_to_vec::<i64>()?, [0]);
                assert!(domain.iterations > 0 && domain.iterations <= config.root.max_iterations);
                let sign = if domain.side == SonicSide::Left {
                    1.0
                } else {
                    -1.0
                };
                for (actual, expected) in [
                    (&domain.query.area, 2.0_f64.powi(-332)),
                    (&domain.depth, 2.0_f64.powi(-166)),
                    (&domain.velocity, sign * 2.0_f64.powi(-83)),
                    (&domain.discharge, sign * 2.0_f64.powi(-415)),
                ] {
                    let [lower, upper] = frozen::endpoints(actual)?[0];
                    assert!(
                        lower <= expected && expected <= upper,
                        "{lower:e}..{upper:e} excludes {expected:e}"
                    );
                    assert!((upper - lower) / expected.abs() < 1e-10);
                }
            }
            assert!(session.callback_points() > points_before);
            let before = work.snapshot();
            let points = session.callback_points();
            let iterations = result
                .domains
                .iter()
                .map(|d| d.iterations)
                .collect::<Vec<_>>();
            // Stricter finite stopping resumes the retained exact-root state without another endpoint admission.
            hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                atol: 2.0_f64.powi(-230),
                rtol: 0.0,
            }))?;
            let after = hydraulic(session.certify_source_sonic())?;
            assert_eq!(
                after
                    .domains
                    .iter()
                    .map(|d| d.iterations)
                    .collect::<Vec<_>>(),
                iterations
            );
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            Ok(())
        })?;
    }
    Ok(())
}
