use super::super::super::source_sonic_types::SourceSonicStatus;
use super::{
    PointCandidate, PointRootSession, Result, chart_test_support, frozen, hydraulic, support,
};
use numr::{dtype::DType, runtime::Runtime};

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    for weak_change in [false, true] {
        let mut config = chart_test_support::config();
        config.gravity_m_per_s2 = 3.;
        let section = chart_test_support::rectangle::<R>(client, 1., 4.)?;
        let owners = support::rows::<R>(client, &[0])?;
        let low = if weak_change {
            2.0_f64.powi(-80).to_bits()
        } else {
            0
        };
        let inputs = [
            [1.0_f64.to_bits(), 0, 0],
            [3.0_f64.to_bits(), low, 0],
            [2.0_f64.to_bits(), 0, 0],
            [3.0_f64.to_bits(), low, 0],
        ]
        .iter()
        .map(|p| frozen::expansion::<R>(client, &[*p], true))
        .collect::<Result<Vec<_>>>()?;
        // At Q=3 both source momenta equal21/2. Q=3+e gives their difference3e+e²/2>0.
        chart_test_support::with_inputs(client, section, owners, inputs, &config, |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let mut session =
                hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
            let result = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(result.status.try_to_vec::<i64>()?, [0]);
            super::diagnostics::check(&session, &result)?;
            let wave = &result.waves[0];
            let shock = &wave.sides[0];
            assert_eq!(shock.shock_actual.try_to_vec::<f64>()?, [1.]);
            assert_eq!(shock.shock_speed.available.try_to_vec::<f64>()?, [1.]);
            let speed = frozen::endpoints(&shock.shock_speed.physical)?[0];
            assert!(
                speed[0] <= 0. && speed[1] >= 0.,
                "Ambiguous left shock loses adjacent alternatives: {speed:?}"
            );
            assert_eq!(wave.regions.left_exterior.try_to_vec::<f64>()?, [1.]);
            assert_eq!(wave.regions.star.try_to_vec::<f64>()?, [1.]);
            assert!(
                result
                    .candidates
                    .iter()
                    .any(|c| matches!(c, PointCandidate::Exterior(_)))
            );
            assert!(
                result
                    .candidates
                    .iter()
                    .any(|c| matches!(c, PointCandidate::Star(_)))
            );
            assert!(
                !result
                    .candidates
                    .iter()
                    .any(|c| matches!(c, PointCandidate::Fan(_)))
            );
            if weak_change {
                assert_eq!(shock.shock_stationary.try_to_vec::<f64>()?, [0.]);
            } else if shock.shock_stationary.try_to_vec::<f64>()? == [1.] {
                assert_eq!(speed, [0., 0.]);
            }
            for endpoint in &wave.star.endpoints {
                for side in &endpoint.sides {
                    assert_eq!(
                        frozen::components(&side.source_discharge)?,
                        [[3.0_f64.to_bits(), low, 0]]
                    );
                }
            }
            for status in &result.sonic_status {
                assert_eq!(
                    status.try_to_vec::<i64>()?,
                    [SourceSonicStatus::NotRequested as i64]
                );
            }
            let before = work.snapshot();
            let points = session.callback_points();
            let again = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(again.status.try_to_vec::<i64>()?, [0]);
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before.residual_points);
            Ok(())
        })?;
    }
    Ok(())
}
