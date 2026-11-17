use super::super::super::source_sonic_types::SonicSide;
use super::{
    PointCandidate, PointRootSession, Result, chart_test_support, frozen, hydraulic, support,
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
    returned_binding::<R>(client)?;
    for initialized in [false, true] {
        foreign_wet::<R>(client, initialized)?;
    }
    foreign_sonic::<R>(client)?;
    snapshots_and_policy::<R>(client)
}
fn returned_binding<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[0., 0., 0., 0.]],
        &config,
        |pair, work| {
            let owners = support::rows::<R>(client, &[0])?;
            let output = {
                let mut first = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                let r = hydraulic(first.certify_point_candidates(&owners))?;
                hydraulic(first.invocation.check(&r.invocation_witness))?;
                r
            };
            let second = hydraulic(PointRootSession::new(pair, &owners, 1))?;
            let before = work.snapshot();
            assert!(second.invocation.check(&output.invocation_witness).is_err());
            assert_eq!(work.snapshot(), before);
            Ok(())
        },
    )
}
fn foreign_wet<R>(client: &R::Client, initialized: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    let state = if initialized {
        [4., 0., 1., 0.]
    } else {
        [1., 0., 1., 0.]
    };
    support::with_rectangle::<R, _>(client, 1., 8., &[state], &config, |pair, work| {
        let parents = support::rows::<R>(client, &[0, 0])?;
        let owners = support::rows::<R>(client, &[0])?;
        let donor = {
            let mut first = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            hydraulic(first.start_wet(&owners))?;
            assert_eq!(first.wet.domains[0].is_deferred(), !initialized);
            first
                .wet
                .domains
                .pop()
                .ok_or("Missing foreign wet continuation")?
        };
        let mut second = hydraulic(PointRootSession::new(
            pair,
            &parents,
            config.root.max_points,
        ))?;
        hydraulic(second.start_wet(&owners))?;
        second.wet.domains[0] = donor;
        let before = work.snapshot();
        let cache = chart_test_support::cache(second.sources())?;
        let empty = Tensor::zeros(&[2], DType::F64, client.device())?;
        assert!(second.certify_wet_star_selected(&empty).is_err());
        assert!(second.certify_wet_waves_selected(&empty).is_err());
        assert!(
            second
                .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                .is_err()
        );
        assert_eq!(work.snapshot(), before);
        assert_eq!(chart_test_support::cache(second.sources())?, cache);
        Ok(())
    })
}
fn foreign_sonic<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        4.,
        &[[1., 0., 0., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 0])?;
            let owners = support::rows::<R>(client, &[0])?;
            let donor = {
                let mut first = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                hydraulic(first.start_source_sonic(&owners, SonicSide::Left))?;
                first
                    .sonic
                    .requests
                    .pop()
                    .ok_or("Missing foreign sonic request")?
            };
            let mut second = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            hydraulic(second.start_source_sonic(&owners, SonicSide::Left))?;
            second.sonic.requests[0] = donor;
            let before = work.snapshot();
            let cache = chart_test_support::cache(second.sources())?;
            assert!(
                second
                    .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                    .is_err()
            );
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(second.sources())?, cache);
            Ok(())
        },
    )
}
fn snapshots_and_policy<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        8.,
        &[[4., 0., 1., 0.], [1., 0., 0., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 1, 0])?;
            let owners = support::rows::<R>(client, &[2, 1, 0])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            let result = hydraulic(session.certify_point_candidates(&owners))?;
            super::diagnostics::check(&session, &result)?;
            let wet = session
                .wet
                .domains
                .iter()
                .map(|d| frozen::endpoints(&d.query.query.parameter))
                .collect::<Result<Vec<_>>>()?;
            let sonic = session
                .sonic
                .requests
                .iter()
                .flat_map(|r| &r.domains)
                .map(|d| frozen::endpoints(&d.query.query.parameter))
                .collect::<Result<Vec<_>>>()?;
            for wave in &result.waves {
                let change = Tensor::full_scalar(
                    wave.star.query.query.parameter.lower().shape(),
                    DType::F64,
                    99.,
                    client.device(),
                )?;
                client.copy_into(wave.star.query.query.parameter.lower(), &change)?;
            }
            for candidate in &result.candidates {
                if let PointCandidate::Fan(fan) = candidate {
                    let change = Tensor::full_scalar(
                        fan.certificate.query.query.parameter.lower().shape(),
                        DType::F64,
                        99.,
                        client.device(),
                    )?;
                    client.copy_into(fan.certificate.query.query.parameter.lower(), &change)?;
                }
            }
            let changed = Tensor::full_scalar(owners.shape(), DType::I64, 99., client.device())?;
            client.copy_into(&result.invocation_owner, &changed)?;
            client.copy_into(&result.parent_rows, &changed)?;
            client.copy_into(&result.wet_status, &changed)?;
            assert_eq!(
                session
                    .wet
                    .domains
                    .iter()
                    .map(|d| frozen::endpoints(&d.query.query.parameter))
                    .collect::<Result<Vec<_>>>()?,
                wet
            );
            assert_eq!(
                session
                    .sonic
                    .requests
                    .iter()
                    .flat_map(|r| &r.domains)
                    .map(|d| frozen::endpoints(&d.query.query.parameter))
                    .collect::<Result<Vec<_>>>()?,
                sonic
            );
            let points = session.callback_points();
            let before = work.snapshot().residual_points;
            let repeat = hydraulic(session.certify_point_candidates(&owners))?;
            assert_eq!(repeat.invocation_owner.try_to_vec::<i64>()?, [2, 1, 0]);
            assert_eq!(repeat.status.try_to_vec::<i64>()?, [0; 3]);
            assert_eq!(session.callback_points(), points);
            assert_eq!(work.snapshot().residual_points, before);
            for bad in [&[-1][..], &[3][..], &[0, 0][..]] {
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                assert!(
                    session
                        .certify_point_candidates(&support::rows::<R>(client, bad)?)
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
            }
            session.sources.evaluation.panels *= 2;
            let before = work.snapshot();
            let cache = chart_test_support::cache(session.sources())?;
            assert!(session.certify_point_candidates(&owners).is_err());
            assert_eq!(work.snapshot(), before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            Ok(())
        },
    )
}
