use super::{PointRootSession, Result, chart_test_support, frozen, hydraulic, support};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::CertifiedRootTolerance;
use std::collections::BTreeMap;

type NativeSnapshot = (usize, usize, usize, bool, Vec<[f64; 2]>);
type StarSnapshot = ([f64; 2], [f64; 2], [f64; 2], i64);
type WaveSnapshot = ([f64; 5], i64);

pub(super) fn run<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    selections::<R>(client)?;
    complete_preflight::<R>(client)
}
fn state<R: Runtime<DType = DType>>(
    session: &PointRootSession<'_, '_, R>,
) -> Result<Vec<NativeSnapshot>> {
    session
        .wet
        .domains
        .iter()
        .map(|d| {
            Ok((
                d.iterations,
                d.callback_points,
                d.mapped_points,
                d.is_deferred(),
                frozen::endpoints(&d.query.query.parameter)?,
            ))
        })
        .collect()
}
fn star_rows<R: Runtime<DType = DType>>(
    domains: &[super::super::super::wet_star_types::WetStarDomain<R>],
) -> Result<BTreeMap<i64, StarSnapshot>> {
    let mut map = BTreeMap::new();
    for d in domains {
        let area = frozen::endpoints(&d.query.area)?;
        let u = frozen::endpoints(&d.velocity)?;
        let q = frozen::endpoints(&d.discharge)?;
        let status = d.status.try_to_vec::<i64>()?;
        for (i, owner) in d
            .invocation_owner
            .try_to_vec::<i64>()?
            .into_iter()
            .enumerate()
        {
            assert!(
                map.insert(owner, (area[i], u[i], q[i], status[i]))
                    .is_none()
            );
        }
    }
    Ok(map)
}
fn wave_rows<R: Runtime<DType = DType>>(
    domains: &[super::super::super::wet_wave_types::WetWaveDomain<R>],
) -> Result<BTreeMap<i64, WaveSnapshot>> {
    let mut map = BTreeMap::new();
    for d in domains {
        let p = &d.regions;
        let fields = [
            &p.left_exterior,
            &p.left_fan,
            &p.star,
            &p.right_fan,
            &p.right_exterior,
        ]
        .iter()
        .map(|t| t.try_to_vec::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
        let status = d.status.try_to_vec::<i64>()?;
        for (i, owner) in d
            .star
            .invocation_owner
            .try_to_vec::<i64>()?
            .into_iter()
            .enumerate()
        {
            assert!(
                map.insert(
                    owner,
                    (
                        [
                            fields[0][i],
                            fields[1][i],
                            fields[2][i],
                            fields[3][i],
                            fields[4][i]
                        ],
                        status[i]
                    )
                )
                .is_none()
            );
        }
    }
    Ok(map)
}
fn selections<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    support::with_rectangle::<R, _>(
        client,
        1.,
        8.,
        &[[1., 0., 1., 0.], [4., 0., 1., 0.]],
        &config,
        |pair, work| {
            let parents = support::rows::<R>(client, &[0, 0, 1, 1])?;
            let mut session = hydraulic(PointRootSession::new(
                pair,
                &parents,
                config.root.max_points,
            ))?;
            hydraulic(session.start_wet(&support::rows::<R>(client, &[0, 1, 2, 3])?))?;
            let before = state(&session)?;
            let cache = chart_test_support::cache(session.sources())?;
            let work_before = work.snapshot();
            let empty = Tensor::zeros(&[4], DType::F64, client.device())?;
            assert!(hydraulic(session.certify_wet_star_selected(&empty))?.is_empty());
            assert!(hydraulic(session.certify_wet_waves_selected(&empty))?.is_empty());
            let empty_owners = support::rows::<R>(client, &[])?;
            let point = hydraulic(session.certify_point_candidates(&empty_owners))?;
            assert!(point.candidates.is_empty() && point.waves.is_empty());
            assert_eq!(point.status.numel(), 0);
            assert_eq!(work.snapshot(), work_before);
            assert_eq!(chart_test_support::cache(session.sources())?, cache);
            assert_eq!(state(&session)?, before);
            let full_stars = hydraulic(session.certify_wet_star())?;
            let expected = star_rows(&full_stars)?;
            let full_waves = hydraulic(session.certify_wet_waves())?;
            let wave_expected = wave_rows(&full_waves)?;
            let mask = Tensor::from_slice(&[1., 0., 0., 1.], &[4], client.device())?;
            let selected = hydraulic(session.certify_wet_star_selected(&mask))?;
            let selected_rows = star_rows(&selected)?;
            assert_eq!(selected_rows.keys().copied().collect::<Vec<_>>(), [0, 3]);
            for (&owner, value) in &selected_rows {
                assert_eq!(Some(value), expected.get(&owner));
            }
            let selected_waves = hydraulic(session.certify_wet_waves_selected(&mask))?;
            for (owner, value) in wave_rows(&selected_waves)? {
                assert_eq!(Some(&value), wave_expected.get(&owner));
            }
            assert_eq!(state(&session)?, before);
            assert_eq!(work.snapshot().residual_points, work_before.residual_points);
            assert_eq!(work.snapshot().root_iterations, work_before.root_iterations);
            for bad in [
                Tensor::zeros(&[4], DType::I64, client.device())?,
                Tensor::zeros(&[3], DType::F64, client.device())?,
                Tensor::from_slice(&[0., 1., 2., 0.], &[4], client.device())?,
            ] {
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                assert!(session.certify_wet_star_selected(&bad).is_err());
                assert!(session.certify_wet_waves_selected(&bad).is_err());
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
            }
            let all = Tensor::full_scalar(&[4], DType::F64, 1., client.device())?;
            let owned = hydraulic(session.certify_wet_waves_selected(&all))?;
            for w in &owned {
                let changed = Tensor::full_scalar(
                    w.star.query.query.parameter.lower().shape(),
                    DType::F64,
                    99.,
                    client.device(),
                )?;
                client.copy_into(w.star.query.query.parameter.lower(), &changed)?;
            }
            assert_eq!(
                state(&session)?,
                before,
                "Full-demand output aliases native wet continuation operands"
            );
            hydraulic(session.refine_wet_depth_in_place(CertifiedRootTolerance {
                atol: 1e-17,
                rtol: 0.,
            }))?;
            assert!(session.callback_points() >= work_before.residual_points);
            assert!(session.callback_points() <= config.root.max_points);
            let refined = hydraulic(session.certify_wet_star_selected(&mask))?;
            assert_eq!(star_rows(&refined)?.len(), 2);
            Ok(())
        },
    )
}
fn complete_preflight<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: support::DepthClient<R>,
{
    let config = chart_test_support::config();
    for mutate_owner in [true, false] {
        support::with_rectangle::<R, _>(
            client,
            1.,
            8.,
            &[[1., 0., 1., 0.], [4., 0., 1., 0.]],
            &config,
            |pair, work| {
                let owners = support::rows::<R>(client, &[0, 1])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                hydraulic(session.start_wet(&owners))?;
                if mutate_owner {
                    session.wet.domains[0].invocation_owner = support::rows::<R>(client, &[1])?;
                } else {
                    session.wet.domains[0].parent_rows = support::rows::<R>(client, &[1])?;
                }
                let mask = Tensor::from_slice(&[0., 1.], &[2], client.device())?;
                let before = work.snapshot();
                let cache = chart_test_support::cache(session.sources())?;
                assert!(session.certify_wet_star_selected(&mask).is_err());
                assert!(session.certify_wet_waves_selected(&mask).is_err());
                assert!(
                    session
                        .certify_point_candidates(&support::rows::<R>(client, &[1])?)
                        .is_err()
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_test_support::cache(session.sources())?, cache);
                Ok(())
            },
        )?;
    }
    Ok(())
}
