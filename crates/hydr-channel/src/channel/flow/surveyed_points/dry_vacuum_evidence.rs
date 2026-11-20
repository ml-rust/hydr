use super::{
    dry_vacuum_types::DryVacuumEdges,
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SonicSide,
    wave_curve_types,
    wet_root_state::selected_value as select_curve,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn owners<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
) -> HydraulicResult<(Tensor<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    if owners.ndim() != 1 || owners.dtype() != DType::I64 || !owners.device().is_same(c.device()) {
        return Err(invalid(
            "dry vacuum owners need one-dimensional caller-device I64 metadata",
        ));
    }
    let zero = Tensor::zeros(&[], DType::I64, c.device())?;
    let count = Tensor::full_scalar(
        &[],
        DType::I64,
        session.invocation_rows.numel() as f64,
        c.device(),
    )?;
    if c.nonzero(&c.maximum(&c.lt(owners, &zero)?, &c.ge(owners, &count)?)?)?
        .numel()
        != 0
    {
        return Err(invalid(
            "dry vacuum owner lies outside the captured invocation",
        ));
    }
    if owners.numel() > 1 {
        let sorted = c.sort(owners, 0, false)?;
        let n = owners.numel() - 1;
        if c.nonzero(&c.eq(&sorted.narrow(0, 0, n)?, &sorted.narrow(0, 1, n)?)?)?
            .numel()
            != 0
        {
            return Err(invalid(
                "dry vacuum request contains duplicate invocation owners",
            ));
        }
    }
    let captured = Tensor::empty(owners.shape(), DType::I64, c.device())?;
    c.copy_into(&captured, owners)?;
    Ok((
        captured,
        c.index_select(&session.invocation_rows, 0, owners)?,
    ))
}

pub(super) fn edges<R>(
    a: &mut Arithmetic<'_, R>,
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    side: SonicSide,
) -> HydraulicResult<DryVacuumEdges<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let mut head = wave_curve_types::unavailable(a, session.invocation_rows.numel())?;
    let mut front = wave_curve_types::unavailable(a, session.invocation_rows.numel())?;
    for request in &session.sonic.requests {
        if request.binding.side != side {
            continue;
        }
        head = wave_curve_types::scatter_value(
            a,
            head,
            &request.binding.owners,
            &request.source_residual,
        )?;
        front = wave_curve_types::scatter_value(
            a,
            front,
            &request.binding.owners,
            &request.dry_residual,
        )?;
    }
    let mut head = select_curve(a, &head, owners)?;
    let mut front = select_curve(a, &front, owners)?;
    if side == SonicSide::Left {
        let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
        let z = a.expansion_point(&zero)?;
        head.retained = a.expansion_sub(&z, &head.retained)?;
        front.retained = a.expansion_sub(&z, &front.retained)?;
        let z = a.point(&zero)?;
        head.physical = a.sub(&z, &head.physical)?;
        front.physical = a.sub(&z, &front.physical)?;
    }
    #[cfg(test)]
    let head_exact_zero = c.mul(
        &head.available,
        &super::wet_star_endpoint::exact_zero(a, &head.retained)?,
    )?;
    #[cfg(test)]
    let front_exact_zero = c.mul(
        &front.available,
        &super::wet_star_endpoint::exact_zero(a, &front.retained)?,
    )?;
    Ok(DryVacuumEdges {
        head,
        front,
        #[cfg(test)]
        head_exact_zero,
        #[cfg(test)]
        front_exact_zero,
    })
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::{
        chart_fixture, dry_vacuum_types::DryVacuumCandidate, point_root_session::PointRootSession,
        source_fixture_support as frozen, wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_dry_vacuum_owners_isolate_selected_snapshots() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        ownership::run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_dry_vacuum_owners_isolate_selected_snapshots() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        ownership::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod ownership {
        use super::{
            DryVacuumCandidate, PointRootSession, Result, chart_fixture, frozen, hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::source_sonic_types::SonicSide;
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
            selected_snapshots::<R>(client)?;
            foreign_output::<R>(client)?;
            preflight::<R>(client)?;
            foreign_state::<R>(client)
        }
        fn selected_snapshots<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.], [0., 0., 1., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 1, 0])?;
                    let mut session = hydraulic(PointRootSession::new(
                        pair,
                        &parents,
                        config.root.max_points,
                    ))?;
                    let started = support::rows::<R>(client, &[2, 0])?;
                    hydraulic(session.start_source_sonic(&started, SonicSide::Left))?;
                    let before = work.snapshot();
                    let points = session.callback_points();
                    let owners = support::rows::<R>(client, &[0])?;
                    let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                    assert_eq!(work.snapshot().residual_points, before.residual_points);
                    assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                    assert_eq!(session.callback_points(), points);
                    assert_eq!(session.sonic.requests.len(), 1);
                    let fan = result
                        .candidates
                        .iter()
                        .find_map(|v| {
                            if let DryVacuumCandidate::Fan(f) = v {
                                Some(f)
                            } else {
                                None
                            }
                        })
                        .ok_or("Missing demanded existing sonic root")?;
                    assert_eq!(fan.invocation_owner.try_to_vec::<i64>()?, [0]);
                    assert_eq!(fan.parent_rows.try_to_vec::<i64>()?, [0]);
                    assert_eq!(fan.query.query.original_rows.try_to_vec::<i64>()?, [101]);
                    assert_eq!(result.candidates.len(), 1);
                    let original = frozen::endpoints(
                        &session.sonic.requests[0].domains[0].query.query.parameter,
                    )?;
                    let evidence =
                        frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?;
                    let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                    let changed = Tensor::full_scalar(&[1], DType::I64, 99., client.device())?;
                    client.copy_into(fan.query.query.parameter.lower(), &zero)?;
                    client.copy_into(fan.depth.lower(), &zero)?;
                    client.copy_into(&result.invocation_owner, &changed)?;
                    client.copy_into(&result.parent_rows, &changed)?;
                    client.copy_into(result.edges[0].head.physical.lower(), &zero)?;
                    assert_eq!(
                        frozen::endpoints(
                            &session.sonic.requests[0].domains[0].query.query.parameter
                        )?,
                        original
                    );
                    assert_eq!(
                        frozen::endpoints(&session.sonic.requests[0].source_residual.physical)?,
                        evidence
                    );
                    assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [0, 1, 0]);
                    let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
                    assert_eq!(repeated.invocation_owner.try_to_vec::<i64>()?, [0]);
                    hydraulic(session.refine_source_sonic(CertifiedRootTolerance {
                        atol: 1e-17,
                        rtol: 0.,
                    }))?;
                    let resumed = hydraulic(session.certify_dry_vacuum(&owners))?;
                    assert_eq!(resumed.candidates.len(), 1);
                    assert!(session.callback_points() >= points);
                    assert!(session.callback_points() <= config.root.max_points);
                    let right =
                        hydraulic(session.certify_dry_vacuum(&support::rows::<R>(client, &[1])?))?;
                    assert_eq!(right.candidates.len(), 1);
                    assert_eq!(session.sonic.requests.len(), 2);
                    Ok(())
                },
            )
        }
        fn foreign_output<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[0., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let result = {
                        let mut first = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                        let result = hydraulic(first.certify_dry_vacuum(&owners))?;
                        hydraulic(first.invocation.check(&result.invocation_witness))?;
                        result
                    };
                    let second = hydraulic(PointRootSession::new(pair, &owners, 1))?;
                    let before = work.snapshot();
                    assert!(second.invocation.check(&result.invocation_witness).is_err());
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }
        fn preflight<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            for alteration in 0..4 {
                support::with_rectangle::<R, _>(
                    client,
                    1.,
                    4.,
                    &[[1., 0., 0., 0.]],
                    &config,
                    |pair, work| {
                        let owners = support::rows::<R>(client, &[0])?;
                        let parents = support::rows::<R>(client, &[0, 0])?;
                        let mut session = hydraulic(PointRootSession::new(
                            pair,
                            &parents,
                            config.root.max_points,
                        ))?;
                        hydraulic(session.start_source_sonic(&owners, SonicSide::Left))?;
                        for bad in [&[-1][..], &[2][..], &[0, 0][..]] {
                            let before = work.snapshot();
                            let cache = chart_fixture::cache(session.sources())?;
                            assert!(
                                session
                                    .certify_dry_vacuum(&support::rows::<R>(client, bad)?)
                                    .is_err()
                            );
                            assert_eq!(work.snapshot(), before);
                            assert_eq!(chart_fixture::cache(session.sources())?, cache);
                        }
                        match alteration {
                            0 => session.sonic.requests[0].terminal = true,
                            1 => session.sonic.requests[0].binding.side = SonicSide::Right,
                            2 => {
                                session.sonic.requests[0].domains[0].parent_rows =
                                    support::rows::<R>(client, &[1])?
                            }
                            3 => session.sources.evaluation.panels *= 2,
                            _ => return Err("Unexpected preflight alteration".into()),
                        }
                        let before = work.snapshot();
                        let cache = chart_fixture::cache(session.sources())?;
                        assert!(
                            session
                                .certify_dry_vacuum(&support::rows::<R>(client, &[1])?)
                                .is_err()
                        );
                        assert_eq!(work.snapshot(), before);
                        assert_eq!(chart_fixture::cache(session.sources())?, cache);
                        Ok(())
                    },
                )?;
            }
            Ok(())
        }
        fn foreign_state<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let donor = {
                        let mut first = hydraulic(PointRootSession::new(
                            pair,
                            &owners,
                            config.root.max_points,
                        ))?;
                        hydraulic(first.start_source_sonic(&owners, SonicSide::Left))?;
                        first
                            .sonic
                            .requests
                            .pop()
                            .ok_or("Missing foreign request")?
                    };
                    let mut second =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    second.sonic.requests.push(donor);
                    let before = work.snapshot();
                    let cache = chart_fixture::cache(second.sources())?;
                    assert!(second.certify_dry_vacuum(&owners).is_err());
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(chart_fixture::cache(second.sources())?, cache);
                    Ok(())
                },
            )
        }
    }
}
