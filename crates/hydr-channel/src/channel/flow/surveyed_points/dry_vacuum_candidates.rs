use super::{
    dry_vacuum_mirror::MirrorRows,
    dry_vacuum_types::{DryVacuumCandidate, DryVacuumPendingFan, DryVacuumPossibilities},
    point_candidate_types::DryPoint,
    point_root_session::PointRootSession,
    source_sonic_types::{SonicSide, SourceSonicCertificate, SourceSonicStatus},
};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{AreaAnchor, AreaDirection, AreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn assemble<R>(
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    parent: &Tensor<R>,
    possible: &DryVacuumPossibilities<R>,
    sonic: Vec<SourceSonicCertificate<R>>,
    mirror: &MirrorRows<R>,
) -> HydraulicResult<(Vec<DryVacuumCandidate<R>>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = session.sources.geometry.client();
    checked::<R, _>(c, "dry vacuum physical candidate coverage", |a| {
        let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
        let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
        let fullzero = Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?;
        let mut physical = [fullzero.clone(), fullzero.clone()];
        let mut present = [fullzero.clone(), fullzero];
        let mut output = Vec::new();
        for certificate in sonic {
            let side = certificate.side.index();
            let success = c.eq(
                &certificate.status,
                &Tensor::zeros(&[], DType::I64, c.device())?,
            )?;
            let success = c.where_cond(
                &success,
                &Tensor::full_scalar(certificate.status.shape(), a.dtype(), 1., c.device())?,
                &Tensor::zeros(certificate.status.shape(), a.dtype(), c.device())?,
            )?;
            physical[side] =
                c.index_put(&physical[side], 0, &certificate.invocation_owner, &success)?;
            present[side] = c.index_put(
                &present[side],
                0,
                &certificate.invocation_owner,
                &Tensor::full_scalar(certificate.status.shape(), a.dtype(), 1., c.device())?,
            )?;
            output.push(DryVacuumCandidate::Fan(Box::new(certificate)));
        }
        let mut complete = one.clone();
        let mut dry = possible.dry.clone();
        for side in [SonicSide::Left, SonicSide::Right] {
            let i = side.index();
            let status = c.index_select(&session.sonic.status[i], 0, owners)?;
            let is = |kind: SourceSonicStatus| -> HydraulicResult<Tensor<R>> {
                Ok(c.where_cond(
                    &c.eq(
                        &status,
                        &Tensor::full_scalar(&[], DType::I64, kind as i64 as f64, c.device())?,
                    )?,
                    &one,
                    &zero,
                )?)
            };
            let fan = if i == 0 {
                &possible.left_fan
            } else {
                &possible.right_fan
            };
            let source = c.mul(fan, &is(SourceSonicStatus::SourceContact)?)?;
            let drycontact = c.mul(fan, &is(SourceSonicStatus::DryContact)?)?;
            dry = c.maximum(&dry, &drycontact)?;
            let exterior = if i == 0 {
                &possible.left_exterior
            } else {
                &possible.right_exterior
            };
            let exterior = c.maximum(exterior, &source)?;
            let rows = c.nonzero(&exterior)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() != 0 {
                output.push(DryVacuumCandidate::Exterior(Box::new(
                    super::point_candidates_source::exterior(
                        a,
                        session.sources,
                        &c.index_select(owners, 0, &rows)?,
                        &c.index_select(parent, 0, &rows)?,
                        side,
                    )?,
                )));
            }
            let valid = c.maximum(
                &source,
                &c.maximum(&drycontact, &c.index_select(&physical[i], 0, owners)?)?,
            )?;
            complete = c.mul(&complete, &c.where_cond(fan, &valid, &one)?)?;
            let known = c.maximum(
                &source,
                &c.maximum(&drycontact, &c.index_select(&present[i], 0, owners)?)?,
            )?;
            let pending = c.mul(fan, &c.sub(&one, &known)?)?;
            let rows = c.nonzero(&pending)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() != 0 {
                output.push(DryVacuumCandidate::PendingFan(Box::new(
                    DryVacuumPendingFan {
                        invocation_owner: c.index_select(owners, 0, &rows)?,
                        parent_rows: c.index_select(parent, 0, &rows)?,
                        side,
                        #[cfg(test)]
                        source_status: c.index_select(&status, 0, &rows)?,
                    },
                )));
            }
        }
        let rows = c.nonzero(&dry)?;
        let rows = rows.reshape(&[rows.numel()])?;
        if rows.numel() != 0 {
            let selected_parent = c.index_select(parent, 0, &rows)?;
            let selected = session.sources.select_sources(a, &selected_parent)?;
            let z = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
            let area = a.expansion_point(&z)?;
            let parameter = a.point(&z)?;
            let query = AreaQuery::band_affine(
                a,
                &selected.bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &area,
                &parameter,
            )?;
            c.copy_into(&query.original_rows, &selected.original_rows)?;
            let query = query.prove_support(a, &selected.bands)?;
            output.push(DryVacuumCandidate::Dry(Box::new(DryPoint {
                invocation_owner: c.index_select(owners, 0, &rows)?,
                parent_rows: selected_parent,
                query,
                area,
                discharge: a.expansion_point(&z)?,
                momentum_upper: c.index_select(&mirror.momentum, 0, &rows)?,
                #[cfg(test)]
                depth: a.bounds(&z, &c.index_select(&mirror.depth, 0, &rows)?)?,
            })));
        }
        let coverage = c.maximum(
            &possible.left_exterior,
            &c.maximum(
                &possible.left_fan,
                &c.maximum(
                    &possible.dry,
                    &c.maximum(&possible.right_fan, &possible.right_exterior)?,
                )?,
            )?,
        )?;
        Ok((output, c.mul(&complete, &coverage)?))
    })
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::{
        chart_fixture,
        dry_vacuum_types::{DryVacuumCandidate, DryVacuumStatus},
        point_root_session::PointRootSession,
        source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    #[test]
    fn configured_cpu_dry_vacuum_candidates_keep_retained_exteriors_and_work_limits() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = CpuClient::new(CpuDevice::new()).with_parallelism(config);
        controls::run::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_dry_vacuum_candidates_keep_retained_exteriors_and_work_limits() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        controls::run::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod controls {
        use super::{
            DryVacuumCandidate, DryVacuumStatus, PointRootSession, Result, chart_fixture, frozen,
            hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::source_sonic_types::SourceSonicStatus;
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
            retained_exterior::<R>(client)?;
            exhausted::<R>(client)
        }

        fn retained_exterior<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let section = chart_fixture::rectangle::<R>(client, 1., 4.)?;
            let owners = support::rows::<R>(client, &[0])?;
            let low = 2.0_f64.powi(-80);
            let area = [1.0_f64.to_bits(), low.to_bits(), 0];
            let discharge = [3.0_f64.to_bits(), (3. * low).to_bits(), 0];
            let inputs = [area, discharge, [0; 3], [0; 3]]
                .iter()
                .map(|v| frozen::expansion::<R>(client, &[*v], true))
                .collect::<Result<Vec<_>>>()?;
            chart_fixture::with_inputs(client, section, owners, inputs, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::Complete as i64]
                );
                assert_eq!(result.candidates.len(), 1);
                let DryVacuumCandidate::Exterior(exterior) = &result.candidates[0] else {
                    return Err("Moving retained source lacks its exterior certificate".into());
                };
                assert_eq!(frozen::components(&exterior.area)?, [area]);
                assert_eq!(frozen::components(&exterior.discharge)?, [discharge]);
                assert_eq!(
                    frozen::components(exterior.recovery.requested_areas_m2())?,
                    [area]
                );
                assert_eq!(
                    exterior.query.query.original_rows.try_to_vec::<i64>()?,
                    [101]
                );
                let selected = hydraulic(
                    session
                        .sources
                        .select_rows(&exterior.parent_rows, &exterior.query),
                )?;
                let retained = hydraulic(crate::channel::flow::surveyed::error::checked::<R, _>(
                    client,
                    "retained exterior source expression",
                    |a| {
                        exterior.query.retained_endpoint_area(
                            a,
                            &selected.bands,
                            &selected.boundaries,
                        )
                    },
                ))?;
                assert_eq!(frozen::components(&retained)?, [area]);
                support::with_rectangle::<R, _>(
                    client,
                    1.,
                    4.,
                    &[[1., 3., 0., 0.]],
                    &config,
                    |other, _| {
                        assert!(other.select_rows(&owners, &exterior.query).is_err());
                        Ok(())
                    },
                )?;
                let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                client.copy_into(exterior.area.high(), &zero)?;
                client.copy_into(exterior.discharge.low(), &zero)?;
                client.copy_into(exterior.recovery.requested_areas_m2().low(), &zero)?;
                client.copy_into(
                    &exterior.query.query.original_rows,
                    &support::rows::<R>(client, &[99])?,
                )?;
                let before = work.snapshot();
                let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(work.snapshot(), before);
                let DryVacuumCandidate::Exterior(exterior) = &repeated.candidates[0] else {
                    return Err("Repeated retained source loses its exterior certificate".into());
                };
                assert_eq!(frozen::components(&exterior.area)?, [area]);
                assert_eq!(frozen::components(&exterior.discharge)?, [discharge]);
                assert_eq!(
                    frozen::components(exterior.recovery.requested_areas_m2())?,
                    [area]
                );
                assert_eq!(
                    exterior.query.query.original_rows.try_to_vec::<i64>()?,
                    [101]
                );
                Ok(())
            })
        }
        fn exhausted<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            support::with_rectangle::<R, _>(
                client,
                1.,
                4.,
                &[[1., 0., 0., 0.], [0., 0., 0., 0.]],
                &config,
                |pair, work| {
                    let parents = support::rows::<R>(client, &[0, 1])?;
                    let mut session = hydraulic(PointRootSession::new(pair, &parents, 1))?;
                    let owner = support::rows::<R>(client, &[0])?;
                    let result = hydraulic(session.certify_dry_vacuum(&owner))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [DryVacuumStatus::Pending as i64]
                    );
                    assert_eq!(
                        result.sonic_status[0].try_to_vec::<i64>()?,
                        [SourceSonicStatus::WorkLimit as i64]
                    );
                    assert_eq!(result.edges[0].head.available.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(result.edges[0].head_exact_zero.try_to_vec::<f64>()?, [0.]);
                    assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [0.]);
                    let before = work.snapshot();
                    hydraulic(session.certify_dry_vacuum(&owner))?;
                    assert_eq!(work.snapshot(), before);
                    assert_eq!(session.callback_points(), 0);
                    let dry =
                        hydraulic(session.certify_dry_vacuum(&support::rows::<R>(client, &[1])?))?;
                    assert_eq!(
                        dry.status.try_to_vec::<i64>()?,
                        [DryVacuumStatus::Complete as i64]
                    );
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }
    }
}
