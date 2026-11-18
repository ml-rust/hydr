use super::{
    dry_vacuum_types::{DryVacuumCertificates, DryVacuumRegime, DryVacuumStatus},
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::{SonicSide, SourceSonicStatus},
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    pub(super) fn certify_dry_vacuum(
        &mut self,
        invocation_owner: &Tensor<R>,
    ) -> HydraulicResult<DryVacuumCertificates<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(invalid(
                "dry vacuum candidates remain sealed after an earlier source sonic numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let (owners, parent) = checked::<R, _>(c, "dry vacuum invocation owner admission", |a| {
            super::dry_vacuum_evidence::owners(a, self, invocation_owner)
        })?;
        let invariant = self.sources.source_invariant_on_rows(&parent)?;
        let (regime, admitted) = checked::<R, _>(c, "dry vacuum retained regime", |a| {
            let selected = self.sources.select_sources(a, &parent)?;
            let input = selected.velocity.input();
            let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
            let leftwet = c.sub(&one, &input.left_dry)?;
            let rightwet = c.sub(&one, &input.right_dry)?;
            let leftonly = c.mul(&leftwet, &input.right_dry)?;
            let rightonly = c.mul(&rightwet, &input.left_dry)?;
            let code = |v: DryVacuumRegime| {
                Tensor::full_scalar(owners.shape(), DType::I64, v as i64 as f64, c.device())
            };
            let mut regime = code(DryVacuumRegime::Unresolved)?;
            for (mask, kind) in [
                (&invariant.wet_candidate, DryVacuumRegime::NotApplicable),
                (&invariant.vacuum, DryVacuumRegime::Vacuum),
                (&invariant.touching, DryVacuumRegime::Touching),
                (&leftonly, DryVacuumRegime::LeftWet),
                (&rightonly, DryVacuumRegime::RightWet),
                (&invariant.both_dry, DryVacuumRegime::BothDry),
            ] {
                regime = c.where_cond(mask, &code(kind)?, &regime)?;
            }
            let both = c.maximum(&invariant.vacuum, &invariant.touching)?;
            Ok((
                regime,
                [c.maximum(&leftonly, &both)?, c.maximum(&rightonly, &both)?],
            ))
        })?;
        for side in [SonicSide::Left, SonicSide::Right] {
            let state = c.index_select(&self.sonic.status[side.index()], 0, &owners)?;
            let missing = c.eq(
                &state,
                &Tensor::full_scalar(
                    &[],
                    DType::I64,
                    SourceSonicStatus::NotRequested as i64 as f64,
                    c.device(),
                )?,
            )?;
            let zero = Tensor::zeros(owners.shape(), admitted[side.index()].dtype(), c.device())?;
            let missing = c.where_cond(&missing, &admitted[side.index()], &zero)?;
            let rows = c.nonzero(&missing)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() != 0 {
                self.start_source_sonic(&c.index_select(&owners, 0, &rows)?, side)?;
            }
        }
        let edges = checked::<R, _>(c, "dry vacuum source wave evidence", |a| {
            Ok([
                super::dry_vacuum_evidence::edges(a, self, &owners, SonicSide::Left)?,
                super::dry_vacuum_evidence::edges(a, self, &owners, SonicSide::Right)?,
            ])
        })?;
        let possible = checked::<R, _>(c, "dry vacuum possible origin regions", |a| {
            super::dry_vacuum_positions::evaluate(a, &regime, &edges)
        })?;
        let demanded = checked::<R, _>(c, "dry vacuum demanded sonic owners", |a| {
            let zero = Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?;
            Ok([
                c.index_put(&zero, 0, &owners, &possible.left_fan)?,
                c.index_put(&zero, 0, &owners, &possible.right_fan)?,
            ])
        })?;
        let sonic = super::source_sonic_certificates::selected(self, [&demanded[0], &demanded[1]])?;
        let (candidates, complete) =
            super::dry_vacuum_candidates::assemble(self, &owners, &parent, &possible, sonic)?;
        let status = checked::<R, _>(c, "dry vacuum complete candidate status", |_a| {
            let code = |v: DryVacuumStatus| {
                Tensor::full_scalar(owners.shape(), DType::I64, v as i64 as f64, c.device())
            };
            let mut status = c.where_cond(
                &complete,
                &code(DryVacuumStatus::Complete)?,
                &code(DryVacuumStatus::Pending)?,
            )?;
            for (kind, outcome) in [
                (
                    DryVacuumRegime::NotApplicable,
                    DryVacuumStatus::NotApplicable,
                ),
                (
                    DryVacuumRegime::Unresolved,
                    DryVacuumStatus::RegimeUnresolved,
                ),
            ] {
                let mask = c.eq(
                    &regime,
                    &Tensor::full_scalar(&[], DType::I64, kind as i64 as f64, c.device())?,
                )?;
                status = c.where_cond(&mask, &code(outcome)?, &status)?;
            }
            Ok(status)
        })?;
        Ok(DryVacuumCertificates {
            invocation_witness: self.invocation.witness(),
            #[cfg(test)]
            sonic_status: [
                c.index_select(&self.sonic.status[0], 0, &owners)?,
                c.index_select(&self.sonic.status[1], 0, &owners)?,
            ],
            invocation_owner: owners,
            parent_rows: parent,
            regime,
            status,
            invariant: invariant.value,
            edges,
            possibilities: possible,
            candidates,
            callback_points: self.callback_points(),
            point_limit: self.point_limit(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed_points::{
        chart_fixture, dry_vacuum_types::DryVacuumCandidate, source_fixture_support as frozen,
        wet_depth::fixture as support,
    };
    use frozen::{Result, hydraulic};

    fn matrix<R>(client: &R::Client) -> Result
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
                let rows = support::rows::<R>(client, &[0, 1, 0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                let owners = support::rows::<R>(client, &[2, 1, 0])?;
                let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(result.invocation_owner.try_to_vec::<i64>()?, [2, 1, 0]);
                assert_eq!(result.parent_rows.try_to_vec::<i64>()?, [0, 1, 0]);
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [
                        DryVacuumRegime::LeftWet as i64,
                        DryVacuumRegime::BothDry as i64,
                        DryVacuumRegime::LeftWet as i64
                    ]
                );
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::Complete as i64; 3]
                );
                let fan = result
                    .candidates
                    .iter()
                    .find_map(|candidate| {
                        if let DryVacuumCandidate::Fan(fan) = candidate {
                            Some(fan)
                        } else {
                            None
                        }
                    })
                    .ok_or("Missing admitted sonic candidate")?;
                for bounds in frozen::endpoints(&fan.query.area)? {
                    assert!(bounds[0] <= 4. / 9. && 4. / 9. <= bounds[1]);
                }
                let dry = result
                    .candidates
                    .iter()
                    .find_map(|candidate| {
                        if let DryVacuumCandidate::Dry(dry) = candidate {
                            Some(dry)
                        } else {
                            None
                        }
                    })
                    .ok_or("Missing exact dry candidate")?;
                for bounds in frozen::endpoints(&dry.depth)? {
                    assert_eq!(bounds, [0., 0.]);
                }
                let points = session.callback_points();
                let before = work.snapshot().residual_points;
                let repeated = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(
                    repeated.status.try_to_vec::<i64>()?,
                    result.status.try_to_vec::<i64>()?
                );
                assert_eq!(session.callback_points(), points);
                assert_eq!(work.snapshot().residual_points, before);
                Ok(())
            },
        )?;
        references::run::<R>(client)?;
        controls::run::<R>(client)?;
        ownership::run::<R>(client)
    }
    #[test]
    fn configured_cpu_dry_vacuum_candidates_preserve_origin_alternatives() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        matrix::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_dry_vacuum_candidates_preserve_origin_alternatives() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }

    mod controls {
        use super::{
            DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus, PointRootSession, Result,
            chart_fixture, frozen, hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::source_sonic_types::{
            SonicSide, SourceSonicStatus,
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
            moving_contacts::<R>(client)?;
            retained_exterior::<R>(client)?;
            uncertain_front::<R>(client)?;
            uncertain_regime::<R>(client)?;
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
        fn moving_contacts<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let states = [
                [1., 2., 0., 0.],
                [0., 0., 1., -2.],
                [1., 1., 0., 0.],
                [0., 0., 1., -1.],
                [1., -2., 0., 0.],
                [0., 0., 1., 2.],
                [1., -1., 1., 3.],
                [1., -3., 1., 1.],
            ];
            support::with_rectangle::<R, _>(client, 1., 4., &states, &config, |pair, _| {
                let parents = support::rows::<R>(client, &[0, 1, 2, 3, 4, 5, 6, 7])?;
                let mut session = hydraulic(PointRootSession::new(
                    pair,
                    &parents,
                    config.root.max_points,
                ))?;
                for owner in 0..8 {
                    let owners = support::rows::<R>(client, &[owner])?;
                    let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                    hydraulic(session.invocation.check(&result.invocation_witness))?;
                    assert_eq!(
                        result.status.try_to_vec::<i64>()?,
                        [DryVacuumStatus::Complete as i64],
                        "owner {owner}"
                    );
                    assert_eq!(result.callback_points, session.callback_points());
                    assert_eq!(result.point_limit, config.root.max_points);
                    let p = &result.possibilities;
                    let masks = [
                        &p.left_exterior,
                        &p.left_fan,
                        &p.dry,
                        &p.right_fan,
                        &p.right_exterior,
                    ]
                    .map(|m| m.try_to_vec::<f64>())
                    .into_iter()
                    .collect::<std::result::Result<Vec<_>, _>>()?;
                    let expected = match owner {
                        0 => [1., 0., 0., 0., 0.],
                        1 => [0., 0., 0., 0., 1.],
                        2 => [1., 1., 0., 0., 0.],
                        3 => [0., 0., 0., 1., 1.],
                        4 => [0., 1., 1., 0., 0.],
                        5 => [0., 0., 1., 1., 0.],
                        6 => [0., 1., 0., 0., 0.],
                        7 => [0., 0., 0., 1., 0.],
                        _ => return Err("Unexpected moving control owner".into()),
                    };
                    for (actual, expected) in masks.iter().zip(expected) {
                        assert_eq!(actual, &[expected], "owner {owner}");
                    }
                    for candidate in &result.candidates {
                        match candidate {
                            DryVacuumCandidate::Exterior(exterior) => {
                                let source = states[usize::try_from(owner)?];
                                let i = exterior.side.index() * 2;
                                assert_eq!(exterior.invocation_owner.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(exterior.parent_rows.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(
                                    frozen::components(&exterior.area)?,
                                    [[source[i].to_bits(), 0, 0]]
                                );
                                assert_eq!(
                                    frozen::components(&exterior.discharge)?,
                                    [[source[i + 1].to_bits(), 0, 0]]
                                );
                                assert_eq!(
                                    frozen::components(&exterior.velocity)?,
                                    [[source[i + 1].to_bits(), 0, 0]]
                                );
                                assert_eq!(frozen::endpoints(&exterior.query.area)?, [[1., 1.]]);
                                let depth = frozen::endpoints(exterior.recovery.depth_m())?[0];
                                assert!(depth[0] <= 1. && 1. <= depth[1]);
                            }
                            DryVacuumCandidate::Dry(dry) => {
                                assert_eq!(dry.invocation_owner.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(dry.parent_rows.try_to_vec::<i64>()?, [owner]);
                                assert_eq!(frozen::endpoints(&dry.query.area)?, [[0., 0.]]);
                            }
                            DryVacuumCandidate::Fan(fan) => {
                                assert!(owner == 6 || owner == 7);
                                let expected =
                                    [0x3fbc71c71c71c71c, 0x3fbc71c71c71c71d].map(f64::from_bits);
                                let area = frozen::endpoints(&fan.query.area)?[0];
                                assert!(area[0] <= expected[0] && expected[1] <= area[1]);
                            }
                            DryVacuumCandidate::PendingFan(_) => {
                                return Err("Exact moving control leaves pending fan".into());
                            }
                        }
                    }
                    if owner >= 6 {
                        assert_eq!(
                            result.regime.try_to_vec::<i64>()?,
                            [DryVacuumRegime::Touching as i64]
                        );
                        assert_eq!(frozen::components(&result.invariant.retained)?, [[0; 3]]);
                        assert_eq!(result.invariant.available.try_to_vec::<f64>()?, [1.]);
                        let speed = if owner == 6 { 1. } else { -1. };
                        for edge in &result.edges {
                            let front = frozen::endpoints(&edge.front.physical)?[0];
                            assert!(front[0] <= speed && speed <= front[1]);
                            assert_eq!(edge.front_exact_zero.try_to_vec::<f64>()?, [0.]);
                        }
                    }
                    if owner == 2 || owner == 3 {
                        let side = if owner == 2 { 0 } else { 1 };
                        assert_eq!(
                            result.edges[side].head_exact_zero.try_to_vec::<f64>()?,
                            [1.]
                        );
                        assert_eq!(
                            result.sonic_status[side].try_to_vec::<i64>()?,
                            [SourceSonicStatus::SourceContact as i64]
                        );
                        assert!(
                            !result
                                .candidates
                                .iter()
                                .any(|v| matches!(v, DryVacuumCandidate::Fan(_)))
                        );
                    }
                    if owner == 4 || owner == 5 {
                        let side = if owner == 4 { 0 } else { 1 };
                        assert_eq!(
                            result.edges[side].front_exact_zero.try_to_vec::<f64>()?,
                            [1.]
                        );
                        assert_eq!(
                            result.sonic_status[side].try_to_vec::<i64>()?,
                            [SourceSonicStatus::DryContact as i64]
                        );
                    }
                }
                Ok(())
            })
        }
        fn uncertain_front<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let section = chart_fixture::rectangle::<R>(client, 3., 4.)?;
            let owners = support::rows::<R>(client, &[0])?;
            // Exact Q=-6+2^-1074 gives a positive, unrepresentable epsilon=2^-1074/3.
            let input = [
                [3.0_f64.to_bits(), 0, 0],
                [(-6.0_f64).to_bits(), 1, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
            .iter()
            .map(|v| frozen::expansion::<R>(client, &[*v], true))
            .collect::<Result<Vec<_>>>()?;
            chart_fixture::with_inputs(client, section, owners, input, &config, |pair, _| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [DryVacuumRegime::LeftWet as i64]
                );
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::Pending as i64]
                );
                let front = frozen::endpoints(&result.edges[0].front.physical)?[0];
                assert!(front[0] <= 0. && front[1] >= 0.);
                assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [0.]);
                assert_eq!(result.possibilities.left_fan.try_to_vec::<f64>()?, [1.]);
                assert_eq!(result.possibilities.dry.try_to_vec::<f64>()?, [1.]);
                let pending = result
                    .candidates
                    .iter()
                    .find_map(|v| {
                        if let DryVacuumCandidate::PendingFan(p) = v {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .ok_or("Uncertain front loses pending fan")?;
                assert_eq!(pending.side, SonicSide::Left);
                assert_eq!(pending.invocation_owner.try_to_vec::<i64>()?, [0]);
                assert_eq!(pending.parent_rows.try_to_vec::<i64>()?, [0]);
                assert_eq!(
                    pending.source_status.try_to_vec::<i64>()?,
                    [SourceSonicStatus::Unresolved as i64]
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

        fn uncertain_regime<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            let config = chart_fixture::config();
            let section = chart_fixture::rectangle::<R>(client, 3., 4.)?;
            let owners = support::rows::<R>(client, &[0])?;
            // Equal low discharges cancel physically, while separate quotient radii do not prove equality.
            let inputs = [
                [3.0_f64.to_bits(), 0, 0],
                [(-6.0_f64).to_bits(), 1, 0],
                [3.0_f64.to_bits(), 0, 0],
                [6.0_f64.to_bits(), 1, 0],
            ]
            .iter()
            .map(|v| frozen::expansion::<R>(client, &[*v], true))
            .collect::<Result<Vec<_>>>()?;
            chart_fixture::with_inputs(client, section, owners, inputs, &config, |pair, work| {
                let owners = support::rows::<R>(client, &[0])?;
                let mut session =
                    hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                assert_eq!(
                    result.regime.try_to_vec::<i64>()?,
                    [DryVacuumRegime::Unresolved as i64]
                );
                assert_eq!(
                    result.status.try_to_vec::<i64>()?,
                    [DryVacuumStatus::RegimeUnresolved as i64]
                );
                let invariant = frozen::endpoints(&result.invariant.physical)?[0];
                assert!(invariant[0] <= 0. && invariant[1] >= 0.);
                assert!(
                    frozen::components(&result.invariant.retained)?[0]
                        .iter()
                        .any(|&v| f64::from_bits(v) != 0.)
                );
                assert!(result.candidates.is_empty());
                assert_eq!(session.callback_points(), 0);
                assert_eq!(work.snapshot().residual_points, 0);
                for status in &result.sonic_status {
                    assert_eq!(
                        status.try_to_vec::<i64>()?,
                        [SourceSonicStatus::NotRequested as i64]
                    );
                }
                Ok(())
            })
        }
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

    mod references {
        use super::{
            DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus, PointRootSession, Result,
            chart_fixture, frozen, hydraulic, support,
        };
        use crate::channel::flow::surveyed_points::source_sonic_types::{
            SonicFanMembership, SonicSide, SourceSonicStatus,
        };
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

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            for reference in REFERENCES {
                let config = chart_fixture::config();
                chart_fixture::with_pair::<R, _>(
                    client,
                    &[reference.id],
                    &config,
                    |pair, work| {
                        let cap = pair.bands.cap_depth.upper().try_to_vec::<f64>()?[0];
                        let owners = support::rows::<R>(client, &[0])?;
                        let mut session = hydraulic(PointRootSession::new(
                            pair,
                            &owners,
                            config.root.max_points,
                        ))?;
                        let initial = hydraulic(session.certify_dry_vacuum(&owners))?;
                        assert_eq!(
                            initial.regime.try_to_vec::<i64>()?,
                            [if reference.side == SonicSide::Left {
                                DryVacuumRegime::LeftWet
                            } else {
                                DryVacuumRegime::RightWet
                            } as i64]
                        );
                        if f64::from_bits(reference.depth[0]) < 1e-30 {
                            let atol = (f64::from_bits(reference.depth[0]) / cap) * 1e-12;
                            assert!(atol > 0. && atol < config.root.atol);
                            hydraulic(
                                session
                                    .refine_source_sonic(CertifiedRootTolerance { atol, rtol: 0. }),
                            )?;
                        }
                        let before = work.snapshot();
                        let points = session.callback_points();
                        let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                        assert_eq!(
                            result.status.try_to_vec::<i64>()?,
                            [DryVacuumStatus::Complete as i64],
                            "{}",
                            reference.id
                        );
                        assert_eq!(session.callback_points(), points);
                        assert_eq!(work.snapshot().residual_points, before.residual_points);
                        assert_eq!(work.snapshot().root_iterations, before.root_iterations);
                        assert_eq!(
                            result.sonic_status[1 - reference.side.index()].try_to_vec::<i64>()?,
                            [SourceSonicStatus::NotRequested as i64]
                        );
                        let cache = chart_fixture::cache(session.sources())?;
                        assert!(
                            cache[1 - reference.side.index()]
                                .ready
                                .iter()
                                .all(|&v| v == 0.)
                        );
                        let fan = result
                            .candidates
                            .iter()
                            .find_map(|v| {
                                if let DryVacuumCandidate::Fan(fan) = v {
                                    Some(fan)
                                } else {
                                    None
                                }
                            })
                            .ok_or("Missing frozen sonic candidate")?;
                        assert_eq!(fan.side, reference.side);
                        assert_eq!(fan.fan_membership, SonicFanMembership::NotChecked);
                        assert_eq!(fan.status.try_to_vec::<i64>()?, [0]);
                        for (actual, expected) in [
                            (&fan.query.area, reference.area),
                            (&fan.depth, reference.depth),
                            (&fan.velocity, reference.velocity),
                            (&fan.discharge, reference.discharge),
                        ] {
                            let actual = frozen::endpoints(actual)?[0];
                            let expected = expected.map(f64::from_bits);
                            assert!(
                                actual[0] <= expected[0] && expected[1] <= actual[1],
                                "{} {actual:?} excludes {expected:?}",
                                reference.id
                            );
                            if f64::from_bits(reference.depth[0]) < 1e-30 {
                                assert!(
                                    (actual[1] - actual[0]) / expected[0].abs() < 1e-10,
                                    "{} {actual:?}",
                                    reference.id
                                );
                            }
                        }
                        let edges = &result.edges[reference.side.index()];
                        let head = frozen::endpoints(&edges.head.physical)?[0];
                        let front = frozen::endpoints(&edges.front.physical)?[0];
                        if reference.side == SonicSide::Left {
                            assert!(head[1] < 0. && front[0] > 0.);
                        } else {
                            assert!(head[0] > 0. && front[1] < 0.);
                        }
                        Ok(())
                    },
                )?;
            }
            regimes::<R>(client)
        }
        fn regimes<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: support::DepthClient<R>,
        {
            for (id, regime) in [
                ("scalar_three_bands_both_dry", DryVacuumRegime::BothDry),
                ("scalar_rectangle_vacuum", DryVacuumRegime::Vacuum),
                (
                    "scalar_rectangle_vacuum_threshold",
                    DryVacuumRegime::Touching,
                ),
                (
                    "scalar_rectangle_vacuum_threshold_outside",
                    DryVacuumRegime::Vacuum,
                ),
                (
                    "scalar_rectangle_vacuum_threshold_inside",
                    DryVacuumRegime::NotApplicable,
                ),
                (
                    "retained_vacuum_inside_film",
                    DryVacuumRegime::NotApplicable,
                ),
            ] {
                let config = chart_fixture::config();
                chart_fixture::with_pair::<R, _>(client, &[id], &config, |pair, work| {
                    let owners = support::rows::<R>(client, &[0])?;
                    let mut session =
                        hydraulic(PointRootSession::new(pair, &owners, config.root.max_points))?;
                    let before = work.snapshot();
                    let result = hydraulic(session.certify_dry_vacuum(&owners))?;
                    assert_eq!(result.regime.try_to_vec::<i64>()?, [regime as i64], "{id}");
                    if regime == DryVacuumRegime::NotApplicable {
                        assert_eq!(
                            result.status.try_to_vec::<i64>()?,
                            [DryVacuumStatus::NotApplicable as i64]
                        );
                        assert!(result.candidates.is_empty());
                        assert_eq!(session.callback_points(), 0);
                        for side in &result.sonic_status {
                            assert_eq!(
                                side.try_to_vec::<i64>()?,
                                [SourceSonicStatus::NotRequested as i64]
                            );
                        }
                    } else {
                        assert_eq!(
                            result.status.try_to_vec::<i64>()?,
                            [DryVacuumStatus::Complete as i64],
                            "{id}"
                        );
                        let dry = result
                            .candidates
                            .iter()
                            .find_map(|v| {
                                if let DryVacuumCandidate::Dry(dry) = v {
                                    Some(dry)
                                } else {
                                    None
                                }
                            })
                            .ok_or("Missing frozen dry candidate")?;
                        assert_eq!(frozen::components(&dry.area)?, [[0; 3]]);
                        assert_eq!(frozen::components(&dry.discharge)?, [[0; 3]]);
                        assert_eq!(frozen::endpoints(&dry.depth)?, [[0., 0.]]);
                        if regime == DryVacuumRegime::BothDry {
                            assert_eq!(session.callback_points(), 0);
                            assert_eq!(work.snapshot().residual_points, before.residual_points);
                        }
                        if regime == DryVacuumRegime::Vacuum {
                            let left = frozen::endpoints(&result.edges[0].front.physical)?[0];
                            let right = frozen::endpoints(&result.edges[1].front.physical)?[0];
                            if id == "scalar_rectangle_vacuum" {
                                assert!(left[1] < 0. && right[0] > 0.);
                                assert!(left[0] <= -1. && -1. <= left[1]);
                                assert!(right[0] <= 1. && 1. <= right[1]);
                            } else {
                                // Frozen outside-threshold edges retain a left contact and right front 2^-51.
                                assert_eq!(
                                    result.edges[0].front_exact_zero.try_to_vec::<f64>()?,
                                    [1.]
                                );
                                assert_eq!(
                                    frozen::components(&result.edges[0].front.retained)?,
                                    [[0; 3]]
                                );
                                assert!(left[0] <= 0. && 0. <= left[1]);
                                let reference = f64::from_bits(0x3cc0000000000000);
                                assert!(right[0] > 0.);
                                assert!(right[0] <= reference && reference <= right[1]);
                                assert_eq!(
                                    frozen::components(&result.edges[1].front.retained)?,
                                    [[0x3cc0000000000000, 0, 0]]
                                );
                                assert_eq!(
                                    result.sonic_status[0].try_to_vec::<i64>()?,
                                    [SourceSonicStatus::DryContact as i64]
                                );
                                assert_eq!(
                                    result.sonic_status[1].try_to_vec::<i64>()?,
                                    [SourceSonicStatus::NoRootDry as i64]
                                );
                            }
                        }
                        if regime == DryVacuumRegime::Touching {
                            assert_eq!(result.edges[0].front_exact_zero.try_to_vec::<f64>()?, [1.]);
                            assert_eq!(result.edges[1].front_exact_zero.try_to_vec::<f64>()?, [1.]);
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
    }
}
