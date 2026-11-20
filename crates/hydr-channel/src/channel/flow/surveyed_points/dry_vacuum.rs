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
            super::dry_vacuum_regime::retained(a, self, &owners, &parent, &invariant)
        })?;
        let mirror = checked::<R, _>(c, "dry vacuum mirror faces", |a| {
            super::dry_vacuum_mirror::rows(a, self, &owners, &parent, &regime, &invariant.value)
        })?;
        let (regime, admitted) = checked::<R, _>(c, "dry vacuum mirror regime", |a| {
            let overlap = &mirror.overlap;
            Ok((
                super::dry_vacuum_mirror::regime(a, &regime, &mirror)?,
                [
                    c.maximum(&admitted[0], overlap)?,
                    c.maximum(&admitted[1], overlap)?,
                ],
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
            let possible = super::dry_vacuum_positions::evaluate(a, &regime, &edges)?;
            super::dry_vacuum_mirror::restrict(a, &regime, &mirror, possible)
        })?;
        let demanded = checked::<R, _>(c, "dry vacuum demanded sonic owners", |a| {
            let zero = Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?;
            Ok([
                c.index_put(&zero, 0, &owners, &possible.left_fan)?,
                c.index_put(&zero, 0, &owners, &possible.right_fan)?,
            ])
        })?;
        let sonic = super::source_sonic_certificates::selected(self, [&demanded[0], &demanded[1]])?;
        let (candidates, complete) = super::dry_vacuum_candidates::assemble(
            self, &owners, &parent, &possible, sonic, &mirror,
        )?;
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
        Ok(())
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
}
