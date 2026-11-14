use super::{
    dry_vacuum_types::{DryVacuumCandidate, DryVacuumRegime, DryVacuumStatus},
    point_candidate_types::{
        PointCandidate, PointCandidateCertificates, PointCandidateStatus, PointDryEvidence,
        PointFan, PointFanAuthority, PointPendingReason, PointRegion,
    },
    point_root_session::{PointRootSession, invalid},
    wet_root_types::WetRootStatus,
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
    pub(super) fn certify_point_candidates(
        &mut self,
        invocation_owner: &Tensor<R>,
    ) -> HydraulicResult<PointCandidateCertificates<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal || self.sonic.requests.iter().any(|r| r.terminal) {
            return Err(invalid(
                "point candidates remain sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let (owners, parent) =
            checked::<R, _>(c, "point candidate complete owner admission", |a| {
                super::dry_vacuum_evidence::owners(a, self, invocation_owner)
            })?;
        super::point_candidates_admission::check(self)?;
        let dry = self.certify_dry_vacuum(&owners)?;
        let (wet, demand) = checked::<R, _>(c, "point candidate strict wet regime routing", |a| {
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
            let wet = c.where_cond(
                &c.eq(
                    &dry.regime,
                    &Tensor::full_scalar(
                        &[],
                        DType::I64,
                        DryVacuumRegime::NotApplicable as i64 as f64,
                        c.device(),
                    )?,
                )?,
                &one,
                &zero,
            )?;
            let demand = c.index_put(
                &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                0,
                &owners,
                &wet,
            )?;
            Ok((wet, demand))
        })?;
        let status = c.index_select(&self.wet.locator_status, 0, &owners)?;
        let missing = c.eq(
            &status,
            &Tensor::full_scalar(
                &[],
                DType::I64,
                WetRootStatus::NotRequested as i64 as f64,
                c.device(),
            )?,
        )?;
        let missing = c.where_cond(
            &missing,
            &wet,
            &Tensor::zeros(wet.shape(), wet.dtype(), c.device())?,
        )?;
        let rows = c.nonzero(&missing)?;
        let rows = rows.reshape(&[rows.numel()])?;
        if rows.numel() != 0 {
            self.start_wet(&c.index_select(&owners, 0, &rows)?)?;
        }
        let waves = self.certify_wet_waves_selected(&demand)?;
        let mut wet_result = super::wet_candidates::evaluate(self, &waves)?;
        let mut candidates = Vec::new();
        checked::<R, _>(c, "point candidate admitted dry payload transfer", |a| {
            for candidate in dry.candidates {
                match candidate {
                    DryVacuumCandidate::Exterior(value) => {
                        candidates.push(PointCandidate::Exterior(value))
                    }
                    DryVacuumCandidate::Dry(value) => candidates.push(PointCandidate::Dry(value)),
                    DryVacuumCandidate::Fan(certificate) => {
                        let zero =
                            Tensor::zeros(certificate.status.shape(), a.dtype(), c.device())?;
                        let one = Tensor::full_scalar(
                            certificate.status.shape(),
                            a.dtype(),
                            1.,
                            c.device(),
                        )?;
                        let active = c.where_cond(
                            &c.eq(
                                &certificate.status,
                                &Tensor::zeros(&[], DType::I64, c.device())?,
                            )?,
                            &one,
                            &zero,
                        )?;
                        let region =
                            if certificate.side == super::source_sonic_types::SonicSide::Left {
                                PointRegion::LeftFan
                            } else {
                                PointRegion::RightFan
                            };
                        super::wet_candidates::pending(
                            a,
                            &mut candidates,
                            &certificate.invocation_owner,
                            &certificate.parent_rows,
                            &c.sub(&one, &active)?,
                            region,
                            PointPendingReason::FanCertificate,
                        )?;
                        candidates.push(PointCandidate::Fan(Box::new(PointFan {
                            certificate: *certificate,
                            active,
                            authority: PointFanAuthority::DryVacuum,
                        })));
                    }
                    DryVacuumCandidate::PendingFan(value) => {
                        let one = Tensor::full_scalar(
                            value.invocation_owner.shape(),
                            a.dtype(),
                            1.,
                            c.device(),
                        )?;
                        let region = if value.side == super::source_sonic_types::SonicSide::Left {
                            PointRegion::LeftFan
                        } else {
                            PointRegion::RightFan
                        };
                        super::wet_candidates::pending(
                            a,
                            &mut candidates,
                            &value.invocation_owner,
                            &value.parent_rows,
                            &one,
                            region,
                            PointPendingReason::FanCertificate,
                        )?;
                    }
                }
            }
            let located = c.index_select(&wet_result.located, 0, &owners)?;
            let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
            let missing = c.mul(&wet, &c.eq(&located, &zero)?)?;
            super::wet_candidates::pending(
                a,
                &mut wet_result.candidates,
                &owners,
                &parent,
                &missing,
                PointRegion::Unlocated,
                PointPendingReason::WetRoot,
            )?;
            Ok(())
        })?;
        candidates.extend(wet_result.candidates);
        let (depth, evidence) = super::point_candidates_evidence::snapshot(self, &demand)?;
        let status = checked::<R, _>(c, "point candidate complete physical alternatives", |a| {
            let code = |v: PointCandidateStatus| {
                Tensor::full_scalar(owners.shape(), DType::I64, v as i64 as f64, c.device())
            };
            let drycomplete = c.eq(
                &dry.status,
                &Tensor::full_scalar(
                    &[],
                    DType::I64,
                    DryVacuumStatus::Complete as i64 as f64,
                    c.device(),
                )?,
            )?;
            let drycomplete = c.where_cond(
                &drycomplete,
                &Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?,
                &Tensor::zeros(owners.shape(), a.dtype(), c.device())?,
            )?;
            let validwet = c.index_select(&wet_result.complete, 0, &owners)?;
            let coverage = c.index_select(&wet_result.covered, 0, &owners)?;
            let wetcomplete = c.mul(&validwet, &coverage)?;
            let complete = c.where_cond(&wet, &wetcomplete, &drycomplete)?;
            let mut status = c.where_cond(
                &complete,
                &code(PointCandidateStatus::Complete)?,
                &code(PointCandidateStatus::Pending)?,
            )?;
            status = c.where_cond(
                &c.eq(
                    &dry.status,
                    &Tensor::full_scalar(
                        &[],
                        DType::I64,
                        DryVacuumStatus::RegimeUnresolved as i64 as f64,
                        c.device(),
                    )?,
                )?,
                &code(PointCandidateStatus::RegimeUnresolved)?,
                &status,
            )?;
            Ok(status)
        })?;
        Ok(PointCandidateCertificates {
            invocation_witness: self.invocation.witness(),
            wet_status: c.index_select(&self.wet.locator_status, 0, &owners)?,
            depth_status: c.index_select(&depth, 0, &owners)?,
            sonic_status: [
                c.index_select(&self.sonic.status[0], 0, &owners)?,
                c.index_select(&self.sonic.status[1], 0, &owners)?,
            ],
            regime: dry.regime.clone(),
            status,
            invocation_owner: owners,
            parent_rows: parent,
            dry_vacuum: PointDryEvidence {
                regime: dry.regime,
                status: dry.status,
                invariant: dry.invariant,
                edges: dry.edges,
                possibilities: dry.possibilities,
            },
            wet_evidence: evidence,
            waves,
            candidates,
            fan_evidence: wet_result.evidence,
            callback_points: self.callback_points(),
            point_limit: self.point_limit(),
        })
    }
}

#[cfg(test)]
#[path = "point_candidates_tests.rs"]
pub(in crate::channel::flow::surveyed_points) mod tests;
