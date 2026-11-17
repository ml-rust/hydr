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
#[path = "dry_vacuum_tests.rs"]
mod tests;
