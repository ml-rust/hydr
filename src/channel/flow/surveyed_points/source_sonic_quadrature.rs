use super::{
    point_root_session::{PointRootSession, invalid},
    source_sonic_quadrature_types::{SourceSonicQuadratureOutcome, SourceSonicQuadratureProgress},
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
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
    pub(super) fn advance_source_sonic_quadrature(
        &mut self,
        owners: &Tensor<R>,
        side: SonicSide,
    ) -> HydraulicResult<SourceSonicQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(invalid(
                "sonic quadrature remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let mut progress =
            checked::<R, _>(c, "sonic quadrature complete request admission", |a| {
                let (owners, parents) = super::dry_vacuum_evidence::owners(a, self, owners)?;
                let count = owners.numel();
                Ok(SourceSonicQuadratureProgress {
                    invocation_owner: owners,
                    parent_rows: parents,
                    side,
                    outcome: Tensor::full_scalar(
                        &[count],
                        DType::I64,
                        SourceSonicQuadratureOutcome::NotApplicable as i64 as f64,
                        c.device(),
                    )?,
                    old_level: Tensor::full_scalar(&[count], DType::I64, -1., c.device())?,
                    new_level: Tensor::full_scalar(&[count], DType::I64, -1., c.device())?,
                    bracket_improved: Tensor::zeros(&[count], a.dtype(), c.device())?,
                    additional_residual_points: 0,
                    additional_simpson_value_points: 0,
                    additional_simpson_derivative_points: 0,
                    additional_mapped_points: 0,
                    callback_points: self.callback_points(),
                    point_limit: self.point_limit(),
                })
            })?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) = checked::<R, _>(c, "sonic quadrature nonempty demand", |a| {
            let owners = &progress.invocation_owner;
            let demand = c.index_put(
                &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                0,
                owners,
                &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
            )?;
            let association =
                super::point_flux_association::inverse(a, self.invocation_rows.numel(), owners)?;
            Ok((demand, association))
        })?;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = mapped::<R>(&self.sonic.requests)?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            for request in &mut self.sonic.requests {
                if request.binding.side != side {
                    continue;
                }
                if c.nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                    .numel()
                    == 0
                {
                    continue;
                }
                if let Err(error) = super::source_sonic_quadrature_request::advance(
                    self.sources,
                    &mut self.invocation,
                    request,
                    &demand,
                    &association,
                    &mut progress,
                ) {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result?;
        let after = self.sources.evaluation.work.snapshot();
        progress.additional_residual_points = delta(before.residual_points, after.residual_points)?;
        progress.additional_simpson_value_points =
            delta(before.simpson_value_points, after.simpson_value_points)?;
        progress.additional_simpson_derivative_points = delta(
            before.simpson_derivative_points,
            after.simpson_derivative_points,
        )?;
        progress.additional_mapped_points = delta(old_mapped, mapped::<R>(&self.sonic.requests)?)?;
        progress.callback_points = self.callback_points();
        Ok(progress)
    }
}
fn delta(before: usize, after: usize) -> HydraulicResult<usize> {
    after
        .checked_sub(before)
        .ok_or_else(|| invalid("sonic quadrature cumulative work decreased"))
}
fn mapped<R: Runtime<DType = DType>>(
    requests: &[super::source_sonic_types::SourceSonicRequest<R>],
) -> HydraulicResult<usize> {
    requests
        .iter()
        .flat_map(|request| &request.domains)
        .try_fold(0usize, |sum, domain| {
            sum.checked_add(domain.mapped_points)
                .ok_or_else(|| invalid("sonic mapped work sum overflows usize"))
        })
}

#[cfg(test)]
#[path = "source_sonic_quadrature_tests.rs"]
mod tests;
