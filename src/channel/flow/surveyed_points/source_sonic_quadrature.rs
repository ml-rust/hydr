use super::{
    point_quadrature_progress,
    point_quadrature_types::PointQuadratureProgress,
    point_root_session::{PointRootSession, invalid},
    source_sonic_types::SonicSide,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
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
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(invalid(
                "sonic quadrature remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let mut progress = point_quadrature_progress::start(
            self,
            owners,
            Some(side),
            "sonic quadrature complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) =
            point_quadrature_progress::demand(self, &progress, "sonic quadrature nonempty demand")?;
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
        point_quadrature_progress::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [old_mapped, mapped::<R>(&self.sonic.requests)?],
            self.callback_points(),
        )?;
        Ok(progress)
    }
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
pub(in crate::channel::flow::surveyed_points) mod tests;
