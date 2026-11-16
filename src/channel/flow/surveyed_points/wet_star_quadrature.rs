use super::{
    point_quadrature_progress,
    point_quadrature_types::PointQuadratureProgress,
    point_root_session::{PointRootSession, invalid},
    wet_depth_types::WetDepthDomain,
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{integrate::CertifiedSimpsonAlgorithms, optimize::CertifiedMappedBisectionAlgorithms};

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
        + CertifiedMappedBisectionAlgorithms<R>,
{
    /// Advances each requested wet star row one Simpson level and re-refines its root bracket.
    /// Rows without a wet depth domain record `NotApplicable`. Star endpoints and candidates
    /// follow the new level at the next selection.
    #[cfg_attr(
        not(test),
        expect(
            dead_code,
            reason = "the point flux controller does not drive star rows yet"
        )
    )]
    pub(super) fn advance_wet_star_quadrature(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal {
            return Err(invalid(
                "wet star quadrature remains sealed after an earlier numerical error",
            ));
        }
        let mut progress = point_quadrature_progress::start(
            self,
            owners,
            None,
            "wet star quadrature complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) = point_quadrature_progress::demand(
            self,
            &progress,
            "wet star quadrature nonempty demand",
        )?;
        let tolerance = self.wet.tolerance;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = mapped(&self.wet.domains)?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for domain in std::mem::take(&mut self.wet.domains) {
                output.extend(super::wet_star_quadrature_request::advance(
                    self.sources,
                    &mut self.invocation,
                    domain,
                    &tolerance,
                    &demand,
                    &association,
                    &mut progress,
                )?);
            }
            self.wet.domains = output;
            Ok(())
        })();
        if result.is_err() {
            self.wet.terminal = true;
        }
        self.reconcile(checkpoint, result.is_ok())?;
        result?;
        point_quadrature_progress::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [old_mapped, mapped(&self.wet.domains)?],
            self.callback_points(),
        )?;
        Ok(progress)
    }
}

fn mapped<R: Runtime<DType = DType>>(domains: &[WetDepthDomain<R>]) -> HydraulicResult<usize> {
    domains.iter().try_fold(0usize, |sum, domain| {
        sum.checked_add(domain.mapped_points)
            .ok_or_else(|| invalid("wet mapped work sum overflows usize"))
    })
}

#[cfg(test)]
#[path = "wet_star_quadrature_tests.rs"]
mod tests;
