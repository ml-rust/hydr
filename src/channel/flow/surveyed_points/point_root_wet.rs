use super::{
    point_root_session::PointRootSession, source_sonic_binding, wet_root_types::WetRootStatus,
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
    pub(super) fn start_wet(&mut self, invocation_owner: &Tensor<R>) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet state remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let (owners, rows) = checked::<R, _>(c, "point root requested wet subset", |a| {
            let (owners, rows) = source_sonic_binding::owners(
                a,
                self,
                invocation_owner,
                &self.wet.locator_status,
                WetRootStatus::NotRequested as i64,
            )?;
            let selected = self.sources.select_sources(a, &rows)?;
            let zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
            if c.nonzero(&c.eq(&selected.velocity.both_wet, &zero)?)?
                .numel()
                != 0
            {
                return Err(super::chart_admission::unresolved(
                    "Wet root locator requires two wet sources. Classify dry sources before locating domains",
                ));
            }
            Ok((owners, rows))
        })?;
        if owners.numel() == 0 {
            return Ok(());
        }
        self.wet.locator_status = c.index_put(
            &self.wet.locator_status,
            0,
            &owners,
            &Tensor::full_scalar(
                owners.shape(),
                DType::I64,
                WetRootStatus::Unresolved as i64 as f64,
                c.device(),
            )?,
        )?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let mut located =
                super::wet_roots::locate_on_invocation(self.sources, &rows, &mut self.invocation)?;
            for domain in &mut located.domains {
                domain.invocation_owner = c.index_select(&owners, 0, &domain.invocation_owner)?;
            }
            #[cfg(test)]
            for evidence in &mut located.evidence {
                evidence.invocation_owner =
                    c.index_select(&owners, 0, &evidence.invocation_owner)?;
            }
            let domains =
                super::wet_depth::capture_domains(self.sources, &self.invocation, located.domains)?;
            self.wet.locator_status =
                c.index_put(&self.wet.locator_status, 0, &owners, &located.status)?;
            #[cfg(test)]
            self.wet.evidence.extend(located.evidence);
            self.wet.domains.extend(domains);
            self.refine_wet_depth_in_place(self.wet.tolerance)
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        if result.is_err() {
            self.wet.terminal = true;
        }
        result
    }
}
