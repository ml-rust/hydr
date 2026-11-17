#[cfg(test)]
use super::source_sonic_types::SourceSonicCertificates;
use super::{
    point_root_session::PointRootSession,
    source_sonic_binding,
    source_sonic_types::{SonicSide, SourceSonicBinding, SourceSonicRequest, SourceSonicStatus},
    wet_depth_types::WetDepthStatus,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance, CertifiedUpperBracketAlgorithms,
    },
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
    pub(super) fn start_source_sonic(
        &mut self,
        invocation_owner: &Tensor<R>,
        side: SonicSide,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        let c = self.sources.geometry.client();
        let (owners, rows) = checked::<R, _>(c, "source sonic request owner admission", |a| {
            source_sonic_binding::owners(
                a,
                self,
                invocation_owner,
                &self.sonic.status[side.index()],
                SourceSonicStatus::NotRequested as i64,
            )
        })?;
        if owners.numel() == 0 {
            return Ok(());
        }
        let tolerance = CertifiedRootTolerance {
            atol: self.invocation.source_witness().config().root.atol,
            rtol: self.invocation.source_witness().config().root.rtol,
        };
        let request = checked::<R, _>(c, "source sonic request registration", |a| {
            let status = Tensor::full_scalar(
                owners.shape(),
                DType::I64,
                SourceSonicStatus::Unresolved as i64 as f64,
                c.device(),
            )?;
            Ok(SourceSonicRequest {
                binding: SourceSonicBinding::new(&self.invocation, side, owners, rows),
                status,
                dry_residual: super::wave_curve_types::unavailable(a, invocation_owner.numel())?,
                source_residual: super::wave_curve_types::unavailable(a, invocation_owner.numel())?,
                domains: Vec::new(),
                tolerance,
                terminal: false,
                endpoint_points: 0,
                bracket_points: 0,
                bracket_expansions: 0,
            })
        })?;
        self.sonic.status[side.index()] = c.index_put(
            &self.sonic.status[side.index()],
            0,
            &request.binding.owners,
            &request.status,
        )?;
        let index = self.sonic.requests.len();
        self.sonic.requests.push(request);
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            super::source_sonic_start::locate(
                self.sources,
                &mut self.invocation,
                &mut self.sonic.requests[index],
            )?;
            refine_request(
                self.sources,
                &mut self.invocation,
                &mut self.sonic.requests[index],
                tolerance,
            )
        })();
        if result.is_err() {
            self.sonic.requests[index].terminal = true;
        }
        self.reconcile(checkpoint, result.is_ok())?;
        let request = &self.sonic.requests[index];
        self.sonic.status[side.index()] = c.index_put(
            &self.sonic.status[side.index()],
            0,
            &request.binding.owners,
            &request.status,
        )?;
        result
    }

    #[cfg(test)]
    pub(super) fn refine_source_sonic(
        &mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        super::wet_depth::check_tolerance(
            &tolerance,
            &self.invocation.source_witness().config().root,
        )?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "source sonic refinement remains sealed after an earlier numerical error",
            ));
        }
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            for request in &mut self.sonic.requests {
                if let Err(error) =
                    refine_request(self.sources, &mut self.invocation, request, tolerance)
                {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result
    }

    #[cfg(test)]
    pub(super) fn certify_source_sonic(&mut self) -> HydraulicResult<SourceSonicCertificates<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "source sonic certificates remain sealed after an earlier numerical error",
            ));
        }
        super::source_sonic_certificates::evaluate(self)
    }

    pub(super) fn refine_source_sonic_selected(
        &mut self,
        owners: &Tensor<R>,
        side: SonicSide,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        super::wet_depth::check_tolerance(
            &tolerance,
            &self.invocation.source_witness().config().root,
        )?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(super::point_root_session::invalid(
                "selected source sonic refinement remains sealed after an earlier numerical error",
            ));
        }
        let c = self.sources.geometry.client();
        let demand = checked::<R, _>(c, "source sonic selected complete request admission", |a| {
            let (owners, _) = super::dry_vacuum_evidence::owners(a, self, owners)?;
            let status = c.index_select(&self.sonic.status[side.index()], 0, &owners)?;
            let unrequested = Tensor::full_scalar(
                &[],
                DType::I64,
                SourceSonicStatus::NotRequested as i64 as f64,
                c.device(),
            )?;
            if c.nonzero(&c.eq(&status, &unrequested)?)?.numel() != 0 {
                return Err(super::point_root_session::invalid(
                    "selected sonic refinement includes an unstarted owner and side",
                ));
            }
            let demand = c.index_put(
                &Tensor::zeros(self.invocation_rows.shape(), a.dtype(), c.device())?,
                0,
                &owners,
                &Tensor::ones(owners.shape(), a.dtype(), c.device())?,
            )?;
            for request in &self.sonic.requests {
                if request.binding.side == side
                    && c.nonzero(&c.index_select(&demand, 0, &request.binding.owners)?)?
                        .numel()
                        != 0
                    && (tolerance.atol > request.tolerance.atol
                        || tolerance.rtol > request.tolerance.rtol)
                {
                    return Err(super::point_root_session::invalid(
                        "selected sonic tolerance exceeds the current request allowance",
                    ));
                }
            }
            Ok(demand)
        })?;
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
                if let Err(error) = super::source_sonic_selection::refine(
                    self.sources,
                    &mut self.invocation,
                    request,
                    &demand,
                    &tolerance,
                ) {
                    request.terminal = true;
                    return Err(error);
                }
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result
    }
}

fn refine_request<R>(
    sources: &mut super::paired_sources::PairedSources<'_, R>,
    invocation: &mut super::root_invocation::RootInvocation,
    request: &mut SourceSonicRequest<R>,
    tolerance: CertifiedRootTolerance,
) -> HydraulicResult<()>
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
    let c = sources.geometry.client();
    let mut output = Vec::with_capacity(request.domains.len());
    for mut domain in std::mem::take(&mut request.domains) {
        request
            .binding
            .check(invocation, domain.side, &domain.binding)?;
        let accepted = checked::<R, _>(c, "source sonic physical depth stopping criterion", |a| {
            super::point_root_depth::satisfied(a, &domain.normalized_depth, &tolerance)
        })?;
        if c.nonzero(&accepted)?.numel() == domain.parent_rows.numel() {
            domain.status = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            output.push(domain);
            continue;
        }
        if domain.classification != SourceSonicStatus::PositiveRoot {
            let passed = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::DepthConverged as i64 as f64,
                c.device(),
            )?;
            let floor = Tensor::full_scalar(
                domain.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::AccuracyFloor as i64 as f64,
                c.device(),
            )?;
            domain.status = c.where_cond(&accepted, &passed, &floor)?;
            output.push(domain);
        } else {
            output.push(super::source_sonic_refinement::advance(
                sources,
                invocation,
                &request.binding,
                domain,
                &tolerance,
            )?);
        }
    }
    request.domains = output;
    request.tolerance = tolerance;
    Ok(())
}

#[cfg(test)]
#[path = "source_sonic_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "source_sonic_tests_selection.rs"]
mod selected_tests;
