use super::{
    paired_sources::PairedSources,
    wet_depth_mapping, wet_depth_refinement, wet_depth_selection,
    wet_depth_types::{DepthRefinementState, WetDepthDomain, WetDepthSession, WetDepthStatus},
};
#[cfg(test)]
use super::{wet_depth_types::WetDepthState, wet_root_types::WetRootDomains};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance},
};

/// Check physical depth while preserving root evidence for later, stricter stopping phases.
#[cfg(test)]
pub(super) fn refine_wet_depth<'s, 'a, R>(
    sources: &'s mut PairedSources<'a, R>,
    located: WetRootDomains<R>,
) -> HydraulicResult<WetDepthSession<'s, 'a, R>>
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
    sources.check_root_witness(located.invocation.source_witness())?;
    located.invocation.remaining()?;
    for domain in &located.domains {
        located.invocation.check(&domain.invocation_witness)?;
    }
    let tolerance = CertifiedRootTolerance {
        atol: located.invocation.source_witness().config().root.atol,
        rtol: located.invocation.source_witness().config().root.rtol,
    };
    let domains = capture_domains(sources, &located.invocation, located.domains)?;
    let c = sources.geometry.client();
    let sonic =
        super::source_sonic_types::SourceSonicState::empty(c, located.invocation_rows.numel())?;
    WetDepthSession {
        sources,
        invocation_witness: located.invocation.witness(),
        invocation: located.invocation,
        invocation_rows: located.invocation_rows,
        wet: WetDepthState {
            locator_status: located.status,
            evidence: located.evidence,
            domains,
            tolerance,
            terminal: false,
        },
        sonic,
        terminal: false,
    }
    .refine_depth_to(tolerance)
}

pub(super) fn capture_domains<R>(
    sources: &PairedSources<'_, R>,
    invocation: &super::root_invocation::RootInvocation,
    originals: Vec<super::wet_root_types::WetRootDomain<R>>,
) -> HydraulicResult<Vec<WetDepthDomain<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    for domain in &originals {
        invocation.check(&domain.invocation_witness)?;
    }
    let c = sources.geometry.client();
    let mut domains = Vec::with_capacity(originals.len());
    for original in originals {
        let (depth, normalized_depth) =
            wet_depth_mapping::certificates(sources, &original.parent_rows, &original.query)?;
        let (query, invocation_owner, parent_rows) =
            checked::<R, _>(c, "wet depth original bracket capture", |a| {
                let selected = sources.select_sources(a, &original.parent_rows)?;
                let query = original.query.with_parameter_bounds(
                    a,
                    &selected.bands,
                    &original.query.query.parameter,
                )?;
                let owners =
                    Tensor::empty(original.invocation_owner.shape(), DType::I64, c.device())?;
                c.copy_into(&owners, &original.invocation_owner)?;
                Ok((query, owners, selected.rows))
            })?;
        domains.push(WetDepthDomain {
            invocation_witness: original.invocation_witness.clone(),
            invocation_owner,
            parent_rows,
            query,
            depth,
            normalized_depth,
            status: Tensor::full_scalar(
                original.parent_rows.shape(),
                DType::I64,
                WetDepthStatus::Unresolved as i64 as f64,
                c.device(),
            )?,
            iterations: 0,
            callback_points: 0,
            mapped_points: 0,
            refinement: DepthRefinementState::Deferred {
                original: Box::new(original),
            },
        });
    }
    Ok(domains)
}

impl<'s, 'a, R> WetDepthSession<'s, 'a, R>
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
    /// Request no looser precision than the original configured normalized-depth allowances.
    #[cfg(test)]
    pub(super) fn refine_depth_to(
        mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<Self> {
        self.refine_wet_depth_in_place(tolerance)?;
        Ok(self)
    }

    pub(super) fn refine_wet_depth_in_place(
        &mut self,
        tolerance: CertifiedRootTolerance,
    ) -> HydraulicResult<()> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        check_tolerance(&tolerance, &self.invocation.source_witness().config().root)?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet refinement remains sealed after an earlier numerical error",
            ));
        }
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let c = self.sources.geometry.client();
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for mut domain in std::mem::take(&mut self.wet.domains) {
                let accepted = checked::<R, _>(c, "wet depth stopping criterion", |a| {
                    wet_depth_mapping::satisfied(a, &domain.normalized_depth, &tolerance)
                })?;
                let passing = c.nonzero(&accepted)?.flatten()?;
                if passing.numel() == domain.parent_rows.numel() {
                    domain.status = Tensor::full_scalar(
                        domain.parent_rows.shape(),
                        DType::I64,
                        WetDepthStatus::DepthConverged as i64 as f64,
                        c.device(),
                    )?;
                    output.push(domain);
                    continue;
                }
                if domain.is_deferred() && passing.numel() != 0 {
                    let zero = Tensor::zeros(accepted.shape(), accepted.dtype(), c.device())?;
                    let failing = c.nonzero(&c.eq(&accepted, &zero)?)?.flatten()?;
                    let (pass, fail) = checked::<R, _>(c, "wet depth deferred partition", |a| {
                        Ok((
                            wet_depth_selection::deferred(
                                a,
                                &domain,
                                &passing,
                                WetDepthStatus::DepthConverged,
                            )?,
                            wet_depth_selection::deferred(
                                a,
                                &domain,
                                &failing,
                                WetDepthStatus::Unresolved,
                            )?,
                        ))
                    })?;
                    output.push(pass);
                    domain = fail;
                }
                output.push(wet_depth_refinement::advance(
                    self.sources,
                    &mut self.invocation,
                    domain,
                    &tolerance,
                )?);
            }
            self.wet.domains = output;
            self.wet.tolerance = tolerance;
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        if result.is_err() {
            self.wet.terminal = true;
        }
        result
    }
}

pub(super) fn check_tolerance(
    tolerance: &CertifiedRootTolerance,
    original: &solvr::optimize::CertifiedBisectionOptions,
) -> HydraulicResult<()> {
    for (name, value, limit) in [
        ("atol", tolerance.atol, original.atol),
        ("rtol", tolerance.rtol, original.rtol),
    ] {
        if !value.is_finite() || value < 0.0 || value > limit {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Wet depth {name}={value} exceeds original finite nonnegative allowance={limit}. Request an allowance within the original ceiling"
                ),
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "wet_depth_tests.rs"]
pub(in crate::channel::flow::surveyed_points) mod tests;
