use super::{
    paired_sources::PairedSources,
    root_invocation::{RootInvocation, RootInvocationWitness},
    source_sonic_types::SourceSonicState,
    wet_depth_types::WetDepthState,
    wet_root_types::WetRootStatus,
};
#[cfg(test)]
use super::{wet_depth_types::WetDepthDomain, wet_root_types::WetRootEvidence};
use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::optimize::CertifiedRootTolerance;

/// One source graph borrow and invocation allowance serve independent wet and source-sonic state.
pub(super) struct PointRootSession<'s, 'a, R: Runtime<DType = DType>> {
    pub(super) sources: &'s mut PairedSources<'a, R>,
    pub(super) invocation: RootInvocation,
    pub(super) invocation_witness: RootInvocationWitness,
    pub(super) invocation_rows: Tensor<R>,
    pub(super) wet: WetDepthState<R>,
    pub(super) sonic: SourceSonicState<R>,
    pub(super) terminal: bool,
}

pub(super) struct WorkCheckpoint {
    used: usize,
    residual: usize,
}

impl<'s, 'a, R: Runtime<DType = DType>> PointRootSession<'s, 'a, R> {
    pub(super) fn sources(&self) -> &PairedSources<'a, R> {
        self.sources
    }
    pub(super) fn invocation_rows(&self) -> &Tensor<R> {
        &self.invocation_rows
    }
    #[cfg(test)]
    pub(super) fn locator_status(&self) -> &Tensor<R> {
        &self.wet.locator_status
    }
    #[cfg(test)]
    pub(super) fn evidence(&self) -> &[WetRootEvidence<R>] {
        &self.wet.evidence
    }
    #[cfg(test)]
    pub(super) fn domains(&self) -> &[WetDepthDomain<R>] {
        &self.wet.domains
    }
    pub(super) fn point_limit(&self) -> usize {
        self.invocation.point_limit()
    }
    pub(super) fn callback_points(&self) -> usize {
        self.invocation.callback_points()
    }
    #[cfg(test)]
    pub(super) fn tolerance(&self) -> &CertifiedRootTolerance {
        &self.wet.tolerance
    }
    pub(super) fn checkpoint(&self) -> WorkCheckpoint {
        WorkCheckpoint {
            used: self.callback_points(),
            residual: self.sources.evaluation.work.snapshot().residual_points,
        }
    }
    pub(super) fn reconcile(
        &mut self,
        before: WorkCheckpoint,
        successful: bool,
    ) -> HydraulicResult<()> {
        let result = (|| -> HydraulicResult<()> {
            let actual = self
                .sources
                .evaluation
                .work
                .snapshot()
                .residual_points
                .checked_sub(before.residual)
                .ok_or_else(|| invalid("native callback count decreased"))?;
            let recorded = self
                .callback_points()
                .checked_sub(before.used)
                .ok_or_else(|| invalid("invocation callback count decreased"))?;
            let missing = actual
                .checked_sub(recorded)
                .ok_or_else(|| invalid("recorded callback work exceeds executed work"))?;
            self.invocation.add_points(missing)?;
            if successful && missing != 0 {
                return Err(invalid(
                    "successful callback diagnostics omit executed work",
                ));
            }
            Ok(())
        })();
        if result.is_err() {
            self.terminal = true;
        }
        result
    }
}

impl<'s, 'a, R> PointRootSession<'s, 'a, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(super) fn new(
        sources: &'s mut PairedSources<'a, R>,
        rows: &Tensor<R>,
        max_points: usize,
    ) -> HydraulicResult<Self> {
        let source = sources.root_witness()?;
        let invocation = RootInvocation::new(source, max_points)?;
        let c = sources.geometry.client();
        let invocation_rows = checked::<R, _>(c, "point root captured invocation rows", |a| {
            Ok(sources.select_sources(a, rows)?.rows)
        })?;
        let count = invocation_rows.numel();
        let config = invocation.source_witness().config();
        let tolerance = CertifiedRootTolerance {
            atol: config.root.atol,
            rtol: config.root.rtol,
        };
        let wet = WetDepthState {
            locator_status: Tensor::full_scalar(
                &[count],
                DType::I64,
                WetRootStatus::NotRequested as i64 as f64,
                c.device(),
            )?,
            #[cfg(test)]
            evidence: Vec::new(),
            domains: Vec::new(),
            tolerance,
            terminal: false,
        };
        let sonic = SourceSonicState::empty(c, count)?;
        let invocation_witness = invocation.witness();
        Ok(Self {
            sources,
            invocation,
            invocation_witness,
            invocation_rows,
            wet,
            sonic,
            terminal: false,
        })
    }

    pub(super) fn check_domains(&self) -> HydraulicResult<()> {
        if self.terminal {
            return Err(invalid(
                "session remains sealed after inconsistent work accounting",
            ));
        }
        self.invocation.check(&self.invocation_witness)?;
        for domain in &self.wet.domains {
            domain.check_invocation(&self.invocation)?;
        }
        super::source_sonic_binding::check(self)
    }
}

pub(super) fn invalid(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Point root {message}. Preserve the captured invocation and bound state"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed_points::{
        chart_test_support,
        source_tests::support::{Result, hydraulic},
        wet_depth::tests::support,
    };

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: support::DepthClient<R>,
    {
        let config = chart_test_support::config();
        support::with_rectangle::<R, _>(
            client,
            1.0,
            4.0,
            &[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            &config,
            |pair, work| {
                let rows = support::rows::<R>(client, &[1, 0, 1])?;
                let before = work.snapshot();
                let session =
                    hydraulic(PointRootSession::new(pair, &rows, config.root.max_points))?;
                assert_eq!(session.invocation_rows().try_to_vec::<i64>()?, [1, 0, 1]);
                assert_eq!(
                    session.locator_status().try_to_vec::<i64>()?,
                    [WetRootStatus::NotRequested as i64; 3]
                );
                assert!(session.domains().is_empty());
                assert_eq!(session.callback_points(), 0);
                assert_eq!(session.point_limit(), config.root.max_points);
                assert_eq!(work.snapshot(), before);
                Ok(())
            },
        )
    }

    #[test]
    fn configured_cpu_point_root_session_captures_nonwet_invocation_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
        matrix::<CpuRuntime>(&CpuClient::new(CpuDevice::new()))
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_point_root_session_captures_nonwet_invocation_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
