use super::{
    paired_sources::PairedSources,
    point_quadrature_capacity::{self, NativeRootUse},
    point_quadrature_progress::{self as progress_rows, Resumed},
    point_quadrature_types::{
        PointQuadratureOutcome as Outcome, PointQuadratureProgress, TIGHTEST_DEPTH as TIGHTEST,
    },
    point_root_session::{PointRootSession, invalid},
    root_invocation::RootInvocation,
    wet_depth::check_tolerance,
    wet_depth_refresh::SelectedContinuation,
    wet_depth_types::{DepthRefinementState, WetDepthDomain},
};
use crate::channel::flow::surveyed::error::HydraulicResult;
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps, UtilityOps,
    },
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
    /// Tightens the wet depth tolerance to zero and resumes the requested converged rows.
    /// Each row keeps its panel level. Later star refreshes use the same tolerance.
    /// Rows without a converged wet depth record `NotApplicable`.
    pub(super) fn tighten_wet_depth(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal {
            return Err(invalid(
                "wet depth tightening remains sealed after an earlier numerical error",
            ));
        }
        check_tolerance(&TIGHTEST, &self.invocation.source_witness().config().root)?;
        let mut progress = progress_rows::start(
            self,
            owners,
            None,
            "wet depth tightening complete request admission",
        )?;
        self.wet.tolerance = TIGHTEST;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) =
            progress_rows::demand(self, &progress, "wet depth tightening nonempty demand")?;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = super::wet_star_quadrature::mapped(&self.wet.domains)?;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for domain in std::mem::take(&mut self.wet.domains) {
                output.extend(tighten(
                    self.sources,
                    &mut self.invocation,
                    domain,
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
        progress_rows::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [
                old_mapped,
                super::wet_star_quadrature::mapped(&self.wet.domains)?,
            ],
            self.callback_points(),
        )?;
        Ok(progress)
    }
}

/// Resumes the demanded rows of one domain that converged wider than the tightest tolerance.
/// A partly demanded deferred domain splits, and its other rows stay deferred.
fn tighten<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    domain: WetDepthDomain<R>,
    demand: &Tensor<R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<Vec<WetDepthDomain<R>>>
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
    let todo = progress_rows::open_rows(
        c,
        "wet depth tightening rows",
        demand,
        &domain.invocation_owner,
        &domain.status,
        &domain.normalized_depth,
    )?;
    if todo.numel() == 0 {
        return Ok(vec![domain]);
    }
    let native = match &domain.refinement {
        DepthRefinementState::Deferred { .. } => None,
        DepthRefinementState::Initialized { state } => Some(NativeRootUse {
            limits: &state.limits,
            iterations: domain.iterations,
            callback_points: domain.callback_points,
        }),
    };
    if !point_quadrature_capacity::permits(sources, invocation, native, &domain.query, &todo)? {
        let owners = c.index_select(&domain.invocation_owner, 0, &todo)?;
        let levels = domain.levels.selected(c, &todo)?.levels().clone();
        progress_rows::mark(
            c,
            &owners,
            &levels,
            Outcome::WorkLimit,
            association,
            progress,
        )?;
        return Ok(vec![domain]);
    }
    let mut output = Vec::with_capacity(2);
    let (domain, rows) = if domain.is_deferred() && todo.numel() != domain.parent_rows.numel() {
        let (selected, complement) = super::wet_star_quadrature_request::split(c, &domain, &todo)?;
        output.push(complement);
        let rows = c.arange(0., selected.parent_rows.numel() as f64, 1., DType::I64)?;
        (selected, rows)
    } else {
        (domain, todo)
    };
    let levels = domain.levels.selected(c, &rows)?.levels().clone();
    let old_lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
    let old_upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
    let domain = if domain.is_deferred() {
        super::wet_depth_refinement::advance(sources, invocation, domain, &TIGHTEST)?
    } else {
        super::wet_depth_refresh::refresh_selected(
            sources,
            invocation,
            domain,
            &TIGHTEST,
            &rows,
            SelectedContinuation::Resume,
        )?
    };
    progress_rows::record_resumed(
        c,
        "wet depth tightening committed progress",
        association,
        progress,
        Resumed {
            rows: &rows,
            levels: &levels,
            old_bounds: [&old_lower, &old_upper],
            owners: &domain.invocation_owner,
            bounds: [
                domain.query.query.parameter.lower(),
                domain.query.query.parameter.upper(),
            ],
            status: &domain.status,
        },
    )?;
    output.push(domain);
    Ok(output)
}
