use super::{
    paired_sources::PairedSources,
    point_quadrature_capacity::{self, NativeRootUse},
    point_quadrature_progress::{self as progress_rows, Resumed},
    point_quadrature_types::{
        PointQuadratureOutcome as Outcome, PointQuadratureProgress, TIGHTEST_DEPTH as TIGHTEST,
    },
    point_root_session::{PointRootSession, invalid},
    root_invocation::RootInvocation,
    source_sonic_types::{SonicSide, SourceSonicBinding, SourceSonicDomain, SourceSonicStatus},
    wet_depth::check_tolerance,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps, UtilityOps,
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
    /// Tightens the sonic depth tolerance of every request that holds a requested owner on
    /// `side` to zero, and resumes its requested converged rows at their current panel level.
    /// Later sonic refreshes of those requests use the same tolerance.
    /// Rows without a converged positive sonic root record `NotApplicable`.
    pub(super) fn tighten_source_sonic(
        &mut self,
        owners: &Tensor<R>,
        side: SonicSide,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.sonic.requests.iter().any(|request| request.terminal) {
            return Err(invalid(
                "sonic depth tightening remains sealed after an earlier numerical error",
            ));
        }
        check_tolerance(&TIGHTEST, &self.invocation.source_witness().config().root)?;
        let c = self.sources.geometry.client();
        let mut progress = progress_rows::start(
            self,
            owners,
            Some(side),
            "sonic depth tightening complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) =
            progress_rows::demand(self, &progress, "sonic depth tightening nonempty demand")?;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = super::source_sonic_quadrature::mapped::<R>(&self.sonic.requests)?;
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
                request.tolerance = TIGHTEST;
                let mut output = Vec::with_capacity(request.domains.len());
                for domain in std::mem::take(&mut request.domains) {
                    match tighten(
                        self.sources,
                        &mut self.invocation,
                        &request.binding,
                        domain,
                        &demand,
                        &association,
                        &mut progress,
                    ) {
                        Ok(domains) => output.extend(domains),
                        Err(error) => {
                            request.terminal = true;
                            return Err(error);
                        }
                    }
                }
                request.domains = output;
            }
            Ok(())
        })();
        self.reconcile(checkpoint, result.is_ok())?;
        result?;
        progress_rows::finish(
            &mut progress,
            before,
            self.sources.evaluation.work.snapshot(),
            [
                old_mapped,
                super::source_sonic_quadrature::mapped::<R>(&self.sonic.requests)?,
            ],
            self.callback_points(),
        )?;
        Ok(progress)
    }
}

/// Resumes the demanded rows of one positive sonic domain that converged wider than the
/// tightest tolerance. A partly demanded deferred domain splits, and its other rows stay deferred.
fn tighten<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    binding: &SourceSonicBinding<R>,
    domain: SourceSonicDomain<R>,
    demand: &Tensor<R>,
    association: &Tensor<R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<Vec<SourceSonicDomain<R>>>
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
    if domain.classification != SourceSonicStatus::PositiveRoot {
        return Ok(vec![domain]);
    }
    let c = sources.geometry.client();
    let todo = progress_rows::open_rows(
        c,
        "sonic depth tightening rows",
        demand,
        &domain.invocation_owner,
        &domain.status,
        &domain.normalized_depth,
    )?;
    if todo.numel() == 0 {
        return Ok(vec![domain]);
    }
    binding.check(invocation, domain.side, &domain.binding)?;
    let native = domain.refinement.as_ref().map(|state| NativeRootUse {
        limits: &state.limits,
        iterations: domain.iterations,
        callback_points: domain.callback_points,
    });
    if !point_quadrature_capacity::permits(sources, invocation, native, &domain.query, &todo)? {
        let owners = c.index_select(&domain.invocation_owner, 0, &todo)?;
        let levels = c.index_select(domain.numerical.levels(), 0, &todo)?;
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
    let (domain, rows) =
        if domain.refinement.is_none() && todo.numel() != domain.parent_rows.numel() {
            let mask = checked::<R, _>(c, "sonic depth tightening deferred partition", |a| {
                let zero = Tensor::zeros(domain.parent_rows.shape(), a.dtype(), c.device())?;
                let mask = c.index_put(
                    &zero,
                    0,
                    &todo,
                    &Tensor::ones(todo.shape(), a.dtype(), c.device())?,
                )?;
                Ok(c.nonzero(&c.eq(&mask, &zero)?)?.flatten()?)
            })?;
            output.push(super::source_sonic_selection::deferred(
                sources, &domain, &mask,
            )?);
            let selected = super::source_sonic_selection::deferred(sources, &domain, &todo)?;
            let rows = c.arange(0., selected.parent_rows.numel() as f64, 1., DType::I64)?;
            (selected, rows)
        } else {
            (domain, todo)
        };
    let levels = c.index_select(domain.numerical.levels(), 0, &rows)?;
    let old_lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
    let old_upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
    let domain = if domain.refinement.is_some() {
        super::source_sonic_refinement::advance_selected(
            sources, invocation, binding, domain, &TIGHTEST, &rows,
        )?
    } else {
        super::source_sonic_refinement::advance(sources, invocation, binding, domain, &TIGHTEST)?
    };
    progress_rows::record_resumed(
        c,
        "sonic depth tightening committed progress",
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
