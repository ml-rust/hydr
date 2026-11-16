use super::{
    paired_sources::PairedSources,
    point_flux_masks::{equals, not},
    point_quadrature_capacity::{self, NativeRootUse},
    point_quadrature_progress::{self as progress_rows, Record},
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    point_root_session::{PointRootSession, invalid},
    root_invocation::RootInvocation,
    wet_depth_refresh::SelectedContinuation,
    wet_depth_types::{DepthRefinementState, WetDepthDomain, WetDepthStatus},
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedRootTolerance},
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
        + CertifiedMappedBisectionAlgorithms<R>,
{
    /// Narrows the star brackets of the requested owners until each area parameter width is at
    /// most `JUMP_FRACTION` of the face's jump scale, or the bracket meets its arithmetic floor.
    /// Each row keeps its panel level and depth status, and a deferred domain initializes first
    /// at the session tolerance. Only rows with a located root, distinct sources and a nonsingleton
    /// bracket resume. Every other row records `NotApplicable`.
    pub(super) fn sharpen_wet_star(
        &mut self,
        owners: &Tensor<R>,
    ) -> HydraulicResult<PointQuadratureProgress<R>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        self.check_domains()?;
        if self.wet.terminal {
            return Err(invalid(
                "wet star sharpening remains sealed after an earlier numerical error",
            ));
        }
        let mut progress = progress_rows::start(
            self,
            owners,
            None,
            "wet star sharpening complete request admission",
        )?;
        if progress.invocation_owner.numel() == 0 {
            return Ok(progress);
        }
        let (demand, association) =
            progress_rows::demand(self, &progress, "wet star sharpening nonempty demand")?;
        let before = self.sources.evaluation.work.snapshot();
        let old_mapped = super::wet_star_quadrature::mapped(&self.wet.domains)?;
        let tolerance = self.wet.tolerance;
        let checkpoint = self.checkpoint();
        let result = (|| -> HydraulicResult<()> {
            let mut output = Vec::with_capacity(self.wet.domains.len());
            for domain in std::mem::take(&mut self.wet.domains) {
                output.extend(sharpen(
                    self.sources,
                    &mut self.invocation,
                    domain,
                    Request {
                        demand: &demand,
                        association: &association,
                        tolerance: &tolerance,
                    },
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

/// Invocation-row demand, progress positions, and the session depth tolerance.
struct Request<'r, R: Runtime<DType = DType>> {
    demand: &'r Tensor<R>,
    association: &'r Tensor<R>,
    tolerance: &'r CertifiedRootTolerance,
}

/// Sharpens the open rows of one domain. A partly demanded deferred domain splits, and its
/// other rows stay deferred.
fn sharpen<R>(
    sources: &mut PairedSources<'_, R>,
    invocation: &mut RootInvocation,
    domain: WetDepthDomain<R>,
    request: Request<'_, R>,
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
    let (scale, uniform) =
        super::wet_star_jump::scale(sources, &domain.parent_rows, &domain.query)?;
    let all = c.arange(0., domain.parent_rows.numel() as f64, 1., DType::I64)?;
    let todo = open(c, &domain, request.demand, &uniform, &all)?;
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
        mark(c, &domain, &todo, Outcome::WorkLimit, &request, progress)?;
        return Ok(vec![domain]);
    }
    let mut output = Vec::with_capacity(2);
    let (domain, rows, scale, uniform) = if domain.is_deferred()
        && todo.numel() != domain.parent_rows.numel()
    {
        let (selected, complement) = super::wet_star_quadrature_request::split(c, &domain, &todo)?;
        output.push(complement);
        let rows = c.arange(0., selected.parent_rows.numel() as f64, 1., DType::I64)?;
        let scale = c.index_select(&scale, 0, &todo)?;
        let uniform = c.index_select(&uniform, 0, &todo)?;
        (selected, rows, scale, uniform)
    } else {
        (domain, todo, scale, uniform)
    };
    let domain = if domain.is_deferred() {
        super::wet_depth_refinement::advance(sources, invocation, domain, request.tolerance)?
    } else {
        domain
    };
    if domain.is_deferred() {
        mark(c, &domain, &rows, Outcome::WorkLimit, &request, progress)?;
        output.push(domain);
        return Ok(output);
    }
    // Initialization can leave a row without a located root.
    let rows = open(c, &domain, request.demand, &uniform, &rows)?;
    if rows.numel() == 0 {
        output.push(domain);
        return Ok(output);
    }
    let levels = domain.levels.selected(c, &rows)?.levels().clone();
    let old_lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
    let old_upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
    let old_status = c.index_select(&domain.status, 0, &rows)?;
    let mut domain = super::wet_depth_refresh::refresh_selected(
        sources,
        invocation,
        domain,
        request.tolerance,
        &rows,
        SelectedContinuation::Sharpen(&scale),
    )?;
    checked::<R, _>(c, "wet star sharpening committed progress", |a| {
        let new_status = c.index_select(&domain.status, 0, &rows)?;
        let limited = c.eq(
            &new_status,
            &Tensor::full_scalar(
                &[],
                DType::I64,
                WetDepthStatus::WorkLimit as i64 as f64,
                c.device(),
            )?,
        )?;
        let outcome = c.where_cond(
            &limited,
            &progress_rows::code::<R>(c, &rows, Outcome::WorkLimit)?,
            &progress_rows::code::<R>(c, &rows, Outcome::Advanced)?,
        )?;
        // The narrower bracket still meets the depth criterion its status records.
        domain.status = c.index_put(&domain.status, 0, &rows, &old_status)?;
        let lower = c.index_select(domain.query.query.parameter.lower(), 0, &rows)?;
        let upper = c.index_select(domain.query.query.parameter.upper(), 0, &rows)?;
        let improved = progress_rows::improved(c, [&old_lower, &old_upper], [&lower, &upper])?;
        let owners = c.index_select(&domain.invocation_owner, 0, &rows)?;
        progress_rows::record(
            a,
            request.association,
            progress,
            Record {
                owners: &owners,
                old: &levels,
                new: &levels,
                outcome: &outcome,
                improved: &improved,
            },
        )
    })?;
    output.push(domain);
    Ok(output)
}

/// The given domain rows that the demand reaches, whose root is located, whose sources
/// differ, and whose parameter bracket is not a singleton.
fn open<R>(
    c: &R::Client,
    domain: &WetDepthDomain<R>,
    demand: &Tensor<R>,
    uniform: &Tensor<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(c, "wet star sharpening open rows", |a| {
        let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
        let status = c.index_select(&domain.status, 0, rows)?;
        let located = c.maximum(
            &equals(a, &status, WetDepthStatus::DepthConverged as i64)?,
            &equals(a, &status, WetDepthStatus::AccuracyFloor as i64)?,
        )?;
        let distinct = not(a, &c.index_select(uniform, 0, rows)?)?;
        let lower = c.index_select(domain.query.query.parameter.lower(), 0, rows)?;
        let upper = c.index_select(domain.query.query.parameter.upper(), 0, rows)?;
        let wide = c.lt(&lower, &upper)?;
        let open = c.mul(
            &c.mul(&c.index_select(demand, 0, &owners)?, &located)?,
            &c.mul(&distinct, &wide)?,
        )?;
        Ok(c.index_select(rows, 0, &c.nonzero(&open)?.flatten()?)?)
    })
}

/// Records unchanged rows at their current panel level.
fn mark<R>(
    c: &R::Client,
    domain: &WetDepthDomain<R>,
    rows: &Tensor<R>,
    outcome: Outcome,
    request: &Request<'_, R>,
    progress: &mut PointQuadratureProgress<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let owners = c.index_select(&domain.invocation_owner, 0, rows)?;
    let levels = domain.levels.selected(c, rows)?.levels().clone();
    progress_rows::mark(c, &owners, &levels, outcome, request.association, progress)
}
