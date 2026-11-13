use super::super::{boundary_area::BoundaryAreas, hydraulic_geometry::GatheredBands};
use super::{
    admission,
    chart::{AreaAnchor, AreaDirection, AreaQuery, SupportedAreaQuery},
    selection,
    support_selection::SupportSelection,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

/// Coverage decisions in invocation order. These decisions do not establish root existence.
pub(crate) struct DomainCoverage<R: Runtime<DType = DType>> {
    /// Mask in input order: one for proved coverage, zero otherwise.
    pub(crate) proved: Tensor<R>,
    /// Zero-based `I64` invocation indices, not section owners or diagnostic row IDs.
    pub(crate) owner: Tensor<R>,
    /// Certificates in `owner` order. `None` when no row proves coverage.
    pub(crate) pair: Option<CoveredDomain<R>>,
}

/// Two supported charts and retained inequalities proving their complete physical-domain coverage.
pub(crate) struct CoveredDomain<R: Runtime<DType = DType>> {
    pub(crate) lower: SupportedAreaQuery<R>,
    pub(crate) cap: SupportedAreaQuery<R>,
    /// Capacity minus the lower chart's maximum area, proved nonnegative.
    pub(crate) cap_minus_lower: CertifiedExpansion<R>,
    /// Capacity minus the cap chart's maximum deficit, proved nonnegative.
    pub(crate) cap_minus_deficit: CertifiedExpansion<R>,
    /// Cap chart's minimum area minus lower chart's maximum area, proved nonpositive.
    pub(crate) overlap_gap: CertifiedExpansion<R>,
}

pub(super) struct ProofCandidates<R: Runtime<DType = DType>> {
    /// Zero-based indices of candidates with positive spans in the supplied capacity batch.
    pub(super) owner: Tensor<R>,
    pub(super) span: CertifiedExpansion<R>,
    /// Mask of proved inequalities in `owner` order, before chart support checks.
    pub(super) proved: Tensor<R>,
    pub(super) cap_minus_lower: CertifiedExpansion<R>,
    pub(super) cap_minus_deficit: CertifiedExpansion<R>,
    pub(super) overlap_gap: CertifiedExpansion<R>,
}

/// Prove coverage with overlapping lower and cap-deficit charts, without diagnosing physical capacity rejection.
/// Results remain provisional until the caller's existing checked context completes.
pub(crate) fn cover_domain<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    boundaries: &BoundaryAreas<R>,
    original_rows: &Tensor<R>,
) -> HydraulicResult<DomainCoverage<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let diagnostics = admit(a, bands, boundaries, original_rows)?;
    let c = a.client();
    let count = bands.section_owner.numel();
    let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
    let capacity = boundaries.boundary(a, boundaries.cap_indices())?;
    let candidates = proof_candidates(a, &capacity)?;
    if candidates.owner.numel() == 0 {
        return Ok(DomainCoverage {
            proved: zero,
            owner: candidates.owner,
            pair: None,
        });
    }

    let selected_bands = bands.select(a, &candidates.owner)?;
    let diagnostics = c.index_select(&diagnostics, 0, &candidates.owner)?;
    let span = a.expansion_interval(&candidates.span)?;
    let parameter = a.bounds(
        &Tensor::zeros(span.lower().shape(), a.dtype(), c.device())?,
        span.upper(),
    )?;
    let scale = a.expansion_point(&Tensor::ones(span.lower().shape(), a.dtype(), c.device())?)?;
    let lower = AreaQuery::band_affine(
        a,
        &selected_bands,
        AreaAnchor::Zero,
        AreaDirection::Add,
        &scale,
        &parameter,
    )?;
    let cap = AreaQuery::cap_deficit(a, &selected_bands, &scale, &parameter)?;
    c.copy_into(&lower.original_rows, &diagnostics)?;
    c.copy_into(&cap.original_rows, &diagnostics)?;
    let lower = lower.select_supported(a, &selected_bands)?;
    let cap = cap.select_supported(a, &selected_bands)?;
    let common = c.mul(&candidates.proved, &c.mul(&lower.proved, &cap.proved)?)?;
    let common_owner = c.nonzero(&common)?.flatten()?;
    let owner = c.index_select(&candidates.owner, 0, &common_owner)?;
    if common_owner.numel() == 0 {
        return Ok(DomainCoverage {
            proved: zero,
            owner,
            pair: None,
        });
    }

    let lower = align(a, lower, &common, common_owner.numel(), "lower")?;
    let cap = align(a, cap, &common, common_owner.numel(), "cap")?;
    let cap_minus_lower = selection::expansion(a, &candidates.cap_minus_lower, &common_owner)?;
    let cap_minus_deficit = selection::expansion(a, &candidates.cap_minus_deficit, &common_owner)?;
    let overlap_gap = selection::expansion(a, &candidates.overlap_gap, &common_owner)?;
    let proved = c.index_put(
        &zero,
        0,
        &owner,
        &Tensor::ones(owner.shape(), a.dtype(), c.device())?,
    )?;
    Ok(DomainCoverage {
        proved,
        owner,
        pair: Some(CoveredDomain {
            lower,
            cap,
            cap_minus_lower,
            cap_minus_deficit,
            overlap_gap,
        }),
    })
}

fn admit<R>(
    a: &Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    boundaries: &BoundaryAreas<R>,
    original_rows: &Tensor<R>,
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
    admission::matching_model(boundaries.same_context(&bands.model))?;
    let count = bands.section_owner.numel();
    admission::metadata(a, boundaries.owners(), &[count], DType::I64)?;
    admission::matching_owners(a, bands, boundaries.owners())?;
    admission::metadata(a, original_rows, &[count], DType::I64)?;
    let c = a.client();
    let captured = Tensor::empty(original_rows.shape(), DType::I64, c.device())?;
    c.copy_into(&captured, original_rows)?;
    let zero = Tensor::zeros(&[], DType::I64, c.device())?;
    admission::reject(
        a,
        &c.cast(&c.lt(&captured, &zero)?, a.dtype())?,
        "domain coverage diagnostic rows are negative. Supply nonnegative invocation identities",
    )?;
    Ok(captured)
}

/// Assess retained cap certificates independently of model binding and query construction.
pub(super) fn proof_candidates<R>(
    a: &mut Arithmetic<'_, R>,
    capacity: &CertifiedExpansion<R>,
) -> HydraulicResult<ProofCandidates<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let count = capacity.high().numel();
    for component in [capacity.high(), capacity.low(), capacity.abs_error()] {
        admission::metadata(a, component, &[count], a.dtype())?;
    }
    let c = a.client();
    let interval = a.expansion_interval(capacity)?;
    // This represented point is used for both chart widths. Its rounded value receives no floor.
    let span = a.expansion_point(&c.mul_scalar(interval.lower(), 0.75)?)?;
    let zero = Tensor::zeros(&[count], a.dtype(), c.device())?;
    let owner = c.nonzero(&c.gt(span.high(), &zero)?)?.flatten()?;
    let capacity = selection::expansion(a, capacity, &owner)?;
    let span = selection::expansion(a, &span, &owner)?;
    let cap_minus_lower = a.expansion_sub(&capacity, &span)?;
    let cap_minus_deficit = a.expansion_sub(&capacity, &span)?;
    let overlap_gap = a.expansion_sub(&cap_minus_lower, &span)?;
    let lower = a.expansion_interval(&cap_minus_lower)?;
    let deficit = a.expansion_interval(&cap_minus_deficit)?;
    let overlap = a.expansion_interval(&overlap_gap)?;
    let zero = Tensor::zeros(owner.shape(), a.dtype(), c.device())?;
    let proved = c.mul(
        &c.mul(&c.ge(lower.lower(), &zero)?, &c.ge(deficit.lower(), &zero)?)?,
        &c.le(overlap.upper(), &zero)?,
    )?;
    Ok(ProofCandidates {
        owner,
        span,
        proved,
        cap_minus_lower,
        cap_minus_deficit,
        overlap_gap,
    })
}

fn align<R>(
    a: &mut Arithmetic<'_, R>,
    support: SupportSelection<R>,
    common: &Tensor<R>,
    count: usize,
    name: &str,
) -> HydraulicResult<SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let positions = c
        .nonzero(&c.index_select(common, 0, &support.owner)?)?
        .flatten()?;
    if positions.numel() != count {
        return Err(HydraulicError::Numerical {
            context: format!(
                "Domain coverage {name} selection has {} rows, expected {count}. Discard the inconsistent cover",
                positions.numel(),
            ),
        });
    }
    let query = support.query.ok_or_else(|| HydraulicError::Numerical {
        context: format!(
            "Domain coverage has common rows but no supported {name} query. Discard the inconsistent cover",
        ),
    })?;
    query.select(a, &positions)
}
