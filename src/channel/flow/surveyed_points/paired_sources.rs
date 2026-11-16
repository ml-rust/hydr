use super::paired_source_cache::{self, SourcePotentialCache};
use super::potential_range::{PotentialEvaluation, admit_evaluation, source_query};
use super::source_binding::{RootSourceIdentity, RootSourceWitness};
use super::types::SourceVelocity;
use super::wave_curve_types::CurveValue;
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked},
    section::{
        BoundaryAreas, CrossSectionError, GatheredBands, HydraulicGeometry, PhysicalEndpoint,
        PrismaticAreaPointRecovery, SupportedAreaQuery, source_difference,
    },
};
#[cfg(test)]
use numr::ops::UtilityOps;
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::sync::Arc;

/// One immutable source graph and two fixed source-potential caches serve every target callback.
pub(super) struct PairedSources<'a, R: Runtime<DType = DType>> {
    pub(super) geometry: &'a HydraulicGeometry<R>,
    pub(super) bands: &'a GatheredBands<R>,
    pub(super) sources: [&'a PrismaticAreaPointRecovery<R>; 2],
    pub(super) velocity: &'a SourceVelocity<R>,
    pub(super) boundaries: &'a BoundaryAreas<R>,
    pub(super) evaluation: PotentialEvaluation<'a>,
    pub(super) original_rows: Tensor<R>,
    pub(super) cache: [SourcePotentialCache<R>; 2],
    pub(super) panel_cache: Vec<(usize, [SourcePotentialCache<R>; 2])>,
    identity: Arc<RootSourceIdentity>,
    construction_witness: RootSourceWitness,
}

#[cfg(test)]
pub(super) struct PairedRange<R: Runtime<DType = DType>> {
    pub(super) lower: CurveValue<R>,
    pub(super) upper: CurveValue<R>,
    pub(super) curve: CurveValue<R>,
}

pub(super) struct SourceInvariant<R: Runtime<DType = DType>> {
    pub(super) value: CurveValue<R>,
    pub(super) vacuum: Tensor<R>,
    pub(super) touching: Tensor<R>,
    pub(super) wet_candidate: Tensor<R>,
    #[cfg(test)]
    pub(super) unresolved: Tensor<R>,
    /// Exact captured dry-state identity, not an available wet velocity residual.
    pub(super) both_dry: Tensor<R>,
}

/// Scoped owned selections have no cache and borrow none of their own storage.
pub(super) struct SelectedRows<R: Runtime<DType = DType>> {
    pub(super) rows: Tensor<R>,
    pub(super) bands: GatheredBands<R>,
    pub(super) sources: [PrismaticAreaPointRecovery<R>; 2],
    pub(super) velocity: SourceVelocity<R>,
    pub(super) boundaries: BoundaryAreas<R>,
    pub(super) original_rows: Tensor<R>,
}

impl<'a, R> PairedSources<'a, R>
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
        geometry: &'a HydraulicGeometry<R>,
        bands: &'a GatheredBands<R>,
        sources: [&'a PrismaticAreaPointRecovery<R>; 2],
        velocity: &'a SourceVelocity<R>,
        boundaries: &'a BoundaryAreas<R>,
        evaluation: PotentialEvaluation<'a>,
        original_rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        admit_evaluation(geometry, bands, &evaluation)?;
        let original_rows = checked(geometry.client(), "paired fixed source binding", |a| {
            let c = a.client();
            let input = velocity.input();
            let n = input.owners.numel();
            metadata(a, &input.owners, n, DType::I64)?;
            metadata(a, &bands.section_owner, n, DType::I64)?;
            metadata(a, original_rows, n, DType::I64)?;
            for value in [
                &input.left_area,
                &input.left_discharge,
                &input.right_area,
                &input.right_discharge,
                &velocity.left,
                &velocity.right,
                &velocity.difference,
            ] {
                for field in [value.high(), value.low(), value.abs_error()] {
                    metadata(a, field, n, a.dtype())?;
                }
            }
            for mask in [
                &input.left_dry,
                &input.right_dry,
                &velocity.both_wet,
                &velocity.uniform,
            ] {
                metadata(a, mask, n, a.dtype())?;
            }
            reject(
                a,
                &c.ne(&input.owners, &bands.section_owner)?,
                "paired captured owners differ from the geometry rows. Select one bound source graph",
            )?;
            let captured = Tensor::empty(original_rows.shape(), DType::I64, c.device())?;
            c.copy_into(&captured, original_rows)?;
            let zero = Tensor::zeros(&[], DType::I64, c.device())?;
            reject(
                a,
                &c.lt(&captured, &zero)?,
                "paired original rows contain a negative identity. Supply nonnegative original rows",
            )?;
            for (source, area, dry) in [
                (sources[0], &input.left_area, &input.left_dry),
                (sources[1], &input.right_area, &input.right_dry),
            ] {
                metadata(a, source.owners(), n, DType::I64)?;
                reject(
                    a,
                    &c.ne(source.owners(), &input.owners)?,
                    "paired recovery owners differ from captured owners. Select matching source recoveries",
                )?;
                let recovered = source.requested_areas_m2();
                for (first, second) in [
                    (recovered.high(), area.high()),
                    (recovered.low(), area.low()),
                    (recovered.abs_error(), area.abs_error()),
                ] {
                    metadata(a, first, n, a.dtype())?;
                    reject(
                        a,
                        &c.ne(first, second)?,
                        "paired recovery area components differ from captured source area. Recover the captured exact source",
                    )?;
                }
                metadata(a, source.dry_mask(), n, a.dtype())?;
                reject(
                    a,
                    &c.ne(source.dry_mask(), dry)?,
                    "paired recovery dry mask differs from captured dry identity. Preserve the captured source state",
                )?;
            }
            Ok(captured)
        })?;
        let left = source_query(geometry, bands, sources[0], &original_rows)?;
        let right = source_query(geometry, bands, sources[1], &original_rows)?;
        let cache = checked(geometry.client(), "paired source witness binding", |a| {
            source_difference(a, bands, &left, sources[0], boundaries)?;
            source_difference(a, bands, &right, sources[1], boundaries)?;
            Ok([
                SourcePotentialCache::new(a, left, original_rows.numel())?,
                SourcePotentialCache::new(a, right, original_rows.numel())?,
            ])
        })?;
        let identity = Arc::new(RootSourceIdentity);
        let construction_witness = RootSourceWitness::capture(&identity, &evaluation);
        Ok(Self {
            geometry,
            bands,
            sources,
            velocity,
            boundaries,
            evaluation,
            original_rows,
            cache,
            panel_cache: Vec::new(),
            identity,
            construction_witness,
        })
    }

    pub(super) fn root_witness(&self) -> HydraulicResult<RootSourceWitness> {
        self.check_construction_policy()?;
        Ok(self.construction_witness.clone())
    }

    pub(super) fn check_root_witness(&self, witness: &RootSourceWitness) -> HydraulicResult<()> {
        self.check_construction_policy()?;
        witness.check(&self.identity, &self.evaluation)
    }

    fn check_construction_policy(&self) -> HydraulicResult<()> {
        self.construction_witness
            .check(&self.identity, &self.evaluation)
    }

    pub(super) fn select_rows(
        &self,
        rows: &Tensor<R>,
        query: &SupportedAreaQuery<R>,
    ) -> HydraulicResult<SelectedRows<R>> {
        checked(
            self.geometry.client(),
            "paired callback row admission",
            |a| {
                let c = a.client();
                let selected = self.select_sources(a, rows)?;
                metadata(
                    a,
                    &query.query.original_rows,
                    selected.rows.numel(),
                    DType::I64,
                )?;
                reject(
                    a,
                    &c.ne(&query.query.original_rows, &selected.original_rows)?,
                    "paired callback original rows differ from the selected source rows. Preserve synchronized query identities",
                )?;
                source_difference(
                    a,
                    &selected.bands,
                    query,
                    &selected.sources[0],
                    &selected.boundaries,
                )?;
                source_difference(
                    a,
                    &selected.bands,
                    query,
                    &selected.sources[1],
                    &selected.boundaries,
                )?;
                Ok(selected)
            },
        )
    }

    pub(super) fn select_sources(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<SelectedRows<R>> {
        let c = a.client();
        metadata(a, rows, rows.numel(), DType::I64)?;
        let captured = Tensor::empty(rows.shape(), DType::I64, c.device())?;
        c.copy_into(&captured, rows)?;
        // Velocity selection checks every row index before any remaining device gather.
        let velocity = self.velocity.select(a, &captured)?;
        let bands = self.bands.select(a, &captured)?;
        let sources = [
            self.sources[0].select(a, &captured)?,
            self.sources[1].select(a, &captured)?,
        ];
        let boundaries = self.boundaries.select(a, &captured)?;
        let original_rows = c.index_select(&self.original_rows, 0, &captured)?;
        Ok(SelectedRows {
            rows: captured,
            bands,
            sources,
            velocity,
            boundaries,
            original_rows,
        })
    }
}

impl<R> PairedSources<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    #[cfg(test)]
    pub(super) fn endpoint(
        &mut self,
        query: &SupportedAreaQuery<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<CurveValue<R>> {
        let rows = self.geometry.client().arange(
            0.0,
            self.original_rows.numel() as f64,
            1.0,
            DType::I64,
        )?;
        self.endpoint_on_rows(&rows, query, endpoint)
    }
    pub(super) fn endpoint_on_rows(
        &mut self,
        rows: &Tensor<R>,
        query: &SupportedAreaQuery<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<CurveValue<R>> {
        super::paired_residual::endpoint(self, rows, query, endpoint)
    }
    #[cfg(test)]
    pub(super) fn range(
        &mut self,
        query: &SupportedAreaQuery<R>,
    ) -> HydraulicResult<PairedRange<R>> {
        let rows = self.geometry.client().arange(
            0.0,
            self.original_rows.numel() as f64,
            1.0,
            DType::I64,
        )?;
        self.range_on_rows(&rows, query)
    }
    #[cfg(test)]
    pub(super) fn range_on_rows(
        &mut self,
        rows: &Tensor<R>,
        query: &SupportedAreaQuery<R>,
    ) -> HydraulicResult<PairedRange<R>> {
        let lower = self.endpoint_on_rows(rows, query, PhysicalEndpoint::Lower)?;
        let upper = self.endpoint_on_rows(rows, query, PhysicalEndpoint::Upper)?;
        let curve = checked(
            self.geometry.client(),
            "paired monotone residual range",
            |a| {
                let physical = a.bounds(lower.physical.lower(), upper.physical.upper())?;
                Ok(CurveValue {
                    retained: a.expansion_from_interval(&physical)?,
                    physical,
                    available: a.client().mul(&lower.available, &upper.available)?,
                })
            },
        )?;
        Ok(PairedRange {
            lower,
            upper,
            curve,
        })
    }
    #[cfg(test)]
    pub(super) fn source_invariant(&mut self) -> HydraulicResult<SourceInvariant<R>> {
        super::paired_source_invariant::evaluate(self)
    }
    pub(super) fn source_invariant_on_rows(
        &mut self,
        rows: &Tensor<R>,
    ) -> HydraulicResult<SourceInvariant<R>> {
        super::paired_source_invariant::evaluate_on_rows(self, rows)
    }
    pub(super) fn source_values(
        &mut self,
        rows: [&Tensor<R>; 2],
    ) -> HydraulicResult<[CurveValue<R>; 2]> {
        self.check_construction_policy()?;
        let [left, right] = &mut self.cache;
        Ok([
            paired_source_cache::ensure(
                left,
                self.geometry,
                self.bands,
                self.sources[0],
                self.boundaries,
                &self.evaluation,
                rows[0],
            )?,
            paired_source_cache::ensure(
                right,
                self.geometry,
                self.bands,
                self.sources[1],
                self.boundaries,
                &self.evaluation,
                rows[1],
            )?,
        ])
    }
}

fn metadata<R>(
    a: &Arithmetic<'_, R>,
    value: &Tensor<R>,
    count: usize,
    dtype: DType,
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
    if value.shape() != [count]
        || value.dtype() != dtype
        || !value.device().is_same(a.client().device())
    {
        return Err(invalid(format!(
            "paired source metadata shape={:?}, dtype={:?}, device={}: supply shape=[{count}], dtype={dtype:?}, device={}",
            value.shape(),
            value.dtype(),
            value.device().id(),
            a.client().device().id()
        )));
    }
    Ok(())
}
fn reject<R>(a: &Arithmetic<'_, R>, mask: &Tensor<R>, message: &str) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if mask.numel() != 0 && a.client().nonzero(mask)?.numel() != 0 {
        return Err(invalid(message.into()));
    }
    Ok(())
}
fn invalid(context: String) -> HydraulicError {
    CrossSectionError::InvalidInput {
        section: None,
        context,
    }
    .into()
}

#[cfg(test)]
#[path = "paired_residual_tests.rs"]
mod tests;
