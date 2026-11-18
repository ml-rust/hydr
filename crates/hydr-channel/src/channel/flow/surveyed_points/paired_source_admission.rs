use super::super::{
    paired_source_cache::SourcePotentialCache,
    potential_range::{PotentialEvaluation, admit_evaluation, source_query},
    source_binding::{RootSourceIdentity, RootSourceWitness},
    source_terms::SourceTerms,
    types::SourceVelocity,
};
use super::{PairedSources, SelectedRows};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked},
    section::{
        BoundaryAreas, CrossSectionError, GatheredBands, HydraulicGeometry,
        PrismaticAreaPointRecovery, SupportedAreaQuery, source_difference,
    },
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use std::sync::Arc;

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
    pub(in crate::channel::flow::surveyed_points) fn new(
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
        let count = original_rows.numel();
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
            source_terms: [SourceTerms::new(count), SourceTerms::new(count)],
            identity,
            construction_witness,
        })
    }

    pub(in crate::channel::flow::surveyed_points) fn select_rows(
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

    pub(in crate::channel::flow::surveyed_points) fn select_sources(
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
