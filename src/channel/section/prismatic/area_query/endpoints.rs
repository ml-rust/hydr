use super::super::hydraulic_geometry::GatheredBands;
use super::{AreaDirection, AreaQuery, SupportedAreaQuery, admission, range, selection};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum PhysicalEndpoint {
    Lower,
    Upper,
}

impl<R> SupportedAreaQuery<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Preserve the continuous area expression at one physical endpoint.
    /// Its transformed area retains every arithmetic radius.
    pub(crate) fn endpoint(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        endpoint: PhysicalEndpoint,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        let parameter = match (self.query.direction, endpoint) {
            (AreaDirection::Add, PhysicalEndpoint::Lower)
            | (AreaDirection::Subtract, PhysicalEndpoint::Upper) => self.query.parameter.lower(),
            (AreaDirection::Add, PhysicalEndpoint::Upper)
            | (AreaDirection::Subtract, PhysicalEndpoint::Lower) => self.query.parameter.upper(),
        };
        let parameter = a.point(parameter)?;
        self.reconstruct(a, bands, parameter)
    }

    /// Rebind the retained affine map and prove support for its complete new parameter image.
    pub(crate) fn with_parameter_bounds(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::chart(a, bands, &self.query.anchor, &self.query.scale, parameter)?;
        admission::metadata(
            a,
            &self.query.original_rows,
            bands.section_owner.shape(),
            DType::I64,
        )?;
        let parameter = a.bounds(parameter.lower(), parameter.upper())?;
        self.reconstruct(a, bands, parameter)
    }

    /// Reprove only changed rows. Preserve every untouched certificate and the complete original affine expression.
    pub(crate) fn with_selected_parameter_bounds(
        mut self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        rows: &Tensor<R>,
        parameter: &Interval<R>,
    ) -> HydraulicResult<Self> {
        admission::matching_model(Arc::ptr_eq(&self.query.model, &bands.model))?;
        admission::matching_owners(a, bands, &self.query.owners)?;
        admission::metadata(a, rows, &[rows.numel()], DType::I64)?;
        admission::metadata(a, parameter.lower(), &[rows.numel()], a.dtype())?;
        admission::metadata(a, parameter.upper(), &[rows.numel()], a.dtype())?;
        let c = a.client();
        let count = i64::try_from(self.query.owners.numel()).map_err(|_| {
            crate::channel::section::CrossSectionError::input(
                "selected parameter row count exceeds I64 indexing. Split the query",
            )
        })?;
        let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
        let upper = Tensor::from_slice(&[count], &[], c.device())?;
        let invalid = c.maximum(&c.lt(rows, &zero)?, &c.ge(rows, &upper)?)?;
        admission::reject(
            a,
            &c.cast(&invalid, a.dtype())?,
            "selected parameter row lies outside the supported query. Supply admitted rows",
        )?;
        if rows.numel() > 1 {
            let sorted = c.sort(rows, 0, false)?;
            let n = rows.numel() - 1;
            let duplicates = c.eq(&sorted.narrow(0, 0, n)?, &sorted.narrow(0, 1, n)?)?;
            admission::reject(
                a,
                &c.cast(&duplicates, a.dtype())?,
                "selected parameter rows repeat a destination. Supply unique rows",
            )?;
        }
        if rows.numel() == 0 {
            return Ok(self);
        }
        let selected_bands = bands.select(a, rows)?;
        let selected =
            self.select(a, rows)?
                .with_parameter_bounds(a, &selected_bands, parameter)?;
        let interval = |a: &mut Arithmetic<'_, R>,
                        old: &Interval<R>,
                        new: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.index_put(old.lower(), 0, rows, new.lower())?,
                &c.index_put(old.upper(), 0, rows, new.upper())?,
            )?)
        };
        let expansion =
            |a: &mut Arithmetic<'_, R>,
             old: &solvr::common::arithmetic::CertifiedExpansion<R>,
             new: &solvr::common::arithmetic::CertifiedExpansion<R>|
             -> HydraulicResult<solvr::common::arithmetic::CertifiedExpansion<R>> {
                Ok(a.expansion_bounds(
                    &c.index_put(old.high(), 0, rows, new.high())?,
                    &c.index_put(old.low(), 0, rows, new.low())?,
                    &c.index_put(old.abs_error(), 0, rows, new.abs_error())?,
                )?)
            };
        self.query.parameter = interval(a, &self.query.parameter, &selected.query.parameter)?;
        self.query.offsets.lower =
            expansion(a, &self.query.offsets.lower, &selected.query.offsets.lower)?;
        self.query.offsets.upper =
            expansion(a, &self.query.offsets.upper, &selected.query.offsets.upper)?;
        self.area = interval(a, &self.area, &selected.area)?;
        self.above_start = interval(a, &self.above_start, &selected.above_start)?;
        self.below_end = interval(a, &self.below_end, &selected.below_end)?;
        self.candidate = c.index_put(&self.candidate, 0, rows, &selected.candidate)?;
        self.lower_endpoint.above_start = interval(
            a,
            &self.lower_endpoint.above_start,
            &selected.lower_endpoint.above_start,
        )?;
        self.lower_endpoint.below_end = interval(
            a,
            &self.lower_endpoint.below_end,
            &selected.lower_endpoint.below_end,
        )?;
        self.upper_endpoint.above_start = interval(
            a,
            &self.upper_endpoint.above_start,
            &selected.upper_endpoint.above_start,
        )?;
        self.upper_endpoint.below_end = interval(
            a,
            &self.upper_endpoint.below_end,
            &selected.upper_endpoint.below_end,
        )?;
        Ok(self)
    }

    fn reconstruct(
        &self,
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        parameter: Interval<R>,
    ) -> HydraulicResult<Self> {
        let offsets = range::offsets(a, &self.query.scale, &parameter, self.query.direction)?;
        let anchor = selection::capture_anchor(a, &self.query.anchor)?;
        let scale = a.expansion_bounds(
            self.query.scale.high(),
            self.query.scale.low(),
            self.query.scale.abs_error(),
        )?;
        let c = a.client();
        let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
            c.copy_into(&owned, value)?;
            Ok(owned)
        };
        AreaQuery {
            model: Arc::clone(&self.query.model),
            anchor,
            offsets,
            direction: self.query.direction,
            scale,
            parameter,
            owners: copy(&self.query.owners)?,
            original_rows: copy(&self.query.original_rows)?,
            band_count: self.query.band_count,
        }
        .prove_support(a, bands)
    }
}
