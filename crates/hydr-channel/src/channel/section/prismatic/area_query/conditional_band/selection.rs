use super::super::super::config;
use super::super::super::hydraulic_geometry::GatheredBands;
use super::super::admission;
use super::endpoint_authority;
use super::{ConditionalBandSegments, types::LocalEndpoint};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::sync::Arc;

pub(super) fn capture<R>(a: &Arithmetic<'_, R>, value: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), a.client().device())?;
    a.client().copy_into(&owned, value)?;
    Ok(owned)
}

pub(super) fn compact<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    m: usize,
) -> HydraulicResult<GatheredBands<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let shape = [bands.section_owner.numel(), bands.band_count];
    let column = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
        Ok(a.bounds(
            &value.lower().reshape(&[m, 1])?,
            &value.upper().reshape(&[m, 1])?,
        )?)
    };
    let cap = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
        Ok(a.bounds(
            &value
                .lower()
                .unsqueeze(1)?
                .broadcast_to(&shape)?
                .contiguous()?
                .reshape(&[m])?,
            &value
                .upper()
                .unsqueeze(1)?
                .broadcast_to(&shape)?
                .contiguous()?
                .reshape(&[m])?,
        )?)
    };
    Ok(GatheredBands {
        model: Arc::clone(&bands.model),
        section_owner: capture(
            a,
            &bands
                .section_owner
                .unsqueeze(1)?
                .broadcast_to(&shape)?
                .contiguous()?
                .reshape(&[m])?,
        )?,
        band_count: 1,
        valid: capture(a, &bands.valid.contiguous()?.reshape(&[m, 1])?)?,
        pointed: capture(a, &bands.pointed.contiguous()?.reshape(&[m, 1])?)?,
        start_depth: column(a, &bands.start_depth)?,
        end_depth: column(a, &bands.end_depth)?,
        thickness: column(a, &bands.thickness)?,
        start_area: column(a, &bands.start_area)?,
        end_area: column(a, &bands.end_area)?,
        area_increment: column(a, &bands.area_increment)?,
        start_width: column(a, &bands.start_width)?,
        slope: column(a, &bands.slope)?,
        start_integral: column(a, &bands.start_integral)?,
        cap_depth: cap(a, &bands.cap_depth)?,
        cap_area: cap(a, &bands.cap_area)?,
        cap_width: cap(a, &bands.cap_width)?,
        cap_integral: cap(a, &bands.cap_integral)?,
    })
}

pub(super) fn interval<R>(
    a: &mut Arithmetic<'_, R>,
    value: &Interval<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<Interval<R>>
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
    Ok(a.bounds(
        &c.index_select(value.lower(), 0, rows)?,
        &c.index_select(value.upper(), 0, rows)?,
    )?)
}

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        admission::metadata(a, rows, &[rows.numel()], DType::I64)?;
        config::check_work(rows.numel(), 1)?;
        let c = a.client();
        let captured = capture(a, rows)?;
        let rows = &captured;
        let endpoint = |a: &mut Arithmetic<'_, R>,
                        value: &LocalEndpoint<R>|
         -> HydraulicResult<LocalEndpoint<R>> {
            Ok(LocalEndpoint {
                above_start: interval(a, &value.above_start, rows)?,
                below_end: interval(a, &value.below_end, rows)?,
                at_start: c.index_select(&value.at_start, 0, rows)?,
                at_end: c.index_select(&value.at_end, 0, rows)?,
            })
        };
        let gap: CertifiedExpansion<R> = a.expansion_bounds(
            &c.index_select(self.gap.high(), 0, rows)?,
            &c.index_select(self.gap.low(), 0, rows)?,
            &c.index_select(self.gap.abs_error(), 0, rows)?,
        )?;
        Ok(Self {
            start_area: endpoint_authority::select(a, &self.start_area, rows)?,
            end_area: endpoint_authority::select(a, &self.end_area, rows)?,
            rows: self.rows.select(a, rows)?,
            original_rows: c.index_select(&self.original_rows, 0, rows)?,
            band_index: c.index_select(&self.band_index, 0, rows)?,
            active: c.index_select(&self.active, 0, rows)?,
            zero_gap: c.index_select(&self.zero_gap, 0, rows)?,
            start: endpoint(a, &self.start)?,
            end: endpoint(a, &self.end)?,
            gap,
            physical_gap: interval(a, &self.physical_gap, rows)?,
            endpoint_hull: interval(a, &self.endpoint_hull, rows)?,
        })
    }
}
