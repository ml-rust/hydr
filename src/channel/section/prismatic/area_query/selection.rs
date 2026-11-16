use super::super::area_point_query::PrismaticAreaPointBandCoordinates;
use super::{
    admission,
    chart::{AreaAnchor, AreaQuery, EndpointDifferences, ExpandedBounds, SupportedAreaQuery},
};
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

pub(super) fn capture_anchor<R>(
    a: &mut Arithmetic<'_, R>,
    anchor: &AreaAnchor<R>,
) -> HydraulicResult<AreaAnchor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    #[cfg(test)]
    let c = a.client();
    #[cfg(test)]
    let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
        c.copy_into(&owned, value)?;
        Ok(owned)
    };
    Ok(match anchor {
        AreaAnchor::Zero => AreaAnchor::Zero,
        AreaAnchor::Cap => AreaAnchor::Cap,
        #[cfg(test)]
        AreaAnchor::BandStart(index) => AreaAnchor::BandStart(copy(index)?),
        #[cfg(test)]
        AreaAnchor::BandEnd(index) => AreaAnchor::BandEnd(copy(index)?),
        AreaAnchor::Source {
            area,
            coordinates: values,
        } => AreaAnchor::Source {
            area: Box::new(a.expansion_bounds(area.high(), area.low(), area.abs_error())?),
            coordinates: Box::new(coordinates(a, values, None)?),
        },
    })
}

pub(in crate::channel::section::prismatic) fn coordinates<R>(
    a: &mut Arithmetic<'_, R>,
    values: &PrismaticAreaPointBandCoordinates<R>,
    rows: Option<&Tensor<R>>,
) -> HydraulicResult<PrismaticAreaPointBandCoordinates<R>>
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
    let gather = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        match rows {
            Some(rows) => Ok(c.index_select(value, 0, rows)?),
            None => {
                let copy = Tensor::empty(value.shape(), value.dtype(), c.device())?;
                c.copy_into(&copy, value)?;
                Ok(copy)
            }
        }
    };
    let bounds = |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
        match rows {
            Some(_) => Ok(a.bounds(&gather(value.lower())?, &gather(value.upper())?)?),
            None => Ok(a.bounds(value.lower(), value.upper())?),
        }
    };
    Ok(PrismaticAreaPointBandCoordinates {
        candidate: gather(values.candidate_mask())?,
        area_above_start: bounds(a, values.area_above_start_m2())?,
        area_below_end: bounds(a, values.area_below_end_m2())?,
        depth_above_start: bounds(a, values.depth_above_start_m())?,
        depth_below_end: bounds(a, values.depth_below_end_m())?,
    })
}

pub(in crate::channel::section::prismatic) fn expansion<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    Ok(a.expansion_bounds(
        &c.index_select(value.high(), 0, rows)?,
        &c.index_select(value.low(), 0, rows)?,
        &c.index_select(value.abs_error(), 0, rows)?,
    )?)
}

pub(in crate::channel::section::prismatic) fn interval<R>(
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

impl<R> AreaQuery<R>
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
        super::super::config::check_work(rows.numel(), self.band_count + 1)?;
        let c = a.client();
        let anchor = match &self.anchor {
            AreaAnchor::Zero => AreaAnchor::Zero,
            AreaAnchor::Cap => AreaAnchor::Cap,
            #[cfg(test)]
            AreaAnchor::BandStart(index) => AreaAnchor::BandStart(c.index_select(index, 0, rows)?),
            #[cfg(test)]
            AreaAnchor::BandEnd(index) => AreaAnchor::BandEnd(c.index_select(index, 0, rows)?),
            AreaAnchor::Source {
                area,
                coordinates: values,
            } => AreaAnchor::Source {
                area: Box::new(expansion(a, area, rows)?),
                coordinates: Box::new(coordinates(a, values, Some(rows))?),
            },
        };
        Ok(Self {
            model: Arc::clone(&self.model),
            anchor,
            offsets: ExpandedBounds {
                lower: expansion(a, &self.offsets.lower, rows)?,
                upper: expansion(a, &self.offsets.upper, rows)?,
            },
            direction: self.direction,
            scale: expansion(a, &self.scale, rows)?,
            parameter: interval(a, &self.parameter, rows)?,
            owners: c.index_select(&self.owners, 0, rows)?,
            original_rows: c.index_select(&self.original_rows, 0, rows)?,
            band_count: self.band_count,
        })
    }
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
    pub(crate) fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        Ok(Self {
            query: self.query.select(a, rows)?,
            area: interval(a, &self.area, rows)?,
            above_start: interval(a, &self.above_start, rows)?,
            below_end: interval(a, &self.below_end, rows)?,
            candidate: c.index_select(&self.candidate, 0, rows)?,
            lower_endpoint: EndpointDifferences {
                above_start: interval(a, &self.lower_endpoint.above_start, rows)?,
                below_end: interval(a, &self.lower_endpoint.below_end, rows)?,
            },
            upper_endpoint: EndpointDifferences {
                above_start: interval(a, &self.upper_endpoint.above_start, rows)?,
                below_end: interval(a, &self.upper_endpoint.below_end, rows)?,
            },
        })
    }
}
