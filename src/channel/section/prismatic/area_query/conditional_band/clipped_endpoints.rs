#[cfg(test)]
use super::super::super::context::Context;
use super::super::super::{
    area_point_inverse, area_point_query::PrismaticAreaPointBandCoordinates, band_geometry::masked,
    hydraulic_geometry::query_error,
};
use super::super::{evaluation, fields, inverse};
#[cfg(test)]
use super::selection::capture;
use super::{
    ConditionalBandSegments, endpoint_authority::StoredEndpointArea, types::LocalEndpoint,
};
use crate::channel::flow::surveyed::{
    analytic_bottom::kinds,
    error::{Arithmetic, HydraulicResult, PrismaticRiemannRejection},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
#[cfg(test)]
use std::sync::Arc;

#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "tests read the depth, integral, hydraulic depth and masks; production runs their fallible geometry"
    )
)]
pub(crate) struct ClippedEndpointGeometry<R: Runtime<DType = DType>> {
    retained_area: CertifiedExpansion<R>,
    raw_available: Tensor<R>,
    physical_area: Interval<R>,
    depth: Interval<R>,
    width: Interval<R>,
    integral: Interval<R>,
    hydraulic_depth: Interval<R>,
    coordinates: PrismaticAreaPointBandCoordinates<R>,
    at_start: Tensor<R>,
    at_end: Tensor<R>,
    exact_dry: Tensor<R>,
}

pub(crate) struct ClippedBandEndpoints<R: Runtime<DType = DType>> {
    #[cfg(test)]
    model: Arc<Context<R>>,
    #[cfg(test)]
    owners: Tensor<R>,
    #[cfg(test)]
    original_rows: Tensor<R>,
    #[cfg(test)]
    band_indices: Tensor<R>,
    #[cfg(test)]
    valid: Tensor<R>,
    bottom_width: Interval<R>,
    rectangular: Tensor<R>,
    pointed: Tensor<R>,
    source: ClippedEndpointGeometry<R>,
    target: ClippedEndpointGeometry<R>,
}

impl<R: Runtime<DType = DType>> ClippedBandEndpoints<R> {
    pub(crate) fn source(&self) -> &ClippedEndpointGeometry<R> {
        &self.source
    }
    pub(crate) fn target(&self) -> &ClippedEndpointGeometry<R> {
        &self.target
    }
    #[cfg(test)]
    pub(crate) fn owners(&self) -> &Tensor<R> {
        &self.owners
    }
    #[cfg(test)]
    pub(crate) fn original_rows(&self) -> &Tensor<R> {
        &self.original_rows
    }
    #[cfg(test)]
    pub(crate) fn band_indices(&self) -> &Tensor<R> {
        &self.band_indices
    }
    #[cfg(test)]
    pub(crate) fn valid_mask(&self) -> &Tensor<R> {
        &self.valid
    }
    pub(crate) fn compiled_bottom_width(&self) -> &Interval<R> {
        &self.bottom_width
    }
    pub(crate) fn rectangular_band0_mask(&self) -> &Tensor<R> {
        &self.rectangular
    }
    pub(crate) fn pointed_band0_mask(&self) -> &Tensor<R> {
        &self.pointed
    }
    #[cfg(test)]
    pub(crate) fn same_model(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.model, &other.model)
    }
}

impl<R: Runtime<DType = DType>> ClippedEndpointGeometry<R> {
    /// Raw authority requires raw_available. Unavailable lanes retain their unclipped input, never a projected-area lift.
    pub(crate) fn retained_area(&self) -> &CertifiedExpansion<R> {
        &self.retained_area
    }
    pub(crate) fn raw_available_mask(&self) -> &Tensor<R> {
        &self.raw_available
    }
    pub(crate) fn physical_area(&self) -> &Interval<R> {
        &self.physical_area
    }
    #[cfg(test)]
    pub(crate) fn depth(&self) -> &Interval<R> {
        &self.depth
    }
    pub(crate) fn top_width(&self) -> &Interval<R> {
        &self.width
    }
    #[cfg(test)]
    pub(crate) fn integral(&self) -> &Interval<R> {
        &self.integral
    }
    #[cfg(test)]
    pub(crate) fn hydraulic_depth(&self) -> &Interval<R> {
        &self.hydraulic_depth
    }
    pub(crate) fn coordinates(&self) -> &PrismaticAreaPointBandCoordinates<R> {
        &self.coordinates
    }
    #[cfg(test)]
    pub(crate) fn at_start_mask(&self) -> &Tensor<R> {
        &self.at_start
    }
    #[cfg(test)]
    pub(crate) fn at_end_mask(&self) -> &Tensor<R> {
        &self.at_end
    }
    #[cfg(test)]
    pub(crate) fn exact_dry_mask(&self) -> &Tensor<R> {
        &self.exact_dry
    }
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
    pub(crate) fn clipped_endpoints(
        &self,
        a: &mut Arithmetic<'_, R>,
        lane_rows: &Tensor<R>,
    ) -> HydraulicResult<ClippedBandEndpoints<R>> {
        let selected = self.select(a, lane_rows)?;
        let c = a.client();
        let n = selected.active.numel();
        let valid = selected.rows.valid.reshape(&[n])?;
        let band0 = c.mul(
            &valid,
            &c.cast(
                &c.eq(
                    &selected.band_index,
                    &Tensor::zeros(&[], DType::I64, c.device())?,
                )?,
                DType::F64,
            )?,
        )?;
        let offsets = c.mul_scalar(selected.owners(), selected.rows.model.maximum_bands as f64)?;
        // Select the original band-zero column directly, without gathering every band for flattened owners.
        let bottom_width = a.bounds(
            &c.index_select(
                &selected.rows.model.start_width.lower().flatten()?,
                0,
                &offsets,
            )?,
            &c.index_select(
                &selected.rows.model.start_width.upper().flatten()?,
                0,
                &offsets,
            )?,
        )?;
        let slope = a.bounds(
            &selected.rows.slope.lower().reshape(&[n])?,
            &selected.rows.slope.upper().reshape(&[n])?,
        )?;
        let (rectangular, pointed) = kinds(
            a,
            &band0,
            &selected.rows.pointed.reshape(&[n])?,
            &bottom_width,
            &slope,
        )?;
        let source = selected.clipped_endpoint(a, &selected.start, &selected.start_area, &band0)?;
        let target = selected.clipped_endpoint(a, &selected.end, &selected.end_area, &band0)?;
        Ok(ClippedBandEndpoints {
            #[cfg(test)]
            model: Arc::clone(&selected.rows.model),
            #[cfg(test)]
            owners: capture(a, selected.owners())?,
            #[cfg(test)]
            original_rows: capture(a, selected.original_rows())?,
            #[cfg(test)]
            band_indices: capture(a, selected.band_indices())?,
            #[cfg(test)]
            valid: capture(a, &valid)?,
            bottom_width,
            rectangular,
            pointed,
            source,
            target,
        })
    }

    fn clipped_endpoint(
        &self,
        a: &mut Arithmetic<'_, R>,
        local: &LocalEndpoint<R>,
        raw: &StoredEndpointArea<R>,
        band0: &Tensor<R>,
    ) -> HydraulicResult<ClippedEndpointGeometry<R>> {
        let c = a.client();
        let n = self.active.numel();
        let zero = &self.rows.model.source.zero;
        let rows = self.rows.rows(a)?;
        let valid = rows.valid.reshape(&[n])?;
        let raw_zero = c.mul(
            &c.eq(raw.value.high(), zero)?,
            &c.mul(
                &c.eq(raw.value.low(), zero)?,
                &c.eq(raw.value.abs_error(), zero)?,
            )?,
        )?;
        let at_start = c.mul(&valid, &local.at_start.reshape(&[n])?)?;
        let exact_dry = c.mul(band0, &c.maximum(&at_start, &raw_zero)?)?;
        let at_start = c.maximum(&at_start, &exact_dry)?;
        let dry_column = exact_dry.reshape(&[n, 1])?;
        let above = a.bounds(
            &c.where_cond(&dry_column, zero, local.above_start.lower())?,
            &c.where_cond(&dry_column, zero, local.above_start.upper())?,
        )?;
        let below = a.bounds(
            &c.where_cond(
                &dry_column,
                rows.area_increment.lower(),
                local.below_end.lower(),
            )?,
            &c.where_cond(
                &dry_column,
                rows.area_increment.upper(),
                local.below_end.upper(),
            )?,
        )?;
        let recovered = if n == 0 {
            inverse::LocalInverse {
                candidate: rows.valid.clone(),
                area_above_start: above,
                area_below_end: below,
                depth_above_start: a.point(&Tensor::zeros(&[0, 1], DType::F64, c.device())?)?,
                depth_below_end: a.point(&Tensor::zeros(&[0, 1], DType::F64, c.device())?)?,
            }
        } else {
            inverse::recover(a, &self.rows.model, &rows, &rows.valid, &above, &below)
                .map_err(query_error)?
        };
        self.reject(&c.mul(&valid, &c.eq(&recovered.candidate, zero)?.reshape(&[n])?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "clipped endpoint has no compatible inverse band. Tighten endpoint or geometry certificates")?;
        let coordinates = PrismaticAreaPointBandCoordinates {
            candidate: recovered.candidate,
            area_above_start: recovered.area_above_start,
            area_below_end: recovered.area_below_end,
            depth_above_start: recovered.depth_above_start,
            depth_below_end: recovered.depth_below_end,
        };
        let values = fields::polynomials(
            a,
            &self.rows.model,
            &rows,
            coordinates.candidate_mask(),
            coordinates.depth_above_start_m(),
        )
        .map_err(query_error)?;
        let available = c.maximum(&raw.available, &exact_dry)?;
        let retained_area = a.expansion_bounds(
            &c.where_cond(&exact_dry, zero, raw.value.high())?,
            &c.where_cond(&exact_dry, zero, raw.value.low())?,
            &c.where_cond(&exact_dry, zero, raw.value.abs_error())?,
        )?;
        let projected_input = a.expansion_bounds(
            &c.where_cond(&available, retained_area.high(), zero)?,
            &c.where_cond(&available, retained_area.low(), zero)?,
            &c.where_cond(&available, retained_area.abs_error(), zero)?,
        )?;
        let projection = a.expansion_interval(&projected_input)?;
        let forward = &values.forward_area;
        let lower = c.maximum(&forward.lower().reshape(&[n])?, zero)?;
        let upper = forward.upper().reshape(&[n])?;
        let lower = c.where_cond(&available, &c.maximum(&lower, projection.lower())?, &lower)?;
        let upper = c.where_cond(&available, &c.minimum(&upper, projection.upper())?, &upper)?;
        self.reject(&c.mul(&valid, &c.gt(&lower, &upper)?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "clipped endpoint area conflicts with retained authority. Tighten endpoint or geometry certificates")?;
        let physical_area = a.bounds(
            &c.where_cond(&valid, &lower, zero)?,
            &c.where_cond(&valid, &upper, zero)?,
        )?;
        let hydraulic = evaluation::hydraulic(
            a,
            &self.rows.model,
            &rows,
            &physical_area,
            &coordinates,
            &values.top_width,
        )?;
        let depth = area_point_inverse::absolute_depth(a, &self.rows.model, &rows, &coordinates)
            .map_err(query_error)?;
        let flatten =
            |a: &mut Arithmetic<'_, R>, value: &Interval<R>| -> HydraulicResult<Interval<R>> {
                let value = masked(a, &rows.valid, value, zero).map_err(query_error)?;
                Ok(a.bounds(&value.lower().reshape(&[n])?, &value.upper().reshape(&[n])?)?)
            };
        Ok(ClippedEndpointGeometry {
            retained_area,
            raw_available: c.mul(&available, &c.ge(projection.lower(), zero)?)?,
            physical_area,
            depth: flatten(a, &depth)?,
            width: flatten(a, &values.top_width)?,
            integral: flatten(a, &values.integral)?,
            hydraulic_depth: flatten(a, &hydraulic)?,
            coordinates,
            at_start,
            at_end: c.mul(&valid, &local.at_end.reshape(&[n])?)?,
            exact_dry,
        })
    }
}
