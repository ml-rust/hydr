use super::super::{
    ConditionalBandSegments, endpoint_authority::StoredEndpointArea, types::LocalEndpoint,
};
use super::ClippedEndpointGeometry;
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicResult, PrismaticRiemannRejection,
};
use crate::channel::section::prismatic::{
    area_point_inverse,
    area_query::{evaluation, fields, inverse},
    band_geometry::masked,
    hydraulic_geometry::query_error,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;

use crate::channel::section::prismatic::area_point_query::PrismaticAreaPointBandCoordinates;

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
    pub(super) fn clipped_endpoint(
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

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::channel::section::prismatic::area_query::{
        AreaAnchor, AreaDirection, AreaQuery, test_support as support,
    };
    use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
    use support::{Client, Result, hydraulic};

    pub(in super::super) fn uncertainty<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
        hydraulic(error::checked::<R, _>(
            client,
            "clipped endpoint raw authority",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zeros = Tensor::zeros(&[1], DType::F64, client.device())?;
                let one = a.point(&Tensor::ones(&[1], DType::F64, client.device())?)?;
                let distance = f64::from_bits(0x35f0000000000000);
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[distance], &[1], client.device())?)?;
                let anchor = Tensor::from_slice(&[1_i64], &[1], client.device())?;
                let query = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandStart(anchor.clone()),
                    AreaDirection::Add,
                    &scale,
                    &one,
                )?
                .prove_support(a, &bands)?;
                let prefix =
                    ConditionalBandSegments::prefix_from_exact_dry(a, &bands, &query, &boundaries)?;
                let lane = Tensor::from_slice(&[1_i64], &[1], client.device())?;
                let endpoints = prefix.segments.clipped_endpoints(a, &lane)?;
                assert!(
                    endpoints
                        .target()
                        .retained_area()
                        .abs_error()
                        .item::<f64>()?
                        > 0.0
                );
                assert_eq!(
                    endpoints
                        .target()
                        .coordinates()
                        .area_above_start_m2()
                        .lower()
                        .item::<f64>()?,
                    distance
                );
                assert_eq!(endpoints.target().at_start_mask().item::<f64>()?, 0.0);
                let zero_scale = a.expansion_point(&zeros)?;
                let mut ambiguous = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandStart(anchor),
                    AreaDirection::Add,
                    &zero_scale,
                    &one,
                )?
                .prove_support(a, &bands)?;
                // A retained offset ball crosses the physical knot. Neither clipping alternative owns its raw area.
                ambiguous.query.offsets.lower = a.expansion_bounds(
                    &zeros,
                    &zeros,
                    &Tensor::from_slice(&[distance], &[1], client.device())?,
                )?;
                let ambiguous = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &ambiguous,
                    &boundaries,
                )?;
                let endpoint = ambiguous.segments.clipped_endpoints(a, &lane)?;
                assert_eq!(endpoint.target().raw_available_mask().item::<f64>()?, 0.0);
                assert_eq!(endpoint.target().at_start_mask().item::<f64>()?, 0.0);
                assert_eq!(endpoint.target().exact_dry_mask().item::<f64>()?, 0.0);
                assert!(
                    endpoint
                        .target()
                        .retained_area()
                        .abs_error()
                        .item::<f64>()?
                        >= distance
                );
                assert!(endpoint.target().physical_area().lower().item::<f64>()? > 0.0);
                Ok(())
            },
        ))?;
        let owners = Tensor::from_slice(&[1_i64], &[1], client.device())?;
        hydraulic(error::checked::<R, _>(
            client,
            "nonexact dry-containing endpoint",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale = a.expansion_point(&Tensor::from_slice(
                    &[f64::from_bits(1)],
                    &[1],
                    client.device(),
                )?)?;
                let parameter = a.point(&Tensor::from_slice(&[0.5], &[1], client.device())?)?;
                let query = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let prefix =
                    ConditionalBandSegments::prefix_from_exact_dry(a, &bands, &query, &boundaries)?;
                let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let endpoint = prefix.segments.clipped_endpoints(a, &first)?;
                let target = endpoint.target();
                assert_eq!(target.retained_area().high().item::<f64>()?, 0.0);
                assert_eq!(target.retained_area().low().item::<f64>()?, 0.0);
                assert!(target.retained_area().abs_error().item::<f64>()? > 0.0);
                assert_eq!(target.raw_available_mask().item::<f64>()?, 0.0);
                assert_eq!(target.exact_dry_mask().item::<f64>()?, 0.0);
                assert_eq!(target.at_start_mask().item::<f64>()?, 0.0);
                assert!(target.physical_area().upper().item::<f64>()? > 0.0);
                Ok(())
            },
        ))
    }
}
