use super::super::{ConditionalBandSegments, selection::capture};
use super::WetDepthGeometry;
use crate::channel::flow::surveyed::error::{
    Arithmetic, HydraulicResult, PrismaticRiemannRejection,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps, UnaryOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
#[cfg(test)]
use std::sync::Arc;

impl<R> ConditionalBandSegments<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn wet_depth_geometry(
        &self,
        a: &mut Arithmetic<'_, R>,
        lane_rows: &Tensor<R>,
    ) -> HydraulicResult<WetDepthGeometry<R>> {
        let selected = self.select(a, lane_rows)?;
        let c = a.client();
        let n = selected.active.numel();
        let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
        let one = Tensor::ones(&[n], DType::F64, c.device())?;
        let first_parameter = a.point(&zero)?;
        let last_parameter = a.point(&one)?;
        let first = selected.evaluate_point(a, &first_parameter)?;
        let last = selected.evaluate_point(a, &last_parameter)?;
        let safe = |a: &mut Arithmetic<'_, R>,
                    value: &Interval<R>,
                    fallback: &Tensor<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.where_cond(&selected.active, value.lower(), fallback)?,
                &c.where_cond(&selected.active, value.upper(), fallback)?,
            )?)
        };
        let first_width = safe(a, &first.top_width, &one)?;
        let last_width = safe(a, &last.top_width, &one)?;
        let first_expansion = a.expansion_from_interval(&first_width)?;
        let last_expansion = a.expansion_from_interval(&last_width)?;
        let half = a.expansion_point(&c.mul_scalar(&one, 0.5)?)?;
        let half_first = a.expansion_mul(&first_expansion, &half)?;
        let half_last = a.expansion_mul(&last_expansion, &half)?;
        let mean = a.expansion_add(&half_first, &half_last)?;
        let mean = a.expansion_interval(&mean)?;
        // The physical mean lies between endpoint widths, including subnormal half-products.
        let mean = a.bounds(
            &c.maximum(
                mean.lower(),
                &c.minimum(first_width.lower(), last_width.lower())?,
            )?,
            &c.minimum(
                mean.upper(),
                &c.maximum(first_width.upper(), last_width.upper())?,
            )?,
        )?;
        let mean = a.expansion_from_interval(&mean)?;
        let depth_gap = a.expansion_div_positive(&selected.gap, &mean)?;
        let projection = a.expansion_interval(&depth_gap)?;
        let from_start = a.sub(
            last.coordinates.depth_above_start_m(),
            first.coordinates.depth_above_start_m(),
        )?;
        let from_end = a.sub(
            first.coordinates.depth_below_end_m(),
            last.coordinates.depth_below_end_m(),
        )?;
        let thickness = selected.rows.thickness.upper().reshape(&[n])?;
        let lower = c.maximum(
            &c.maximum(
                &c.maximum(projection.lower(), &from_start.lower().reshape(&[n])?)?,
                &from_end.lower().reshape(&[n])?,
            )?,
            &c.neg(&thickness)?,
        )?;
        let upper = c.minimum(
            &c.minimum(
                &c.minimum(projection.upper(), &from_start.upper().reshape(&[n])?)?,
                &from_end.upper().reshape(&[n])?,
            )?,
            &thickness,
        )?;
        let lower = c.where_cond(
            &c.ge(selected.physical_gap.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?;
        let upper = c.where_cond(
            &c.le(selected.physical_gap.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?;
        selected.reject(&c.mul(&selected.active, &c.gt(&lower, &upper)?)?, PrismaticRiemannRejection::GeometryUnresolved,
            "wet depth gap is incompatible with local endpoints. Tighten endpoint or geometry certificates")?;
        let physical_depth_gap = a.bounds(
            &c.where_cond(&selected.active, &lower, &zero)?,
            &c.where_cond(&selected.active, &upper, &zero)?,
        )?;
        let hull = |a: &mut Arithmetic<'_, R>,
                    first: &Interval<R>,
                    last: &Interval<R>|
         -> HydraulicResult<Interval<R>> {
            Ok(a.bounds(
                &c.minimum(first.lower(), last.lower())?,
                &c.maximum(first.upper(), last.upper())?,
            )?)
        };
        let area_hull = hull(a, &first.area, &last.area)?;
        let width_hull = hull(a, &first.top_width, &last.top_width)?;
        let slope = a.bounds(
            &selected.rows.slope.lower().reshape(&[n])?,
            &selected.rows.slope.upper().reshape(&[n])?,
        )?;
        Ok(WetDepthGeometry {
            #[cfg(test)]
            model: Arc::clone(&selected.rows.model),
            owners: capture(a, selected.owners())?,
            original_rows: capture(a, selected.original_rows())?,
            active: capture(a, selected.active_mask())?,
            area: safe(a, &first.area, &one)?,
            width: first_width,
            slope: safe(a, &slope, &zero)?,
            #[cfg(test)]
            depth_gap,
            physical_depth_gap,
            area_hull: safe(a, &area_hull, &one)?,
            width_hull: safe(a, &width_hull, &one)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::section::prismatic::area_query::{
        AreaAnchor, AreaDirection, AreaQuery, test_support as support,
    };
    use crate::channel::{
        flow::surveyed::error::{self, HydraulicError},
        section::{HydraulicGeometry, PrismaticRecoveryConfig},
    };
    use support::{Client, Result, hydraulic};

    fn geometry<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::<R>::from_slice(&[1_i64, 1, 3, 3], &[4], client.device())?;
        let areas = support::expansion::<R>(
            client,
            &[
                [2.0_f64.to_bits(), 0, 0],
                [4.0_f64.to_bits(), 0, 0],
                [1.0_f64.to_bits(), 0, 0],
                [2.0_f64.to_bits(), 0, 0],
            ],
            true,
        )?;
        let source = section.recover_area_points_tensor(
            &owners,
            &areas,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "wet local depth-gap geometry",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale = a.expansion_point(&Tensor::from_slice(
                    &[2.0, 2.0, 1.0, 1.0],
                    &[4],
                    client.device(),
                )?)?;
                let parameter = a.point(&Tensor::from_slice(
                    &[1.0, -1.0, 1.0, -1.0],
                    &[4],
                    client.device(),
                )?)?;
                let endpoint = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &endpoint, &boundaries)?;
                let lanes = Tensor::from_slice(&[0_i64, 3, 6, 9, 1, 4], &[6], client.device())?;
                let wet = segments.wet_depth_geometry(a, &lanes)?;
                assert_eq!(wet.owners().try_to_vec::<i64>()?, [1, 1, 3, 3, 1, 1]);
                assert_eq!(wet.original_rows().try_to_vec::<i64>()?, [0, 1, 2, 3, 0, 1]);
                assert_eq!(
                    wet.active_mask().try_to_vec::<f64>()?,
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
                );
                let high = wet.retained_depth_gap().high().try_to_vec::<f64>()?;
                let low = wet.retained_depth_gap().low().try_to_vec::<f64>()?;
                let radius = wet.retained_depth_gap().abs_error().try_to_vec::<f64>()?;
                assert_eq!(&high[..2], &[1.0, -1.0]);
                assert_eq!(&low[..2], &[0.0; 2]);
                assert_eq!(&radius[..2], &[0.0; 2]);
                assert!(radius[2] > 0.0 && radius[3] > 0.0);
                let curved = Tensor::from_slice(&[2_i64, 3], &[2], client.device())?;
                let curved_parameter =
                    a.point(&Tensor::from_slice(&[0.25, 0.75], &[2], client.device())?)?;
                let curved_value = crate::channel::flow::surveyed::derivatives::wet_integrand(
                    a,
                    &wet,
                    &curved,
                    &curved_parameter,
                    1.0,
                )?;
                assert!(curved_value.lower().try_to_vec::<f64>()?[0] > 0.0);
                assert!(curved_value.upper().try_to_vec::<f64>()?[1] < 0.0);
                let curved_fourth = crate::channel::flow::surveyed::derivatives::wet_fourth_bound(
                    a,
                    &wet,
                    &curved,
                    &curved_parameter,
                    1.0,
                )?
                .try_to_vec::<f64>()?;
                assert!(
                    curved_fourth
                        .iter()
                        .all(|value| value.is_finite() && *value >= 0.0)
                );
                assert_eq!(
                    wet.retained_depth_gap().abs_error().try_to_vec::<f64>()?,
                    radius
                );
                let lower = wet.physical_depth_gap().lower().try_to_vec::<f64>()?;
                let upper = wet.physical_depth_gap().upper().try_to_vec::<f64>()?;
                assert_eq!(&lower[..2], &[1.0, -1.0]);
                assert_eq!(&upper[..2], &[1.0, -1.0]);
                assert!(lower[2] > 0.0 && upper[3] < 0.0);
                assert_eq!(&lower[4..], &[0.0; 2]);
                assert_eq!(&upper[4..], &[0.0; 2]);
                for value in [
                    wet.start_area(),
                    wet.start_width(),
                    wet.area_hull(),
                    wet.width_hull(),
                ] {
                    let lower = value.lower().try_to_vec::<f64>()?;
                    assert_eq!(&lower[4..], &[1.0; 2]);
                }
                assert_eq!(&wet.slope().lower().try_to_vec::<f64>()?[4..], &[0.0; 2]);
                let reordered = Tensor::from_slice(&[9_i64, 0, 9], &[3], client.device())?;
                let other = segments.wet_depth_geometry(a, &reordered)?;
                assert!(wet.same_model(&other));
                assert_eq!(other.original_rows().try_to_vec::<i64>()?, [3, 0, 3]);
                client.copy_into(
                    segments.rows.slope.lower(),
                    &Tensor::zeros(
                        segments.rows.slope.lower().shape(),
                        DType::F64,
                        client.device(),
                    )?,
                )?;
                assert_eq!(wet.retained_depth_gap().high().try_to_vec::<f64>()?, high);
                assert_eq!(
                    wet.retained_depth_gap().abs_error().try_to_vec::<f64>()?,
                    radius
                );
                let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
                let empty_wet = segments.wet_depth_geometry(a, &empty)?;
                assert_eq!(empty_wet.physical_depth_gap().lower().shape(), &[0]);
                let zero_scale =
                    a.expansion_point(&Tensor::zeros(&[4], DType::F64, client.device())?)?;
                let zero_parameter = a.point(&Tensor::zeros(&[4], DType::F64, client.device())?)?;
                let dry_endpoint = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &zero_scale,
                    &zero_parameter,
                )?
                .prove_support(a, &bands)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                assert!(matches!(
                    prefix.segments.wet_depth_geometry(a, &first),
                    Err(HydraulicError::Rejected {
                        reason: PrismaticRiemannRejection::QuadratureUnresolved,
                        ..
                    })
                ));
                let dry = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &dry_endpoint,
                    &boundaries,
                )?;
                assert_eq!(
                    dry.segments
                        .wet_depth_geometry(a, &first)?
                        .active_mask()
                        .item::<f64>()?,
                    0.0
                );
                Ok(())
            },
        ))
    }

    #[test]
    fn cpu_wet_depth_geometry_preserves_signed_gaps_and_original_rows() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        geometry::<CpuRuntime>(
            &CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_wet_depth_geometry_preserves_signed_gaps_and_original_rows() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        geometry::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
