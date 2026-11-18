mod clipped_endpoint;

use super::super::super::area_point_query::PrismaticAreaPointBandCoordinates;
#[cfg(test)]
use super::super::super::context::Context;
use super::ConditionalBandSegments;
#[cfg(test)]
use super::selection::capture;
use crate::channel::flow::surveyed::{
    analytic_bottom::kinds,
    error::{Arithmetic, HydraulicResult},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps,
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
}

#[cfg(test)]
mod tests {
    use super::clipped_endpoint::tests::uncertainty;
    use super::*;
    use crate::channel::section::prismatic::area_query::{
        AreaAnchor, AreaDirection, AreaQuery, test_support as support,
    };
    use crate::channel::{
        flow::surveyed::error,
        section::{HydraulicGeometry, PrismaticRecoveryConfig},
    };
    use numr::ops::BinaryOps;
    use support::{Client, Result, hydraulic};

    fn geometry<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let foreign = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let foreign_geometry = HydraulicGeometry::from_batch(&foreign);
        let owners = Tensor::from_slice(&[2_i64, 1, 4], &[3], client.device())?;
        let source_area = support::expansion::<R>(
            client,
            &[
                [4.0_f64.to_bits(), 0, 0],
                [2.0_f64.to_bits(), 0, 0],
                [1.5_f64.to_bits(), 0, 0],
            ],
            true,
        )?;
        let source = section.recover_area_points_tensor(
            &owners,
            &source_area,
            &PrismaticRecoveryConfig::default(),
        )?;
        hydraulic(error::checked::<R, _>(
            client,
            "clipped endpoint identities",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zeros = Tensor::zeros(&[3], DType::F64, client.device())?;
                let scale = a.expansion_point(&zeros)?;
                let parameter = a.point(&zeros)?;
                let dry = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let segments =
                    ConditionalBandSegments::prepare(a, &bands, &source, &dry, &boundaries)?;
                let lanes = Tensor::from_slice(&[6_i64, 0, 3, 6, 4], &[5], client.device())?;
                let endpoints = segments.clipped_endpoints(a, &lanes)?;
                assert_eq!(endpoints.owners().try_to_vec::<i64>()?, [4, 2, 1, 4, 1]);
                assert_eq!(
                    endpoints.original_rows().try_to_vec::<i64>()?,
                    [2, 0, 1, 2, 1]
                );
                assert_eq!(
                    endpoints.band_indices().try_to_vec::<i64>()?,
                    [0, 0, 0, 0, 1]
                );
                assert_eq!(
                    endpoints.valid_mask().try_to_vec::<f64>()?,
                    [1.0, 1.0, 1.0, 1.0, 0.0]
                );
                assert_eq!(
                    endpoints
                        .compiled_bottom_width()
                        .lower()
                        .try_to_vec::<f64>()?,
                    [0.0, 2.0, 2.0, 0.0, 2.0]
                );
                assert_eq!(
                    endpoints.rectangular_band0_mask().try_to_vec::<f64>()?,
                    [0.0, 0.0, 1.0, 0.0, 0.0]
                );
                assert_eq!(
                    endpoints.pointed_band0_mask().try_to_vec::<f64>()?,
                    [1.0, 0.0, 0.0, 1.0, 0.0]
                );
                let target = endpoints.target();
                assert_eq!(
                    target.exact_dry_mask().try_to_vec::<f64>()?,
                    [1.0, 1.0, 1.0, 1.0, 0.0]
                );
                assert_eq!(
                    target.raw_available_mask().try_to_vec::<f64>()?,
                    [1.0, 1.0, 1.0, 1.0, 0.0]
                );
                assert_eq!(
                    target.top_width().lower().try_to_vec::<f64>()?,
                    [0.0, 2.0, 2.0, 0.0, 0.0]
                );
                for field in [
                    target.depth(),
                    target.integral(),
                    target.hydraulic_depth(),
                    target.physical_area(),
                ] {
                    assert_eq!(field.lower().try_to_vec::<f64>()?, [0.0; 5]);
                    assert_eq!(field.upper().try_to_vec::<f64>()?, [0.0; 5]);
                }
                for component in [
                    target.retained_area().high(),
                    target.retained_area().low(),
                    target.retained_area().abs_error(),
                ] {
                    assert_eq!(component.try_to_vec::<f64>()?, [0.0; 5]);
                }
                // The global source lies above band zero. Its clipped endpoint is the original first knot.
                let clipped = endpoints.source();
                assert_eq!(clipped.retained_area().high().try_to_vec::<f64>()?[1], 3.0);
                assert_eq!(clipped.at_end_mask().try_to_vec::<f64>()?[1], 1.0);
                assert_eq!(clipped.depth().lower().try_to_vec::<f64>()?[1], 1.0);
                assert_eq!(clipped.top_width().lower().try_to_vec::<f64>()?[1], 4.0);
                assert_eq!(
                    clipped
                        .coordinates()
                        .area_below_end_m2()
                        .lower()
                        .contiguous()?
                        .try_to_vec::<f64>()?[1],
                    0.0
                );
                let same = AreaQuery::source_affine(
                    a,
                    &bands,
                    &source,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let same =
                    ConditionalBandSegments::prepare(a, &bands, &source, &same, &boundaries)?;
                let first = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                assert_eq!(same.zero_gap_mask().try_to_vec::<f64>()?[0], 1.0);
                let shared = same.clipped_endpoints(a, &first)?;
                assert!(endpoints.same_model(&shared));
                assert_eq!(shared.source().retained_area().high().item::<f64>()?, 3.0);
                assert_eq!(shared.target().retained_area().high().item::<f64>()?, 3.0);
                assert_eq!(shared.source().at_end_mask().item::<f64>()?, 1.0);
                assert_eq!(shared.target().at_end_mask().item::<f64>()?, 1.0);
                let foreign_bands = foreign_geometry.gather(a, &owners)?;
                let foreign_boundaries = foreign_geometry.boundary_areas(a)?.gather(a, &owners)?;
                assert!(
                    ConditionalBandSegments::prepare(
                        a,
                        &foreign_bands,
                        &source,
                        &dry,
                        &foreign_boundaries
                    )
                    .is_err()
                );
                let foreign_dry = AreaQuery::band_affine(
                    a,
                    &foreign_bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &foreign_bands)?;
                let foreign_prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &foreign_bands,
                    &foreign_dry,
                    &foreign_boundaries,
                )?;
                let foreign_endpoints = foreign_prefix.segments.clipped_endpoints(a, &first)?;
                assert!(!endpoints.same_model(&foreign_endpoints));
                let raw = clipped.retained_area().high().try_to_vec::<f64>()?;
                client.copy_into(
                    segments.start_area.value.high(),
                    &Tensor::zeros(
                        segments.start_area.value.high().shape(),
                        DType::F64,
                        client.device(),
                    )?,
                )?;
                client.copy_into(
                    segments.original_rows(),
                    &Tensor::zeros(
                        segments.original_rows().shape(),
                        DType::I64,
                        client.device(),
                    )?,
                )?;
                assert_eq!(
                    endpoints
                        .source()
                        .retained_area()
                        .high()
                        .try_to_vec::<f64>()?,
                    raw
                );
                assert_eq!(
                    endpoints.original_rows().try_to_vec::<i64>()?,
                    [2, 0, 1, 2, 1]
                );
                let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
                let empty = segments.clipped_endpoints(a, &empty)?;
                assert_eq!(empty.target().physical_area().lower().shape(), &[0]);
                assert_eq!(
                    empty.target().coordinates().candidate_mask().shape(),
                    &[0, 1]
                );
                Ok(())
            },
        ))?;
        uncertainty::<R>(client)
    }

    #[test]
    fn cpu_clipped_endpoints_preserve_physical_identity_and_raw_authority() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        geometry::<CpuRuntime>(
            &CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_clipped_endpoints_preserve_physical_identity_and_raw_authority() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        geometry::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
