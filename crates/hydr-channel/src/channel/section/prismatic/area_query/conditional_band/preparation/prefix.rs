use super::SourceOrigin;
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use crate::channel::section::prismatic::{
    area_query::{
        SupportedAreaQuery,
        conditional_band::{ConditionalBandSegments, PreparedAreaPrefix},
    },
    boundary_area::BoundaryAreas,
    hydraulic_geometry::GatheredBands,
};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

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
    pub(crate) fn prefix_from_exact_dry(
        a: &mut Arithmetic<'_, R>,
        bands: &GatheredBands<R>,
        endpoint: &SupportedAreaQuery<R>,
        boundaries: &BoundaryAreas<R>,
    ) -> HydraulicResult<PreparedAreaPrefix<R>> {
        let endpoint_area = endpoint.retained_endpoint_area(a, bands, boundaries)?;
        let zeros = Tensor::zeros(bands.section_owner.shape(), DType::F64, a.client().device())?;
        let origin = a.expansion_point(&zeros)?;
        let segments = Self::prepare_origin(
            a,
            bands,
            endpoint,
            boundaries,
            SourceOrigin {
                area: &origin,
                delta: &endpoint_area,
                target_area: &endpoint_area,
                exact_dry: true,
            },
        )?;
        Ok(PreparedAreaPrefix {
            segments,
            endpoint_area,
            band_count: bands.band_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::section::prismatic::area_query::{
        AreaAnchor, AreaDirection, AreaQuery, test_support as support,
    };
    use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
    use numr::ops::BinaryOps;
    use support::{Client, Result, hydraulic};

    fn prefixes<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let foreign = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let foreign_geometry = HydraulicGeometry::from_batch(&foreign);
        let owners = Tensor::<R>::from_slice(&[2_i64, 1, 0], &[3], client.device())?;
        hydraulic(error::checked::<R, _>(
            client,
            "exact-dry prefix identities",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let foreign_boundaries = foreign_geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zero = Tensor::zeros(&[3], DType::F64, client.device())?;
                let scale = a.expansion_point(&zero)?;
                let parameter = a.point(&zero)?;
                let ends = Tensor::from_slice(&[1_i64, 0, 0], &[3], client.device())?;
                let endpoint = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandEnd(ends),
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                assert!(
                    endpoint
                        .retained_endpoint_area(a, &bands, &foreign_boundaries)
                        .is_err()
                );
                assert!(
                    ConditionalBandSegments::prefix_from_exact_dry(
                        a,
                        &bands,
                        &endpoint,
                        &foreign_boundaries
                    )
                    .is_err()
                );
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                assert_eq!(
                    prefix.segments.physical_gap().lower().try_to_vec::<f64>()?,
                    [
                        3.0,
                        4.5,
                        0.0,
                        6.0,
                        0.0,
                        0.0,
                        f64::from_bits(0x3feccccccccccccc),
                        0.0,
                        0.0
                    ]
                );
                assert_eq!(
                    prefix
                        .segments
                        .start
                        .above_start
                        .lower()
                        .try_to_vec::<f64>()?,
                    [0.0; 9]
                );
                let values = prefix.segments.retained_gap();
                let mut total = a.expansion_point(&zero)?;
                for band in 0..3 {
                    let term = a.expansion_bounds(
                        &values
                            .high()
                            .reshape(&[3, 3])?
                            .narrow(1, band, 1)?
                            .squeeze(Some(1)),
                        &values
                            .low()
                            .reshape(&[3, 3])?
                            .narrow(1, band, 1)?
                            .squeeze(Some(1)),
                        &values
                            .abs_error()
                            .reshape(&[3, 3])?
                            .narrow(1, band, 1)?
                            .squeeze(Some(1)),
                    )?;
                    total = a.expansion_add(&total, &term)?;
                }
                let difference = a.expansion_sub(&total, &prefix.endpoint_area)?;
                assert_eq!(difference.high().try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(difference.low().try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(difference.abs_error().try_to_vec::<f64>()?, [0.0; 3]);
                let rows = Tensor::from_slice(&[6_i64, 0, 6], &[3], client.device())?;
                let selected = prefix.segments.select(a, &rows)?;
                assert_eq!(selected.original_rows().try_to_vec::<i64>()?, [2, 0, 2]);
                assert_eq!(selected.owners().try_to_vec::<i64>()?, [0, 2, 0]);
                let retained = prefix.segments.retained_gap().high().try_to_vec::<f64>()?;
                client.copy_into(prefix.endpoint_area.high(), &zero)?;
                assert_eq!(
                    prefix.segments.retained_gap().high().try_to_vec::<f64>()?,
                    retained
                );
                let dry = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let dry =
                    ConditionalBandSegments::prefix_from_exact_dry(a, &bands, &dry, &boundaries)?;
                assert_eq!(dry.segments.zero_gap_mask().try_to_vec::<f64>()?, [1.0; 9]);
                assert_eq!(
                    dry.segments
                        .retained_gap()
                        .abs_error()
                        .try_to_vec::<f64>()?,
                    [0.0; 9]
                );
                Ok(())
            },
        ))?;
        local_offsets::<R>(client)
    }

    fn local_offsets<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::<R>::from_slice(&[0_i64], &[1], client.device())?;
        hydraulic(error::checked::<R, _>(
            client,
            "retained absolute expression and conditioned prefix",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let distance = f64::from_bits(0x35f0000000000000);
                let scale =
                    a.expansion_point(&Tensor::from_slice(&[distance], &[1], client.device())?)?;
                let parameter = a.point(&Tensor::ones(&[1], DType::F64, client.device())?)?;
                let anchor = Tensor::from_slice(&[1_i64], &[1], client.device())?;
                let endpoint = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandStart(anchor.clone()),
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                let prefix = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                assert!(prefix.endpoint_area.abs_error().item::<f64>()? > 0.0);
                assert_eq!(
                    prefix.segments.physical_gap().lower().try_to_vec::<f64>()?[1],
                    distance
                );
                assert_eq!(
                    prefix.segments.physical_gap().upper().try_to_vec::<f64>()?[1],
                    distance
                );
                assert_eq!(
                    prefix
                        .segments
                        .end
                        .above_start
                        .lower()
                        .try_to_vec::<f64>()?[1],
                    distance
                );
                let lane = Tensor::from_slice(&[1_i64], &[1], client.device())?;
                let panel = prefix.segments.panel(a, &lane, &parameter)?;
                assert_eq!(
                    panel
                        .coordinates
                        .area_above_start_m2()
                        .lower()
                        .item::<f64>()?,
                    distance
                );
                assert_eq!(
                    panel
                        .coordinates
                        .area_above_start_m2()
                        .upper()
                        .item::<f64>()?,
                    distance
                );
                client.copy_into(
                    boundaries.areas().abs_error(),
                    &Tensor::full_scalar(
                        boundaries.areas().abs_error().shape(),
                        DType::F64,
                        f64::EPSILON,
                        client.device(),
                    )?,
                )?;
                let widened = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &endpoint,
                    &boundaries,
                )?;
                assert!(widened.endpoint_area.abs_error().item::<f64>()? >= f64::EPSILON);
                assert_eq!(
                    widened
                        .segments
                        .end
                        .above_start
                        .lower()
                        .try_to_vec::<f64>()?[1],
                    distance
                );
                assert_eq!(
                    widened
                        .segments
                        .end
                        .above_start
                        .upper()
                        .try_to_vec::<f64>()?[1],
                    distance
                );
                let zeros = Tensor::zeros(&[1], DType::F64, client.device())?;
                let zero_scale = a.expansion_point(&zeros)?;
                let mut uncertain = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::BandStart(anchor),
                    AreaDirection::Add,
                    &zero_scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                uncertain.query.offsets.lower = a.expansion_bounds(
                    &zeros,
                    &zeros,
                    &Tensor::from_slice(&[distance], &[1], client.device())?,
                )?;
                let uncertain = ConditionalBandSegments::prefix_from_exact_dry(
                    a,
                    &bands,
                    &uncertain,
                    &boundaries,
                )?;
                assert!(uncertain.endpoint_area.abs_error().item::<f64>()? >= distance);
                assert_eq!(
                    uncertain.segments.zero_gap_mask().try_to_vec::<f64>()?[1],
                    0.0
                );
                assert!(
                    uncertain
                        .segments
                        .physical_gap()
                        .upper()
                        .try_to_vec::<f64>()?[1]
                        > 0.0
                );
                assert!(
                    uncertain
                        .segments
                        .retained_gap()
                        .abs_error()
                        .try_to_vec::<f64>()?[1]
                        > 0.0
                );
                Ok(())
            },
        ))
    }

    #[test]
    fn cpu_exact_dry_prefixes_retain_local_offsets_and_absolute_radii() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        prefixes::<CpuRuntime>(
            &CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
        )
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_exact_dry_prefixes_retain_local_offsets_and_absolute_radii() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        prefixes::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
