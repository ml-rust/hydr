use super::super::{AreaAnchor, AreaDirection, AreaQuery, tests::test_support as support};
use super::ConditionalBandSegments;
use crate::channel::{
    flow::surveyed::error,
    section::{HydraulicGeometry, PrismaticRecoveryConfig},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
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
            let segments = ConditionalBandSegments::prepare(a, &bands, &source, &dry, &boundaries)?;
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
            let same = ConditionalBandSegments::prepare(a, &bands, &source, &same, &boundaries)?;
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

fn uncertainty<R>(client: &R::Client) -> Result
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
            let ambiguous =
                ConditionalBandSegments::prefix_from_exact_dry(a, &bands, &ambiguous, &boundaries)?;
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

#[test]
fn cpu_clipped_endpoints_preserve_physical_identity_and_raw_authority() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    geometry::<CpuRuntime>(
        &CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_clipped_endpoints_preserve_physical_identity_and_raw_authority() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    geometry::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
