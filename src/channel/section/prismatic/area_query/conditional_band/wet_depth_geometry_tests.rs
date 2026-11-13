use super::super::{AreaAnchor, AreaDirection, AreaQuery, tests::test_support as support};
use super::ConditionalBandSegments;
use crate::channel::{
    flow::surveyed::error::{self, HydraulicError, PrismaticRiemannRejection},
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
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
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
            let prefix =
                ConditionalBandSegments::prefix_from_exact_dry(a, &bands, &endpoint, &boundaries)?;
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
        &CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_wet_depth_geometry_preserves_signed_gaps_and_original_rows() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    geometry::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
