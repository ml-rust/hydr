use super::super::{AreaDirection, AreaQuery, test_support as support};
use super::ConditionalBandSegments;
use crate::channel::{
    flow::surveyed::{
        derivatives::{WetDepthPotential, wet_fourth_bound, wet_integrand},
        error::{self, HydraulicError, PrismaticRiemannRejection},
    },
    section::{HydraulicGeometry, PrismaticRecoveryConfig},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use support::{Client, Result, hydraulic};

fn physical_constructor<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = Tensor::<R>::from_slice(&[1_i64, 1], &[2], client.device())?;
    let areas = support::expansion::<R>(
        client,
        &[[2.0_f64.to_bits(), 0, 0], [4.0_f64.to_bits(), 0, 0]],
        false,
    )?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    hydraulic(error::checked::<R, _>(
        client,
        "signed wet physical constructor",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale =
                a.expansion_point(&Tensor::from_slice(&[2.0; 2], &[2], client.device())?)?;
            let parameter = a.point(&Tensor::from_slice(&[1.0, -1.0], &[2], client.device())?)?;
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
            let lanes = Tensor::from_slice(&[0_i64, 3, 1, 4], &[4], client.device())?;
            let wet = segments.wet_depth_geometry(a, &lanes)?;
            let quadrature = Tensor::from_slice(&[0_i64, 1, 0, 1, 2, 3], &[6], client.device())?;
            let x = a.point(&Tensor::from_slice(
                &[0.0, 1.0, 0.5, 0.5, 0.0, 1.0],
                &[6],
                client.device(),
            )?)?;
            let evaluator = WetDepthPotential::from_geometry(a, &wet, &quadrature, &x, 1.0)?;
            let value = evaluator.value(a)?;
            let fourth = evaluator.fourth_derivative(a)?;
            let lower = value.lower().try_to_vec::<f64>()?;
            let upper = value.upper().try_to_vec::<f64>()?;
            for (lane, expected) in [1.0_f64, -1.0].into_iter().enumerate() {
                assert!(lower[lane] <= expected && expected <= upper[lane]);
                assert!(
                    (lower[lane] - expected)
                        .abs()
                        .max((upper[lane] - expected).abs())
                        <= 1e-10
                );
            }
            assert!(lower[2] > 0.0 && upper[3] < 0.0);
            assert_eq!(&lower[4..], &[0.0; 2]);
            assert_eq!(&upper[4..], &[0.0; 2]);
            let lower = fourth.lower().try_to_vec::<f64>()?;
            let upper = fourth.upper().try_to_vec::<f64>()?;
            // Rectangle G=(1+x)^(-1/2), or -(2-x)^(-1/2), has endpoint fourth derivatives +/-105/16.
            for (lane, expected) in [6.5625_f64, -6.5625].into_iter().enumerate() {
                assert!(lower[lane] <= expected && expected <= upper[lane]);
                assert!(
                    (lower[lane] - expected)
                        .abs()
                        .max((upper[lane] - expected).abs())
                        <= 1e-10 * expected.abs()
                );
            }
            assert_eq!(&lower[4..], &[0.0; 2]);
            assert_eq!(&upper[4..], &[0.0; 2]);
            let wrapped = wet_integrand(a, &wet, &quadrature, &x, 1.0)?;
            assert_eq!(
                wrapped.lower().try_to_vec::<f64>()?,
                value.lower().try_to_vec::<f64>()?
            );
            let bound = wet_fourth_bound(a, &wet, &quadrature, &x, 1.0)?.try_to_vec::<f64>()?;
            assert!(
                bound[..2]
                    .iter()
                    .all(|&bound| (6.5625..=6.5625 * (1.0 + 1e-10)).contains(&bound))
            );
            assert_eq!(&bound[4..], &[0.0; 2]);
            let reversed = Tensor::from_slice(&[1_i64, 0, 1], &[3], client.device())?;
            let whole = a.bounds(
                &Tensor::zeros(&[3], DType::F64, client.device())?,
                &Tensor::ones(&[3], DType::F64, client.device())?,
            )?;
            let broad = wet_integrand(a, &wet, &reversed, &whole, 1.0)?;
            assert!(broad.upper().try_to_vec::<f64>()?[0] < 0.0);
            assert!(broad.lower().try_to_vec::<f64>()?[1] > 0.0);
            let outside = a.point(&Tensor::from_slice(&[1.5; 3], &[3], client.device())?)?;
            assert!(matches!(
                WetDepthPotential::from_geometry(a, &wet, &reversed, &outside, 1.0),
                Err(HydraulicError::Rejected {
                    reason: PrismaticRiemannRejection::QuadratureUnresolved,
                    query: Some(1),
                    section_owner: Some(1),
                    ..
                })
            ));
            let empty = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
            let empty_parameter = a.point(&Tensor::zeros(&[0], DType::F64, client.device())?)?;
            assert_eq!(
                wet_integrand(a, &wet, &empty, &empty_parameter, 1.0)?
                    .lower()
                    .shape(),
                &[0]
            );
            client.copy_into(
                wet.start_area().lower(),
                &Tensor::full_scalar(&[4], DType::F64, -10.0, client.device())?,
            )?;
            client.copy_into(
                wet.start_area().upper(),
                &Tensor::full_scalar(&[4], DType::F64, -10.0, client.device())?,
            )?;
            assert!(
                matches!(WetDepthPotential::from_geometry(a, &wet, &quadrature, &x, 1.0), Err(HydraulicError::Rejected { reason: PrismaticRiemannRejection::QuadratureUnresolved, ref context, .. }) if context.contains("physical hull"))
            );
            Ok(())
        },
    ))
}

#[test]
fn cpu_wet_depth_potential_uses_signed_physical_geometry_and_private_recurrence() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    physical_constructor::<CpuRuntime>(
        &CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_wet_depth_potential_uses_signed_physical_geometry_and_private_recurrence() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    physical_constructor::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
