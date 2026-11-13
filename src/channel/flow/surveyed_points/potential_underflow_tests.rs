use super::potential_absolute::endpoint_potential;
use super::potential_increment::endpoint_increment;
use super::potential_range::{PotentialContext, PotentialEvaluation};
use super::source_tests::support::{self, Client, Result, endpoints, hydraulic};
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry, PhysicalEndpoint},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

fn composed_underflow<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let section = support::batch::<R>(client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    // The existing canonical rectangle has unit width and a zero bed elevation.
    let owners = Tensor::from_slice(&[1_i64], &[1], client.device())?;
    let source_area = support::expansion::<R>(client, &[[0; 3]], true)?;
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: 1.0,
        ..PrismaticRiemannConfig::default()
    };
    let source = section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
    let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
        client,
        "composed positive underflow potential map",
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
            for endpoint in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
                let target = query.endpoint(a, &bands, endpoint)?;
                let raw = target.retained_endpoint_area(a, &bands, &boundaries)?;
                let high = raw.high().item::<f64>()?;
                let low = raw.low().item::<f64>()?;
                let radius = raw.abs_error().item::<f64>()?;
                assert_eq!(high, 0.0);
                assert_eq!(low, 0.0);
                assert!(radius > 0.0);
                let exact_dry = high == 0.0 && low == 0.0 && radius == 0.0;
                assert!(!exact_dry);
                let raw_bounds = a.expansion_interval(&raw)?;
                assert!(raw_bounds.lower().item::<f64>()? < 0.0);
                let physical = geometry.recover_query(a, &bands, &target)?.state;
                assert_eq!(physical.area.lower().item::<f64>()?, 0.0);
                assert!(physical.area.upper().item::<f64>()? > 0.0);
                assert_eq!(physical.wet_possible.item::<f64>()?, 1.0);
            }
            Ok((bands, boundaries, query))
        },
    ))?;
    let work = WorkLedger::new(&config);
    let context = hydraulic(PotentialContext::new(
        &geometry,
        &bands,
        &source,
        &query,
        &boundaries,
        PotentialEvaluation {
            config: &config,
            panels: 4,
            work: &work,
        },
    ))?;
    // A=2^-1075 gives Phi=2*sqrt(A)=sqrt(2)*2^-537.
    // Thus 2^-537 < Phi < 2^-536, using exact normal powers of two.
    let truth_lower = f64::from_bits((1023_u64 - 537) << 52);
    let truth_upper = f64::from_bits((1023_u64 - 536) << 52);
    for endpoint in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
        let absolute = hydraulic(endpoint_potential(&context, endpoint).inspect_err(|error| {
            eprintln!("composed underflow endpoint_potential: {error:?}");
        }))?;
        let increment = hydraulic(endpoint_increment(&context, endpoint).inspect_err(|error| {
            eprintln!("composed underflow endpoint_increment: {error:?}");
        }))?;
        for (name, value) in [("absolute", &absolute), ("increment", &increment)] {
            if let Some(analytic) = &value.analytic {
                assert_eq!(analytic.eligible.item::<f64>()?, 0.0, "{name}");
            }
            let projected = support::bounds::<R>(client, &value.retained)?;
            for interval in [&value.physical, &projected] {
                let actual = endpoints(interval)?[0];
                assert!(
                    actual[0].is_finite() && actual[1].is_finite(),
                    "{name}: {actual:?}"
                );
                assert!(
                    actual[0] <= truth_lower && truth_upper <= actual[1],
                    "{name}: {actual:?} excludes the independent power-of-two enclosure"
                );
                assert!(actual[1] > 0.0, "{name}: {actual:?}");
            }
            assert!(value.physical.lower().item::<f64>()? >= 0.0, "{name}");
        }
        hydraulic(error::checked::<R, _>(
            client,
            "retained underflow input after composition",
            |a| {
                let target = query.endpoint(a, &bands, endpoint)?;
                let raw = target.retained_endpoint_area(a, &bands, &boundaries)?;
                assert_eq!(raw.high().item::<f64>()?, 0.0);
                assert_eq!(raw.low().item::<f64>()?, 0.0);
                assert!(raw.abs_error().item::<f64>()? > 0.0);
                Ok(())
            },
        ))?;
    }
    Ok(())
}

#[test]
fn configured_cpu_composed_potential_retains_positive_underflow_water() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    composed_underflow::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_composed_potential_retains_positive_underflow_water() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    composed_underflow::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
