use super::potential_absolute::endpoint_potential;
use super::potential_range::{PotentialContext, PotentialEvaluation};
use super::source_tests::support::{Client, Result, components, endpoints, hydraulic};
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{
        AreaAnchor, AreaDirection, AreaQuery, CrossSectionBatch, CrossSectionDefinition,
        CrossSectionPoint, HydraulicGeometry, PhysicalEndpoint, PrismaticRecoveryConfig,
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

fn composed_band_contribution<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let coordinates = [
        (-1.0, 3.0),
        (0.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (2.0, 3.0),
    ];
    let definition = CrossSectionDefinition {
        id: "rectangular bottom and widening upper band".into(),
        vertical_datum: "Exact binary retention control".into(),
        points: coordinates
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 3.0,
        profile_provenance: "Independent original-coordinate area and width identities".into(),
    };
    let section =
        CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let area = hydraulic(error::checked::<R, _>(
        &client,
        "band retention source area",
        |a| Ok(a.expansion_point(&Tensor::ones(&[1], DType::F64, client.device())?)?),
    ))?;
    let source =
        section.recover_area_points_tensor(&owners, &area, &PrismaticRecoveryConfig::default())?;
    let delta = 2.0_f64.powi(-80);
    let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
        &client,
        "band retention endpoint chart",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let scale = a.expansion_point(&Tensor::from_slice(&[delta], &[1], client.device())?)?;
            let parameter = a.bounds(
                &Tensor::zeros(&[1], DType::F64, client.device())?,
                &Tensor::ones(&[1], DType::F64, client.device())?,
            )?;
            let indices = Tensor::from_slice(&[1_i64], &[1], client.device())?;
            let query = AreaQuery::band_affine(
                a,
                &bands,
                AreaAnchor::BandStart(indices),
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            Ok((bands, boundaries, query))
        },
    ))?;
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: 1.0,
        ..PrismaticRiemannConfig::default()
    };
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
    let base = hydraulic(endpoint_potential(&context, PhysicalEndpoint::Lower))?;
    let target = hydraulic(endpoint_potential(&context, PhysicalEndpoint::Upper))?;
    let base_before = components(&base.retained)?;
    let target_before = components(&target.retained)?;
    let (difference, centre) = hydraulic(error::checked::<R, _>(
        &client,
        "retained composed band increment",
        |a| {
            let difference = a.expansion_sub(&target.retained, &base.retained)?;
            // This diagnostic centre does not replace either physical certificate or its retained radius.
            let zero = Tensor::zeros(difference.high().shape(), DType::F64, client.device())?;
            let centre = a.expansion_bounds(difference.high(), difference.low(), &zero)?;
            let centre = a.expansion_interval(&centre)?;
            Ok((difference, centre))
        },
    ))?;
    assert_eq!(base_before, components(&base.retained)?);
    assert_eq!(target_before, components(&target.retained)?);
    let difference_parts = components(&difference)?[0].map(f64::from_bits);
    assert!(difference_parts[2] >= f64::from_bits(base_before[0][2]));
    assert!(difference_parts[2] >= f64::from_bits(target_before[0][2]));
    assert!(difference_parts[2] > 0.0);

    // For x=h−1: A=1+x+x²/2, T=1+x, Phi(1)=2, and dPhi/dA=1/sqrt(A*T).
    // At A=1+delta, 0<=x<=delta gives delta−delta² <= DeltaPhi <= delta.
    // delta² is below the predecessor spacing, so these directed F64 endpoints enclose that exact range.
    let truth = [f64::from_bits(delta.to_bits() - 1), delta];
    let allowance = f64::from_bits((delta * 1e-10).to_bits() - 1);
    let reference_uncertainty = truth[1] - truth[0];
    assert!(reference_uncertainty <= delta * 1e-12);
    let actual = endpoints(&centre)?[0];
    assert!(
        actual[0] > 0.0 && actual[1] > 0.0,
        "retained centre={actual:?}"
    );
    let endpoint_error = (actual[0] - truth[1])
        .abs()
        .max((actual[1] - truth[0]).abs());
    assert!(
        endpoint_error <= allowance,
        "retained band error={endpoint_error:e}, allowance={allowance:e}, centre={actual:?}"
    );
    assert_eq!(work.snapshot().root_solves, 0);
    assert_eq!(work.snapshot().refinements, 0);
    Ok(())
}

#[test]
fn configured_cpu_composed_potential_retains_a_band_below_prefix_spacing() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    composed_band_contribution::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_composed_potential_retains_a_band_below_prefix_spacing() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    composed_band_contribution::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
