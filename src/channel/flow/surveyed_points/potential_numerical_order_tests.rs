use super::super::potential_range::{PotentialContext, PotentialEvaluation};
use super::super::potential_range_tests::frozen;
use super::super::source_tests::support::{
    Client, Result, components, endpoints, expansion, hydraulic,
};
use super::wet_contributions;
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{
        AreaDirection, AreaQuery, ConditionalBandSegments, CrossSectionBatch,
        CrossSectionDefinition, CrossSectionPoint, HydraulicGeometry, PrismaticRecoveryConfig,
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

fn face_band_order<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let profile = frozen::query_profiles::PROFILES[2];
    let section = CrossSectionBatch::<R>::new(
        client.clone(),
        vec![CrossSectionDefinition {
            id: profile.id.into(),
            vertical_datum: "Frozen potential datum".into(),
            points: profile
                .coordinates
                .iter()
                .map(|point| CrossSectionPoint {
                    station_m: f64::from_bits(point[0]),
                    elevation_m: f64::from_bits(point[1]),
                })
                .collect(),
            maximum_stage_m: f64::from_bits(profile.cap_stage),
            profile_provenance: "Independent original-coordinate potential reference".into(),
        }],
    )?
    .certify_prismatic()?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let cases = [
        frozen::query_profiles::CASES[13],
        frozen::query_profiles::CASES[17],
        frozen::query_profiles::CASES[18],
    ];
    assert_eq!(
        cases.map(|case| case.id),
        [
            "three_band_broad_reverse__physical_anchor",
            "first_knot_from_second_source",
            "second_knot_from_first_source",
        ]
    );
    let owners = Tensor::from_slice(&[0_i64; 3], &[3], client.device())?;
    let source_components =
        cases.map(|case| [case.source_components[0], case.source_components[1], 0]);
    let source_area = expansion::<R>(&client, &source_components, true)?;
    let target_components = [
        [1.0_f64.to_bits(), 0, 0],
        [3.0_f64.to_bits(), (-2.0_f64.powi(-80)).to_bits(), 0],
        [7.5_f64.to_bits(), 2.0_f64.powi(-80).to_bits(), 0],
    ];
    let target_area = expansion::<R>(&client, &target_components, true)?;
    let before_source = components(&source_area)?;
    let before_target = components(&target_area)?;
    let recovery = PrismaticRecoveryConfig::default();
    let source = section.recover_area_points_tensor(&owners, &source_area, &recovery)?;
    let target = section.recover_area_points_tensor(&owners, &target_area, &recovery)?;
    let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
        &client,
        "ordered potential test binding",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
            let scale = a.expansion_point(&zero)?;
            let parameter = a.point(&zero)?;
            let query = AreaQuery::source_affine(
                a,
                &bands,
                &target,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            Ok((bands, boundaries, query))
        },
    ))?;
    assert_eq!(bands.band_count, 3);
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
            panels: 1024,
            work: &work,
        },
    ))?;
    let segments = hydraulic(error::checked::<R, _>(
        &client,
        "ordered potential test segments",
        |a| ConditionalBandSegments::prepare(a, &bands, &source, &query, &boundaries),
    ))?;
    let work_before_rejection = work.snapshot();
    for rows in [
        Tensor::from_slice(&[6_148_914_691_236_517_206_i64], &[1], client.device())?,
        Tensor::from_slice(&[-1_i64], &[1], client.device())?,
        Tensor::from_slice(&[3_i64], &[1], client.device())?,
        Tensor::from_slice(&[0_i64], &[1, 1], client.device())?,
        Tensor::from_slice(&[0_i64], &[], client.device())?,
        Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
    ] {
        assert!(matches!(
            wet_contributions(&context, &segments, &rows),
            Err(error::HydraulicError::Numerical { .. })
        ));
        assert_eq!(work_before_rejection, work.snapshot());
        assert_eq!(before_source, components(&source_area)?);
        assert_eq!(before_target, components(&target_area)?);
    }
    let references = [
        cases[0].endpoints[1].increment,
        cases[1].endpoints[0].increment,
        cases[2].endpoints[0].increment,
    ];
    let mut singleton = Vec::new();
    for (face, reference) in references.iter().enumerate() {
        let rows = Tensor::from_slice(&[i64::try_from(face)?], &[1], client.device())?;
        let value = hydraulic(wet_contributions(&context, &segments, &rows))?;
        let physical = endpoints(&value.physical)?[0];
        check_reference(physical, *reference, "singleton physical");
        singleton.push(physical);
    }
    for layout in [&[2_i64, 0, 1][..], &[1, 2, 0], &[2, 0], &[1, 1, 0, 2]] {
        let rows = Tensor::from_slice(layout, &[layout.len()], client.device())?;
        let value = hydraulic(wet_contributions(&context, &segments, &rows))?;
        let physical = endpoints(&value.physical)?;
        let retained = hydraulic(error::checked::<R, _>(
            &client,
            "ordered potential test retained bounds",
            |a| Ok(a.expansion_interval(&value.retained)?),
        ))?;
        assert_eq!(physical.len(), layout.len());
        let retained = endpoints(&retained)?;
        for (position, face) in layout.iter().enumerate() {
            let face = usize::try_from(*face)?;
            let label = format!("layout={layout:?}, position={position}, face={face}");
            check_reference(physical[position], references[face], &label);
            check_reference(retained[position], references[face], &label);
            assert!(
                physical[position][0] <= singleton[face][1]
                    && singleton[face][0] <= physical[position][1],
                "{label}"
            );
        }
    }
    assert_eq!(before_source, components(&source_area)?);
    assert_eq!(before_target, components(&target_area)?);
    assert_eq!(work.snapshot().root_solves, 0);
    assert_eq!(work.snapshot().bracket_solves, 0);
    Ok(())
}

fn check_reference(actual: [f64; 2], reference: frozen::Reference, label: &str) {
    let expected = reference.bounds.map(f64::from_bits);
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: actual={actual:?}, independent={expected:?}"
    );
    let excess = (actual[0] - expected[1])
        .abs()
        .max((actual[1] - expected[0]).abs());
    assert!(
        excess <= f64::from_bits(reference.native_allowance),
        "{label}: excess={excess:e}, allowance={:e}",
        f64::from_bits(reference.native_allowance)
    );
}

#[test]
fn configured_cpu_potential_numerical_preserves_face_and_band_order() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    face_band_order::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_potential_numerical_preserves_face_and_band_order() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    face_band_order::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
