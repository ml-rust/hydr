use super::potential_absolute::{endpoint_potential, source_potential};
use super::potential_increment::endpoint_increment;
use super::potential_range::{PotentialContext, PotentialEvaluation, PotentialValue};
use super::source_tests::support::{
    self, Client, Result, components, endpoints, expansion, hydraulic, zero,
};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig, PrismaticRiemannRejection,
        equation_ops::{gravity_depth_root, scalar},
        error::{self, HydraulicError},
        work::WorkLedger,
    },
    section::{
        AreaAnchor, AreaDirection, AreaQuery, CrossSectionBatch, CrossSectionDefinition,
        CrossSectionPoint, HydraulicGeometry, PhysicalEndpoint, PrismaticRecoveryConfig,
        PrismaticSectionBatch,
    },
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

#[path = "potential_increment_fixtures/mod.rs"]
pub(super) mod frozen;

#[path = "potential_range_test_support.rs"]
mod acceptance;
#[path = "potential_film_tests.rs"]
mod film;
use acceptance::{check, check_value, config, section, target_check};

fn bindings<R>(client: &R::Client, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let section = section::<R>(client, frozen::query_profiles::PROFILES)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let config = config(frozen::query_profiles::GRAVITY_BITS);
    let table = hydraulic(error::checked::<R, _>(
        client,
        "frozen potential boundary table",
        |a| geometry.boundary_areas(a),
    ))?;
    for case in frozen::query_profiles::CASES {
        let case: &frozen::query_profiles::Case = case;
        let chart: &frozen::query_profiles::Chart =
            &frozen::query_profiles::QUERIES[case.query_index];
        assert_eq!(chart.id, case.query_id);
        // Repeated owners exercise independent owned rows. Input components retain their original strided storage.
        let owners = Tensor::from_slice(
            &[i64::try_from(chart.profile_index)?; 3],
            &[3],
            client.device(),
        )?;
        let source_area = expansion::<R>(
            client,
            &[[case.source_components[0], case.source_components[1], 0]; 3],
            strided,
        )?;
        let before = components(&source_area)?;
        let source = section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
        let anchor_area = expansion::<R>(
            client,
            &[[
                chart.anchor_source_components[0],
                chart.anchor_source_components[1],
                0,
            ]; 3],
            strided,
        )?;
        let anchor_source =
            section.recover_area_points_tensor(&owners, &anchor_area, &config.recovery)?;
        let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
            client,
            "frozen potential binding",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = table.gather(a, &owners)?;
                let scale = a.expansion_bounds(
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.scale_components[0]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.scale_components[1]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::zeros(&[3], DType::F64, client.device())?,
                )?;
                let parameter = a.bounds(
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.parameter_bounds[0]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.parameter_bounds[1]); 3],
                        &[3],
                        client.device(),
                    )?,
                )?;
                let direction = if chart.add {
                    AreaDirection::Add
                } else {
                    AreaDirection::Subtract
                };
                let query = if chart.anchor == frozen::query_profiles::AnchorKind::Source {
                AreaQuery::source_affine(a, &bands, &anchor_source, direction, &scale, &parameter)?
            } else {
                let indices = Tensor::from_slice(&[chart.band; 3], &[3], client.device())?;
                let anchor = match chart.anchor {
                    frozen::query_profiles::AnchorKind::Zero => AreaAnchor::Zero,
                    frozen::query_profiles::AnchorKind::BandStart => AreaAnchor::BandStart(indices),
                    frozen::query_profiles::AnchorKind::BandEnd => AreaAnchor::BandEnd(indices),
                    frozen::query_profiles::AnchorKind::Cap => AreaAnchor::Cap,
                    frozen::query_profiles::AnchorKind::Source => return Err(HydraulicError::Numerical {
                        context: "source chart reached a geometric-anchor branch. Check fixture dispatch".into() }),
                };
                AreaQuery::band_affine(a, &bands, anchor, direction, &scale, &parameter)?
            }.prove_support(a, &bands)?;
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
                panels: 1024,
                work: &work,
            },
        ))?;
        let source_value = hydraulic(source_potential(&context))?;
        check_value::<R>(client, &source_value, case.source_potential, case.id)?;
        let source_sound = hydraulic(error::checked::<R, _>(client, "frozen source sound", |a| {
            let g = scalar(a, config.gravity_m_per_s2)?;
            gravity_depth_root(a, &g, source.hydraulic_depth_m(), None)
        }))?;
        for actual in endpoints(&source_sound)? {
            check(actual, case.source_sound, case.id);
        }
        let state = hydraulic(error::checked::<R, _>(
            client,
            "frozen potential range geometry",
            |a| geometry.recover_query(a, &bands, &query),
        ))?
        .state;
        assert_eq!(
            state.wet_possible.try_to_vec::<f64>()?,
            [if case.wet_possible { 1.0 } else { 0.0 }; 3]
        );
        assert_eq!(
            state.dry_possible.try_to_vec::<f64>()?,
            [if case.dry_possible { 1.0 } else { 0.0 }; 3]
        );
        for (endpoint, index) in [
            (PhysicalEndpoint::Lower, case.minimum_endpoint),
            (PhysicalEndpoint::Upper, case.maximum_endpoint),
        ] {
            let value = hydraulic(endpoint_potential(&context, endpoint))?;
            let increment = hydraulic(endpoint_increment(&context, endpoint))?;
            target_check::<R>(client, &value, &increment, &case.endpoints[index], case.id)?;
            if case.query_index == 14 && matches!(endpoint, PhysicalEndpoint::Lower) {
                acceptance::guarded_prefix::<R>(
                    client,
                    &context,
                    endpoint,
                    case.endpoints[index].potential,
                )?;
            }
            let sound = hydraulic(error::checked::<R, _>(
                client,
                "frozen endpoint sound",
                |a| {
                    let target = query.endpoint(a, &bands, endpoint)?;
                    let state = geometry.recover_query(a, &bands, &target)?.state;
                    let g = scalar(a, config.gravity_m_per_s2)?;
                    gravity_depth_root(a, &g, &state.hydraulic_depth, None)
                },
            ))?;
            for actual in endpoints(&sound)? {
                check(actual, case.endpoints[index].sound, case.id);
            }
            if let Some(analytic) = &value.analytic {
                let eligible = analytic.eligible.try_to_vec::<f64>()?;
                let projected = support::bounds::<R>(client, &analytic.sound)?;
                for (row, actual) in endpoints(&projected)?.into_iter().enumerate() {
                    if eligible[row] != 0.0 {
                        check(actual, case.endpoints[index].sound, case.id);
                    }
                }
            }
        }
        assert_eq!(
            before,
            components(&source_area)?,
            "{}: input mutation",
            case.id
        );
        assert_eq!(work.snapshot().root_solves, 0);
        assert_eq!(work.snapshot().bracket_solves, 0);
        assert_eq!(work.snapshot().refinements, 0);
    }
    Ok(())
}

fn lazy_controls<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let section = section::<R>(client, frozen::query_profiles::PROFILES)?;
    let foreign = acceptance::section::<R>(client, frozen::query_profiles::PROFILES)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let other_geometry = HydraulicGeometry::from_batch(&foreign);
    let owners = Tensor::from_slice(&[3_i64], &[1], client.device())?;
    let area = expansion::<R>(client, &[[1.0_f64.to_bits(), 0, 0]], true)?;
    let source =
        section.recover_area_points_tensor(&owners, &area, &PrismaticRecoveryConfig::default())?;
    let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
        client,
        "equal potential control",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
            let scale = a.expansion_point(&zero)?;
            let parameter = a.point(&zero)?;
            let query = AreaQuery::source_affine(
                a,
                &bands,
                &source,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            Ok((bands, boundaries, query))
        },
    ))?;
    let mut config = config(frozen::query_profiles::GRAVITY_BITS);
    config.max_native_query_points = 0;
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
    let increment = hydraulic(endpoint_increment(&context, PhysicalEndpoint::Lower))?;
    zero(
        components(&increment.retained)?[0],
        "equal nonanalytic potential",
    );
    assert_eq!(endpoints(&increment.physical)?, [[0.0; 2]]);
    assert_eq!(work.snapshot(), Default::default());
    assert!(matches!(
        source_potential(&context),
        Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            ..
        })
    ));
    assert_eq!(work.snapshot(), Default::default());
    assert!(matches!(
        PotentialContext::new(
            &other_geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: &config,
                panels: 4,
                work: &work
            }
        ),
        Err(HydraulicError::Geometry(_))
    ));
    Ok(())
}

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    for strided in [false, true] {
        bindings::<R>(&client, strided)?;
    }
    lazy_controls::<R>(&client)?;
    film::run::<R>(&client)
}

#[test]
fn configured_cpu_potential_ranges_preserve_independent_leaves_and_film_invariants() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    matrix::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_potential_ranges_preserve_independent_leaves_and_film_invariants() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
