use super::super::PairedSources;
use crate::channel::flow::surveyed_points::potential_range::PotentialEvaluation;
use crate::channel::flow::surveyed_points::{
    analytic_bottom_tests::fixtures as source_references,
    source_tests::support::{self, Client, Result, components, expansion, hydraulic},
    source_velocity::{self, SourceVelocity},
};
use crate::channel::section::PrismaticSectionBatch;
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry, PhysicalEndpoint},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(super) fn source_reference(id: &str) -> Result<&source_references::Case> {
    source_references::CASES
        .iter()
        .find(|case| case.id == id)
        .ok_or_else(|| "Missing original source potential reference".into())
}

pub(super) struct Corpus<R: Runtime<DType = DType>> {
    pub section: PrismaticSectionBatch<R>,
    pub velocity: SourceVelocity<R>,
    pub inputs: Vec<CertifiedExpansion<R>>,
    pub owners: Tensor<R>,
}

pub(super) fn corpus<R>(client: &R::Client, strided: bool) -> Result<Corpus<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let cases = support::cases();
    assert_eq!(cases.len(), 70);
    let section = support::batch::<R>(client)?;
    let owner_values = cases
        .iter()
        .map(|case| i64::try_from(case.owner))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let owners = if strided {
        let storage: Vec<_> = owner_values
            .iter()
            .flat_map(|&owner| [i64::MAX, owner])
            .collect();
        let view = Tensor::from_slice(&storage, &[cases.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(!view.is_contiguous() && view.offset() > 0);
        view
    } else {
        Tensor::from_slice(&owner_values, &[cases.len()], client.device())?
    };
    let inputs = (0..4)
        .map(|field| {
            expansion::<R>(
                client,
                &cases
                    .iter()
                    .map(|case| case.inputs[field])
                    .collect::<Vec<_>>(),
                strided,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let captured = hydraulic(
        section.capture_riemann_points(&owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3]),
    )?;
    let velocity = hydraulic(source_velocity::prepare::<R>(client, &captured))?;
    assert!(std::ptr::eq(velocity.input(), captured.as_ref()));
    Ok(Corpus {
        section,
        velocity,
        inputs,
        owners,
    })
}

#[derive(Debug, PartialEq)]
pub(super) struct Snapshot {
    owners: Vec<i64>,
    components: Vec<Vec<[u64; 3]>>,
    masks: Vec<Vec<u64>>,
}

pub(super) fn snapshot<R: Runtime<DType = DType>>(
    velocity: &SourceVelocity<R>,
) -> Result<Snapshot> {
    let input = velocity.input();
    Ok(Snapshot {
        owners: input.owners.try_to_vec::<i64>()?,
        components: [
            &input.left_area,
            &input.left_discharge,
            &input.right_area,
            &input.right_discharge,
            &velocity.left,
            &velocity.right,
            &velocity.difference,
        ]
        .into_iter()
        .map(components)
        .collect::<Result<_>>()?,
        masks: [
            &input.left_dry,
            &input.right_dry,
            &velocity.both_wet,
            &velocity.uniform,
        ]
        .into_iter()
        .map(|mask| {
            Ok(mask
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect())
        })
        .collect::<Result<_>>()?,
    })
}

pub(super) fn reference(actual: [f64; 2], bounds: [u64; 2], allowance: u64, label: &str) {
    let expected = bounds.map(f64::from_bits);
    let allowance = f64::from_bits(allowance);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
    let excess = (expected[1] - actual[0]).max(actual[1] - expected[0]);
    assert!(
        excess <= allowance,
        "{label}: excess={excess:e}, allowance={allowance:e}, {actual:?}"
    );
}

#[derive(Debug, PartialEq)]
struct CacheSnapshot {
    ready: Vec<u64>,
    retained: Vec<[u64; 3]>,
    physical: Vec<[u64; 2]>,
}

fn cache_snapshot<R: Runtime<DType = DType>>(
    pair: &PairedSources<'_, R>,
) -> Result<Vec<CacheSnapshot>> {
    pair.cache
        .iter()
        .map(|cache| {
            Ok(CacheSnapshot {
                ready: cache
                    .ready
                    .try_to_vec::<f64>()?
                    .into_iter()
                    .map(f64::to_bits)
                    .collect(),
                retained: components(&cache.value.retained)?,
                physical: support::endpoints(&cache.value.physical)?
                    .into_iter()
                    .map(|value| value.map(f64::to_bits))
                    .collect(),
            })
        })
        .collect()
}

pub(super) fn cache_controls<R>(client: &R::Client, corpus: &Corpus<R>) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let truth: &source_references::Case = source_references::CASES
        .iter()
        .find(|case| case.id == "scalar_three_bands_uniform_upper")
        .ok_or("Missing independent multiband source potential")?;
    assert!(truth.sides.iter().all(
        |side: &source_references::SideReference| !side.physical_eligible
            && !side.within_first_band
            && !side.dry
    ));
    let source_rows = Tensor::from_slice(
        &[i64::try_from(truth.source_index)?; 3],
        &[3],
        client.device(),
    )?;
    let velocity = hydraulic(error::checked::<R, _>(
        client,
        "cached multiband source binding",
        |a| corpus.velocity.select(a, &source_rows),
    ))?;
    let geometry = HydraulicGeometry::from_batch(&corpus.section);
    let input = velocity.input();
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(truth.gravity_bits),
        ..PrismaticRiemannConfig::default()
    };
    let left = corpus.section.recover_area_points_tensor(
        &input.owners,
        &input.left_area,
        &config.recovery,
    )?;
    let right = corpus.section.recover_area_points_tensor(
        &input.owners,
        &input.right_area,
        &config.recovery,
    )?;
    let (bands, boundaries, dry, equal) = hydraulic(error::checked::<R, _>(
        client,
        "cached multiband source graph",
        |a| {
            let bands = geometry.gather(a, &input.owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &input.owners)?;
            let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
            let scale = a.expansion_point(&zero)?;
            let parameter = a.point(&zero)?;
            let dry = AreaQuery::band_affine(
                a,
                &bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &bands)?;
            let equal =
                AreaQuery::source_affine(a, &bands, &left, AreaDirection::Add, &scale, &parameter)?
                    .prove_support(a, &bands)?;
            Ok((bands, boundaries, dry, equal))
        },
    ))?;
    let originals = Tensor::from_slice(&[0_i64, 1, 2], &[3], client.device())?;
    let work = WorkLedger::new(&config);
    let mut pair = hydraulic(PairedSources::new(
        &geometry,
        &bands,
        [&left, &right],
        &velocity,
        &boundaries,
        PotentialEvaluation {
            config: &config,
            panels: 1024,
            work: &work,
        },
        &originals,
    ))?;
    assert_eq!(work.snapshot(), Default::default());
    for cache in &pair.cache {
        assert_eq!(cache.ready.try_to_vec::<f64>()?, [0.0; 3]);
    }
    let storage = Tensor::from_slice(&[99_i64, 2, 99, 0, 99, 2], &[3, 2], client.device())?;
    let rows = storage.narrow(1, 1, 1)?.squeeze(Some(1));
    let selected = hydraulic(error::checked::<R, _>(
        client,
        "cached compact dry query",
        |a| dry.select(a, &rows),
    ))?;
    let value = hydraulic(pair.endpoint_on_rows(&rows, &selected, PhysicalEndpoint::Lower))?;
    assert_eq!(value.available.try_to_vec::<f64>()?, [1.0; 3]);
    assert!(
        value
            .physical
            .upper()
            .try_to_vec::<f64>()?
            .iter()
            .all(|&upper| upper < 0.0)
    );
    for (side, cache) in pair.cache.iter().enumerate() {
        let phi: &source_references::Reference = &truth.sides[side].phi;
        assert_eq!(cache.ready.try_to_vec::<f64>()?, [1.0, 0.0, 1.0]);
        for interval in [
            &cache.value.physical,
            &support::bounds::<R>(client, &cache.value.retained)?,
        ] {
            let values = support::endpoints(interval)?;
            for row in [0, 2] {
                reference(values[row], phi.bounds, phi.native_tolerance, truth.id);
            }
        }
    }
    let partial = cache_snapshot(&pair)?;
    let partial_work = work.snapshot();
    assert!(partial_work.simpson_value_points > 0 && partial_work.simpson_derivative_points > 0);
    let _ = hydraulic(pair.endpoint_on_rows(&rows, &selected, PhysicalEndpoint::Upper))?;
    assert_eq!(cache_snapshot(&pair)?, partial);
    assert_eq!(work.snapshot(), partial_work);
    let _ = hydraulic(pair.source_invariant())?;
    for (side, cache) in pair.cache.iter().enumerate() {
        assert_eq!(cache.ready.try_to_vec::<f64>()?, [1.0; 3]);
        for interval in [
            &cache.value.physical,
            &support::bounds::<R>(client, &cache.value.retained)?,
        ] {
            for value in support::endpoints(interval)? {
                reference(
                    value,
                    truth.sides[side].phi.bounds,
                    truth.sides[side].phi.native_tolerance,
                    truth.id,
                );
            }
        }
    }
    let complete = cache_snapshot(&pair)?;
    let completed_work = work.snapshot();
    assert!(completed_work.simpson_value_points > partial_work.simpson_value_points);
    let _ = hydraulic(pair.source_invariant())?;
    let identity = hydraulic(pair.endpoint(&equal, PhysicalEndpoint::Lower))?;
    assert_eq!(components(&identity.retained)?, [[0; 3]; 3]);
    assert_eq!(cache_snapshot(&pair)?, complete);
    assert_eq!(work.snapshot(), completed_work);
    let wrong_rows = Tensor::from_slice(&[1_i64, 0, 2], &[3], client.device())?;
    let wrong_originals = hydraulic(error::checked::<R, _>(
        client,
        "misordered paired original rows",
        |a| dry.select(a, &wrong_rows),
    ))?;
    assert!(
        pair.endpoint(&wrong_originals, PhysicalEndpoint::Lower)
            .is_err()
    );
    assert_eq!(cache_snapshot(&pair)?, complete);
    assert_eq!(work.snapshot(), completed_work);
    let foreign_section = support::batch::<R>(client)?;
    let foreign_geometry = HydraulicGeometry::from_batch(&foreign_section);
    let foreign_query = hydraulic(error::checked::<R, _>(
        client,
        "foreign paired query graph",
        |a| {
            let foreign_bands = foreign_geometry.gather(a, &input.owners)?;
            let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
            let scale = a.expansion_point(&zero)?;
            let parameter = a.point(&zero)?;
            AreaQuery::band_affine(
                a,
                &foreign_bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &foreign_bands)
        },
    ))?;
    let mut untouched = hydraulic(PairedSources::new(
        &geometry,
        &bands,
        [&left, &right],
        &velocity,
        &boundaries,
        PotentialEvaluation {
            config: &config,
            panels: 1024,
            work: &work,
        },
        &originals,
    ))?;
    let unfilled = cache_snapshot(&untouched)?;
    assert!(
        untouched
            .endpoint(&wrong_originals, PhysicalEndpoint::Lower)
            .is_err()
    );
    assert!(
        untouched
            .endpoint(&foreign_query, PhysicalEndpoint::Lower)
            .is_err()
    );
    assert!(
        pair.endpoint(&foreign_query, PhysicalEndpoint::Lower)
            .is_err()
    );
    let other_owners = Tensor::from_slice(&[1_i64; 3], &[3], client.device())?;
    assert_ne!(input.owners.try_to_vec::<i64>()?, [1_i64; 3]);
    let other_query = hydraulic(error::checked::<R, _>(
        client,
        "different paired query owners",
        |a| {
            let other_bands = geometry.gather(a, &other_owners)?;
            let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
            let scale = a.expansion_point(&zero)?;
            let parameter = a.point(&zero)?;
            AreaQuery::band_affine(
                a,
                &other_bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &scale,
                &parameter,
            )?
            .prove_support(a, &other_bands)
        },
    ))?;
    assert!(
        untouched
            .endpoint(&other_query, PhysicalEndpoint::Lower)
            .is_err()
    );
    assert_eq!(cache_snapshot(&untouched)?, unfilled);
    assert!(
        pair.endpoint(&other_query, PhysicalEndpoint::Lower)
            .is_err()
    );
    assert_eq!(cache_snapshot(&pair)?, complete);
    assert_eq!(work.snapshot(), completed_work);
    assert_eq!(work.snapshot().root_solves, 0);
    assert_eq!(work.snapshot().bracket_solves, 0);
    assert_eq!(work.snapshot().refinements, 0);
    Ok(())
}
