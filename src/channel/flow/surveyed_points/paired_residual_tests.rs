use super::PairedSources;
use crate::channel::flow::surveyed_points::{
    potential_range::PotentialEvaluation,
    source_tests::support::{self, Client, Result, bounds, components, endpoints, hydraulic},
    wave_curve_types::CurveValue,
};
use crate::channel::{
    flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger},
    section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry, PhysicalEndpoint},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::collections::BTreeMap;

#[path = "paired_residual_test_support.rs"]
mod capture;
#[path = "paired_residual_fixtures.rs"]
mod frozen;
#[path = "paired_source_invariant_tests.rs"]
mod invariant;

fn check<R>(
    client: &R::Client,
    value: &CurveValue<R>,
    reference: &frozen::Reference,
    label: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert!(
        value
            .available
            .try_to_vec::<f64>()?
            .iter()
            .all(|&mask| mask == 1.0),
        "{label}"
    );
    for interval in [&value.physical, &bounds::<R>(client, &value.retained)?] {
        for actual in endpoints(interval)? {
            capture::reference(actual, reference.bounds, reference.allowance, label);
            match reference.sign {
                frozen::Sign::Negative => assert!(actual[1] < 0.0, "{label}: {actual:?}"),
                frozen::Sign::Positive => assert!(actual[0] > 0.0, "{label}: {actual:?}"),
                frozen::Sign::ContainsZero => {
                    assert!(actual[0] <= 0.0 && actual[1] >= 0.0, "{label}: {actual:?}")
                }
            }
        }
    }
    Ok(())
}

fn check_expected<R>(
    client: &R::Client,
    value: &CurveValue<R>,
    expected: &frozen::Expected,
    upper: bool,
    label: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    match expected {
        frozen::Expected::Unavailable => {
            assert!(
                value
                    .available
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&mask| mask == 0.0),
                "{label}"
            );
        }
        frozen::Expected::ExactZero => {
            assert!(
                value
                    .available
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&mask| mask == 1.0),
                "{label}"
            );
            for point in components(&value.retained)? {
                assert_eq!(point, [0; 3], "{label}");
            }
            for physical in endpoints(&value.physical)? {
                assert_eq!(physical, [0.0; 2], "{label}");
            }
        }
        frozen::Expected::Point(reference) => check::<R>(client, value, reference, label)?,
        frozen::Expected::Range { lower, upper: high } => {
            check::<R>(client, value, if upper { high } else { lower }, label)?
        }
    }
    Ok(())
}

fn chart<R>(
    client: &R::Client,
    bands: &crate::channel::section::GatheredBands<R>,
    source: &crate::channel::section::PrismaticAreaPointRecovery<R>,
    request: &frozen::Request,
) -> error::HydraulicResult<crate::channel::section::SupportedAreaQuery<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    error::checked::<R, _>(client, "paired frozen residual chart", |a| {
        let count = bands.section_owner.numel();
        let zero = Tensor::zeros(&[count], a.dtype(), client.device())?;
        let scale = a.expansion_bounds(
            &Tensor::from_slice(
                &vec![f64::from_bits(request.query.scale[0]); count],
                &[count],
                client.device(),
            )?,
            &Tensor::from_slice(
                &vec![f64::from_bits(request.query.scale[1]); count],
                &[count],
                client.device(),
            )?,
            &zero,
        )?;
        let parameter = a.bounds(
            &Tensor::from_slice(
                &vec![f64::from_bits(request.query.parameter[0]); count],
                &[count],
                client.device(),
            )?,
            &Tensor::from_slice(
                &vec![f64::from_bits(request.query.parameter[1]); count],
                &[count],
                client.device(),
            )?,
        )?;
        let direction = if request.query.add {
            AreaDirection::Add
        } else {
            AreaDirection::Subtract
        };
        let query = match request.query.anchor {
            frozen::Anchor::LeftSource => {
                AreaQuery::source_affine(a, bands, source, direction, &scale, &parameter)?
            }
            frozen::Anchor::Zero => {
                AreaQuery::band_affine(a, bands, AreaAnchor::Zero, direction, &scale, &parameter)?
            }
        };
        query.prove_support(a, bands)
    })
}

fn matrix<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    assert_eq!(frozen::REQUESTS.len(), 13);
    for partition in [
        frozen::Partition::Local,
        frozen::Partition::Below,
        frozen::Partition::Exact,
        frozen::Partition::Above,
        frozen::Partition::Crossing,
    ] {
        assert!(
            frozen::REQUESTS
                .iter()
                .any(|request| request.partition == partition)
        );
    }
    let source_cases = support::cases();
    let mut groups = BTreeMap::new();
    for request in frozen::REQUESTS {
        let source_index = source_cases
            .iter()
            .position(|case| case.id == request.source_case_id)
            .ok_or("Paired request source is absent from original70 corpus")?;
        groups
            .entry((source_index, request.gravity_bits))
            .or_insert_with(Vec::new)
            .push(request);
    }
    let mut request_count = 0;
    for strided in [false, true] {
        let corpus = capture::corpus::<R>(client, strided)?;
        let corpus_before = capture::snapshot(&corpus.velocity)?;
        assert_eq!(
            corpus.owners.contiguous()?.try_to_vec::<i64>()?,
            corpus.velocity.input().owners.try_to_vec::<i64>()?
        );
        // The bound source context owns conserved operands independently from the original caller tensors.
        for value in corpus
            .inputs
            .iter()
            .flat_map(|value| [value.high(), value.low(), value.abs_error()])
        {
            client.copy_into(
                value,
                &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
            )?;
        }
        capture::cache_controls::<R>(client, &corpus)?;
        invariant::controls::<R>(client, &corpus)?;
        assert_eq!(capture::snapshot(&corpus.velocity)?, corpus_before);
        for (&(source_index, gravity_bits), requests) in &groups {
            let rows =
                Tensor::from_slice(&[i64::try_from(source_index)?; 3], &[3], client.device())?;
            let velocity = hydraulic(error::checked::<R, _>(
                client,
                "paired original source binding",
                |a| corpus.velocity.select(a, &rows),
            ))?;
            let source_before = capture::snapshot(&velocity)?;
            let geometry = HydraulicGeometry::from_batch(&corpus.section);
            let input = velocity.input();
            let config = PrismaticRiemannConfig {
                gravity_m_per_s2: f64::from_bits(gravity_bits),
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
            let (bands, boundaries) = hydraulic(error::checked::<R, _>(
                client,
                "paired original geometry",
                |a| {
                    Ok((
                        geometry.gather(a, &input.owners)?,
                        geometry.boundary_areas(a)?.gather(a, &input.owners)?,
                    ))
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
            if requests
                .iter()
                .any(|request| request.id == "film_dry_target")
            {
                let invariant = hydraulic(pair.source_invariant())?;
                assert_eq!(invariant.value.available.try_to_vec::<f64>()?, [1.0; 3]);
                assert_eq!(
                    components(&invariant.value.retained)?,
                    [[
                        frozen::FILM_SOURCE_INVARIANT[0],
                        frozen::FILM_SOURCE_INVARIANT[1],
                        0
                    ]; 3]
                );
                assert_eq!(invariant.vacuum.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(invariant.touching.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(invariant.wet_candidate.try_to_vec::<f64>()?, [1.0; 3]);
                assert_eq!(invariant.unresolved.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(invariant.both_dry.try_to_vec::<f64>()?, [0.0; 3]);
            }
            for request in requests {
                request_count += 1;
                let query = hydraulic(chart::<R>(client, &bands, &left, request))?;
                for (endpoint, upper) in [
                    (PhysicalEndpoint::Lower, false),
                    (PhysicalEndpoint::Upper, true),
                ] {
                    let value = hydraulic(pair.endpoint(&query, endpoint))?;
                    check_expected::<R>(client, &value, &request.expected, upper, request.id)?;
                }
                let range = hydraulic(pair.range(&query))?;
                check_expected::<R>(client, &range.lower, &request.expected, false, request.id)?;
                check_expected::<R>(client, &range.upper, &request.expected, true, request.id)?;
                if let frozen::Expected::Range { lower, upper } = &request.expected {
                    for interval in [
                        &range.curve.physical,
                        &bounds::<R>(client, &range.curve.retained)?,
                    ] {
                        for actual in endpoints(interval)? {
                            let low = lower.bounds.map(f64::from_bits);
                            let high = upper.bounds.map(f64::from_bits);
                            assert!(
                                actual[0] <= low[0] && high[1] <= actual[1],
                                "{}: {actual:?}",
                                request.id
                            );
                            assert!(
                                low[1] - actual[0] <= f64::from_bits(lower.allowance),
                                "{}: lower range gate",
                                request.id
                            );
                            assert!(
                                actual[1] - high[0] <= f64::from_bits(upper.allowance),
                                "{}: upper range gate",
                                request.id
                            );
                        }
                    }
                } else {
                    check_expected::<R>(
                        client,
                        &range.curve,
                        &request.expected,
                        false,
                        request.id,
                    )?;
                }
                let row_storage = Tensor::from_slice(
                    &[99_i64, 2, 99, 0, 99, 2, 99, 1],
                    &[4, 2],
                    client.device(),
                )?;
                let rows = row_storage.narrow(1, 1, 1)?.squeeze(Some(1));
                let selected = hydraulic(error::checked::<R, _>(
                    client,
                    "paired callback query selection",
                    |a| query.select(a, &rows),
                ))?;
                assert_eq!(
                    selected.query.original_rows.try_to_vec::<i64>()?,
                    [2, 0, 2, 1]
                );
                let selected_value =
                    hydraulic(pair.endpoint_on_rows(&rows, &selected, PhysicalEndpoint::Lower))?;
                check_expected::<R>(
                    client,
                    &selected_value,
                    &request.expected,
                    false,
                    request.id,
                )?;
                let selected_range = hydraulic(pair.range_on_rows(&rows, &selected))?;
                check_expected::<R>(
                    client,
                    &selected_range.lower,
                    &request.expected,
                    false,
                    request.id,
                )?;
                check_expected::<R>(
                    client,
                    &selected_range.upper,
                    &request.expected,
                    true,
                    request.id,
                )?;
                assert_eq!(capture::snapshot(&velocity)?, source_before);
            }
            assert_eq!(work.snapshot().root_solves, 0);
            assert_eq!(work.snapshot().bracket_solves, 0);
            assert_eq!(work.snapshot().refinements, 0);
        }
        assert_eq!(capture::snapshot(&corpus.velocity)?, corpus_before);
    }
    assert_eq!(request_count, 26);
    Ok(())
}

#[test]
fn configured_cpu_paired_residuals_preserve_frozen_ranges_and_source_caches() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    matrix::<CpuRuntime>(&client)?;
    assert_eq!(client.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_paired_residuals_preserve_frozen_ranges_and_source_caches() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
}
