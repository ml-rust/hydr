use super::corpus_coverage::Coverage;
use super::data::{CASE_GROUPS, Case};
use super::exact_width;
use super::support::{self, Client, Result};
use hydr::channel::{PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannResult};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn reflected_discharge(actual: [f64; 2], frozen: [u64; 2], label: &str) {
    let frozen = frozen.map(f64::from_bits);
    assert!(
        frozen[0] <= 0.0 && frozen[1] >= 0.0,
        "{label}: frozen central discharge excludes the exact reflection value: {frozen:?}"
    );
    assert_eq!(actual, [0.0, 0.0], "{label}: exact reflected discharge");
}

pub(super) fn check<R: Runtime<DType = DType>>(
    result: &PrismaticRiemannResult<R>,
    cases: &[&Case],
    config: &PrismaticRiemannConfig,
) -> Result {
    support::snapshots(result, cases)?;
    support::work(result.work(), config)?;
    let n = cases.len();
    let water = support::bounds(result.water_flux_m3_per_s(), n)?;
    let momentum = support::bounds(result.momentum_flux_m4_per_s2(), n)?;
    let area = support::bounds(result.trace_area_m2(), n)?;
    let discharge = support::bounds(result.trace_discharge_m3_per_s(), n)?;
    let depth = support::bounds(result.trace_depth_m(), n)?;
    let waves = support::bounds(result.wave_speed_m_per_s(), n)?;
    for mask in [result.trace_dry_possible(), result.trace_wet_possible()] {
        assert_eq!(mask.shape(), &[n]);
        assert_eq!(mask.dtype(), DType::F64);
    }
    let dry = result.trace_dry_possible().try_to_vec::<f64>()?;
    let wet = result.trace_wet_possible().try_to_vec::<f64>()?;
    for (row, case) in cases.iter().enumerate() {
        let [left_area, left_discharge, right_area, right_discharge] =
            case.input.map(f64::from_bits);
        let reflected =
            left_area > 0.0 && left_area == right_area && left_discharge == -right_discharge;
        for (field, (actual, truth, atol)) in [
            (water[row], case.flux[0], 1e-9f64),
            (momentum[row], case.flux[1], 1e-8f64),
        ]
        .into_iter()
        .enumerate()
        {
            if reflected && field == 0 {
                reflected_discharge(actual, truth, case.id);
            } else {
                support::contains(actual, truth, case.id);
            }
            assert!(
                exact_width::accepted(actual[0], actual[1], atol, 1e-10)?,
                "{}: flux width exceeds the original F64 allowance: {actual:?}",
                case.id
            );
        }
        for candidate in case.candidates {
            support::contains(area[row], candidate.area, case.id);
            // These fixture candidates describe the central sampled trace, where exact reflection proves Q=0.
            if reflected {
                reflected_discharge(discharge[row], candidate.discharge, case.id);
            } else {
                support::contains(discharge[row], candidate.discharge, case.id);
            }
            support::contains(depth[row], candidate.depth, case.id);
        }
        for edge in case.edges {
            support::contains(waves[row], *edge, case.id);
        }
        assert_eq!(
            dry[row],
            f64::from(u8::from(case.dry)),
            "{} dry mask",
            case.id
        );
        assert_eq!(
            wet[row],
            f64::from(u8::from(case.wet)),
            "{} wet mask",
            case.id
        );
    }
    Ok(())
}

fn isolated<R>(client: &R::Client, strided: bool, coverage: Coverage) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let mut required_successes = 0;
    let mut conditional_successes = 0;
    let mut conditional_rejections = 0;
    let mut required_rejections = 0;
    let replayed = coverage.select(CASE_GROUPS.iter().flat_map(|group| group.iter()), |case| {
        (
            case.rejection.is_some(),
            !case.allowed.is_empty(),
            case.zero_work,
            case.dry,
            case.wet,
        )
    });
    for case in replayed {
        let config = support::config(case);
        let [owners, la, lq, ra, rq] = support::inputs::<R>(client, &[case], strided)?;
        let result = batch.riemann_flux_tensor(&owners, &la, &lq, &ra, &rq, &config);
        match result {
            Ok(result) => {
                assert!(
                    case.rejection.is_none(),
                    "{}: required rejection accepted",
                    case.id
                );
                check(&result, &[case], &config)?;
                if case.allowed.is_empty() {
                    required_successes += 1;
                } else {
                    conditional_successes += 1;
                }
            }
            Err(PrismaticRiemannError::Rejected {
                reason,
                query,
                section_owner,
                work,
                context,
            }) => {
                support::work(*work, &config)?;
                assert!(!context.is_empty(), "{}: rejection lacks context", case.id);
                if let Some(query) = query {
                    assert_eq!(query, 0, "{}", case.id);
                }
                if let Some(owner) = section_owner {
                    assert_eq!(owner, case.profile, "{}", case.id);
                }
                if let Some(expected) = case.rejection {
                    assert_eq!(reason, expected, "{}: {context}", case.id);
                    if !case.zero_work {
                        assert_eq!(query, Some(0), "{}: {context}", case.id);
                        assert_eq!(section_owner, Some(case.profile), "{}: {context}", case.id);
                    }
                    required_rejections += 1;
                    if case.zero_work {
                        assert_eq!(work.residual_points, 0);
                        assert_eq!(work.simpson_value_points, 0);
                        assert_eq!(work.simpson_derivative_points, 0);
                    }
                } else {
                    assert!(
                        case.allowed.contains(&reason),
                        "{}: unexpected {reason:?}: {context}",
                        case.id
                    );
                    conditional_rejections += 1;
                    eprintln!(
                        "conditional shallow-film rejection: {} {reason:?}: {context}",
                        case.id
                    );
                }
            }
            Err(error) => return Err(format!("{}: {error}", case.id).into()),
        }
    }
    if coverage.is_full() {
        assert_eq!(required_successes, 34);
        assert_eq!(required_rejections, 2);
        assert_eq!(conditional_successes + conditional_rejections, 1);
    }
    eprintln!(
        "required successes={required_successes}, conditional successes={conditional_successes}, conditional rejections={conditional_rejections}, required rejections={required_rejections}"
    );
    Ok(())
}

fn mixed_compaction<R>(client: &R::Client, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let eligible: Vec<_> = CASE_GROUPS
        .iter()
        .flat_map(|group| group.iter())
        .filter(|case| {
            case.rejection.is_none()
                && case.allowed.is_empty()
                && case.gravity == 1.0f64.to_bits()
                && !case.loose_recovery
        })
        .collect();
    assert!(eligible.len() > 10);
    assert!(eligible.len() <= 33);
    let cases: Vec<_> = (0..33)
        .map(|i| eligible[eligible.len() - 1 - i % eligible.len()])
        .collect();
    assert!(cases.iter().any(|case| case.dry));
    assert!(cases.iter().any(|case| case.wet));
    assert!(cases.iter().any(|case| case.id.contains("vacuum")));
    assert!(
        cases
            .windows(2)
            .any(|pair| pair[0].profile > pair[1].profile)
    );
    let [owners, la, lq, ra, rq] = support::inputs::<R>(client, &cases, strided)?;
    let config = support::config(cases[0]);
    let result = batch.riemann_flux_tensor(&owners, &la, &lq, &ra, &rq, &config)?;
    check(&result, &cases, &config)?;
    for (source, snapshot) in [
        (&owners, result.owners()),
        (&la, result.left_area_m2()),
        (&lq, result.left_discharge_m3_per_s()),
        (&ra, result.right_area_m2()),
        (&rq, result.right_discharge_m3_per_s()),
    ] {
        assert_ne!(source.storage().ptr(), snapshot.storage().ptr());
        client.copy_into(
            source,
            &Tensor::<R>::zeros(source.shape(), source.dtype(), client.device())?,
        )?;
    }
    check(&result, &cases, &config)?;
    let fresh = support::inputs::<R>(client, &cases, strided)?;
    let again = batch.riemann_flux_tensor(
        &fresh[0], &fresh[1], &fresh[2], &fresh[3], &fresh[4], &config,
    )?;
    check(&again, &cases, &config)?;
    assert_eq!(again.work(), result.work());
    Ok(())
}

pub(super) fn run<R>(client: R::Client, strided: bool) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    // Strided storage changes only the input layout, so one case per class reaches it.
    let coverage = if strided {
        Coverage::Sample
    } else {
        Coverage::Full
    };
    isolated::<R>(&client, strided, coverage)?;
    mixed_compaction::<R>(&client, strided)
}

fn configured_cpu(strided: bool) -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let original_config = original.parallelism();
    let client = original.with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    let before = client.parallelism();
    run::<CpuRuntime>(client.clone(), strided)?;
    assert_eq!(client.parallelism(), before);
    assert_eq!(original.parallelism(), original_config);
    Ok(())
}

#[test]
fn configured_cpu_encloses_all_frozen_waves() -> Result {
    configured_cpu(false)
}

#[test]
fn configured_cpu_encloses_all_frozen_waves_from_strided_views() -> Result {
    configured_cpu(true)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_encloses_all_frozen_waves() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    isolated::<CudaRuntime>(&client, false, Coverage::Sample)?;
    mixed_compaction::<CudaRuntime>(&client, true)
}
