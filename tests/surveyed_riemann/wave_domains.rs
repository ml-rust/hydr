use super::data::PROFILES;
use super::support::{self, Result as TestResult};
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint, PrismaticRecoveryConfig,
    PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection, PrismaticRiemannWork,
};
use numr::dtype::{DType, Element};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::optimize::CertifiedBisectionOptions;

fn values<R, T>(client: &R::Client, input: &[T], strided: bool) -> TestResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    T: Element + Copy,
{
    if strided {
        let storage: Vec<T> = input
            .iter()
            .flat_map(|&value| [T::from_f64(-1.0), value])
            .collect();
        let result = Tensor::from_slice(&storage, &[input.len(), 2], client.device())?
            .narrow(1, 1, 1)?
            .squeeze(Some(1));
        assert!(result.offset() > 0);
        assert!(!result.is_contiguous());
        Ok(result)
    } else {
        Ok(Tensor::from_slice(input, &[input.len()], client.device())?)
    }
}

fn finite<R: Runtime<DType = DType>>(interval: &Interval<R>) -> TestResult {
    let lower = interval.lower().try_to_vec::<f64>()?;
    let upper = interval.upper().try_to_vec::<f64>()?;
    assert_eq!(lower.len(), 3);
    assert_eq!(upper.len(), 3);
    for (&lower, &upper) in lower.iter().zip(&upper) {
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
    }
    Ok(())
}

fn profile(id: &str, points: &[[u64; 2]], cap: u64) -> CrossSectionDefinition {
    CrossSectionDefinition {
        id: id.into(),
        vertical_datum: "Synthetic SI datum".into(),
        points: points
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(cap),
        profile_provenance: "Frozen original-coordinate acceptance survey".into(),
    }
}

fn strict_config() -> PrismaticRiemannConfig {
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: 9.81,
        recovery: PrismaticRecoveryConfig {
            depth_atol_m: 1e-12,
            depth_rtol: 1e-12,
        },
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        root: CertifiedBisectionOptions {
            atol: 1e-15,
            rtol: 1e-15,
            max_iterations: 256,
            max_points: 1 << 20,
        },
        max_bracket_expansions: 4096,
        initial_simpson_panels: 4,
        max_simpson_panels: 4096,
        max_refinements: 10,
        max_native_query_points: 1 << 24,
    };
    assert_eq!(config.gravity_m_per_s2, 9.81);
    assert_eq!(config.water_flux_atol_m3_per_s, 1e-12);
    assert_eq!(config.momentum_flux_atol_m4_per_s2, 1e-12);
    assert_eq!(config.flux_rtol, 1e-12);
    assert_eq!(config.root.atol, 1e-15);
    assert_eq!(config.root.rtol, 1e-15);
    assert_eq!(config.root.max_iterations, 256);
    assert_eq!(config.max_bracket_expansions, 4096);
    config
}

const ORIGINAL_WITNESS: [u64; 4] = [
    0x3b78f9be50d9e3c0,
    0x3b9f708d92b31467,
    0x3ac76bb234c27232,
    0x3aed77d30479b7cc,
];

const DONOR_WITNESS: [u64; 4] = [
    0x399777f21e1a835c,
    0x39bd89ee2425974f,
    0x38e0225a9152b801,
    0x39044cd69a2d92bf,
];

const TINY_SHOCK_WITNESS: [u64; 4] = [
    0x1a70000000000000,
    0x1a94000000000000,
    0x1a70000000000000,
    0x1a90000000000000,
];

fn check_work(work: PrismaticRiemannWork, config: &PrismaticRiemannConfig) -> TestResult {
    let total = work
        .residual_points
        .checked_add(work.simpson_value_points)
        .and_then(|points| points.checked_add(work.simpson_derivative_points))
        .ok_or("Executed work counter sum overflows usize")?;
    assert!(total <= config.max_native_query_points);
    assert!(work.refinements <= config.max_refinements);
    assert!(work.maximum_simpson_panels <= config.max_simpson_panels);
    assert!(work.maximum_root_iterations <= config.root.max_iterations);
    assert!(work.root_iterations <= work.root_solves * config.root.max_iterations);
    assert!(work.maximum_bracket_expansion_rounds <= config.max_bracket_expansions);
    assert!(work.maximum_bracket_expansion_rounds <= work.bracket_expansion_rounds);
    Ok(())
}

fn rightward_trace<R>(client: R::Client, witness: [u64; 4]) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: support::Client<R>,
{
    let case = PROFILES
        .iter()
        .find(|case| case.id == "three_bands")
        .ok_or("Missing original multiband dry-fan profile")?;
    let profile = profile("both_knots_dry_fan", case.points, case.cap);
    let batch = CrossSectionBatch::<R>::new(client.clone(), vec![profile])?.certify_prismatic()?;
    let config = strict_config();
    let [left_area, left_discharge, right_area, right_discharge] = witness.map(f64::from_bits);
    for strided in [false, true] {
        let owners = values::<R, _>(&client, &[0_i64; 3], strided)?;
        let input = |value| values::<R, _>(&client, &[value; 3], strided);
        let result = batch.riemann_flux_tensor(
            &owners,
            &input(left_area)?,
            &input(left_discharge)?,
            &input(right_area)?,
            &input(right_discharge)?,
            &config,
        )?;
        check_work(result.work(), &config)?;
        assert!(result.work().bracket_solves > 0);
        if witness == DONOR_WITNESS {
            assert!(result.work().bracket_expansion_rounds > 0);
            assert!(result.work().maximum_bracket_expansion_rounds > 0);
        }
        assert_eq!(result.owners().try_to_vec::<i64>()?, [0_i64; 3]);
        for (snapshot, original) in [
            (result.left_area_m2(), left_area),
            (result.left_discharge_m3_per_s(), left_discharge),
            (result.right_area_m2(), right_area),
            (result.right_discharge_m3_per_s(), right_discharge),
        ] {
            assert!(
                snapshot
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|value| value.to_bits() == original.to_bits())
            );
        }
        for interval in [
            result.water_flux_m3_per_s(),
            result.momentum_flux_m4_per_s2(),
            result.trace_area_m2(),
            result.trace_discharge_m3_per_s(),
            result.trace_depth_m(),
            result.wave_speed_m_per_s(),
        ] {
            finite(interval)?;
        }
        // The public wave envelope includes its zero initialization and conservatively bounds rightward signals.
        assert!(
            result
                .wave_speed_m_per_s()
                .lower()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&value| value >= 0.0)
        );
        assert!(
            result
                .wave_speed_m_per_s()
                .upper()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&value| value > 0.0)
        );
        // Strictly rightward waves select the original left conserved trace at the stationary interface.
        for (actual, original) in [
            (result.trace_area_m2(), left_area),
            (result.trace_discharge_m3_per_s(), left_discharge),
            (result.water_flux_m3_per_s(), left_discharge),
        ] {
            for endpoint in [actual.lower(), actual.upper()] {
                assert!(
                    endpoint
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|value| value.to_bits() == original.to_bits())
                );
            }
        }
        for selected_discharge in result.water_flux_m3_per_s().upper().try_to_vec::<f64>()? {
            assert!(2.0 * left_area / selected_discharge > 0.002);
        }
    }
    Ok(())
}

#[test]
fn rightward_shallow_waves_retain_the_original_upwind_trace_cpu() -> TestResult {
    rightward_trace::<numr::runtime::cpu::CpuRuntime>(support::cpu(), ORIGINAL_WITNESS)
}

#[cfg(feature = "cuda")]
#[test]
fn rightward_shallow_waves_retain_the_original_upwind_trace_cuda() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    rightward_trace::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?, ORIGINAL_WITNESS)
}

#[test]
fn captured_shallow_donor_retains_the_original_upwind_trace_cpu() -> TestResult {
    rightward_trace::<numr::runtime::cpu::CpuRuntime>(support::cpu(), DONOR_WITNESS)
}

#[cfg(feature = "cuda")]
#[test]
fn captured_shallow_donor_retains_the_original_upwind_trace_cuda() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    rightward_trace::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?, DONOR_WITNESS)
}

#[test]
fn rightward_tiny_shocks_retain_the_original_upwind_trace_cpu() -> TestResult {
    rightward_trace::<numr::runtime::cpu::CpuRuntime>(support::cpu(), TINY_SHOCK_WITNESS)
}

#[cfg(feature = "cuda")]
#[test]
fn rightward_tiny_shocks_retain_the_original_upwind_trace_cuda() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    rightward_trace::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?, TINY_SHOCK_WITNESS)
}

fn expansion_limit<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: support::Client<R>,
{
    let case = PROFILES
        .iter()
        .find(|case| case.id == "three_bands")
        .ok_or("Missing original multiband dry-fan profile")?;
    let profiles = ["donor-section-0", "donor-section-1"]
        .into_iter()
        .map(|id| profile(id, case.points, case.cap))
        .collect();
    let batch = CrossSectionBatch::<R>::new(client.clone(), profiles)?.certify_prismatic()?;
    let mut config = strict_config();
    config.max_bracket_expansions = 0;
    for strided in [false, true] {
        for (query, owner, owners) in [(1, 1, [0_i64, 1, 0]), (2, 0, [1_i64, 1, 0])] {
            let owners = values::<R, _>(&client, &owners, strided)?;
            let mut input = Vec::new();
            for value in DONOR_WITNESS.map(f64::from_bits) {
                let mut lanes = [0.0; 3];
                lanes[query] = value;
                input.push(values::<R, _>(&client, &lanes, strided)?);
            }
            match batch
                .riemann_flux_tensor(&owners, &input[0], &input[1], &input[2], &input[3], &config)
            {
                Err(PrismaticRiemannError::Rejected {
                    reason,
                    query: original_query,
                    section_owner,
                    work,
                    context,
                }) => {
                    assert_eq!(reason, PrismaticRiemannRejection::WorkLimit, "{context}");
                    assert_eq!(original_query, Some(query), "{context}");
                    assert_eq!(section_owner, Some(owner), "{context}");
                    assert_eq!(work.bracket_expansion_rounds, 0);
                    assert_eq!(work.maximum_bracket_expansion_rounds, 0);
                    assert_eq!(work.bracket_solves, 1);
                    assert_eq!(work.root_solves, 0);
                    assert_eq!(work.root_iterations, 0);
                    assert!(work.residual_points > 0);
                    check_work(*work, &config)?;
                }
                Err(error) => return Err(format!("Expansion ceiling returned {error}").into()),
                Ok(_) => return Err("Zero expansion ceiling accepted the captured donor".into()),
            }
        }
    }
    Ok(())
}

#[test]
fn captured_donor_expansion_limit_retains_original_lane_identity_cpu() -> TestResult {
    expansion_limit::<numr::runtime::cpu::CpuRuntime>(support::cpu())
}

#[cfg(feature = "cuda")]
#[test]
fn captured_donor_expansion_limit_retains_original_lane_identity_cuda() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    expansion_limit::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

fn shared_residual_budget<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: support::Client<R>,
{
    let case = PROFILES
        .iter()
        .find(|case| case.id == "three_bands")
        .ok_or("Missing original multiband dry-fan profile")?;
    let profile = profile("both_knots_dry_fan", case.points, case.cap);
    let batch = CrossSectionBatch::<R>::new(client.clone(), vec![profile])?.certify_prismatic()?;
    let mut config = strict_config();
    config.max_bracket_expansions = 0;
    config.root.max_points = 2;
    for strided in [false, true] {
        let owners = values::<R, _>(&client, &[0_i64; 3], strided)?;
        let input = |value| values::<R, _>(&client, &[0.0, value, 0.0], strided);
        // Equal wet states with separating velocities bracket below their common source depth.
        // Two endpoint queries exhaust the shared budget before bisection re-evaluates them.
        match batch.riemann_flux_tensor(
            &owners,
            &input(1.0)?,
            &input(-0.1)?,
            &input(1.0)?,
            &input(0.1)?,
            &config,
        ) {
            Err(PrismaticRiemannError::Rejected {
                reason,
                work,
                context,
                ..
            }) => {
                assert_eq!(reason, PrismaticRiemannRejection::WorkLimit, "{context}");
                assert!(context.contains("remaining"), "{context}");
                assert_eq!(work.bracket_solves, 1);
                assert_eq!(work.root_solves, 0);
                assert_eq!(work.residual_points, 2);
                assert_eq!(work.root_iterations, 0);
                assert_eq!(work.bracket_expansion_rounds, 0);
                assert_eq!(work.maximum_bracket_expansion_rounds, 0);
                check_work(*work, &config)?;
            }
            Err(error) => return Err(format!("Shared residual ceiling returned {error}").into()),
            Ok(_) => return Err("Bisection accepted an exhausted shared residual budget".into()),
        }
    }
    Ok(())
}

#[test]
fn star_bracket_and_bisection_share_the_residual_budget_cpu() -> TestResult {
    shared_residual_budget::<numr::runtime::cpu::CpuRuntime>(support::cpu())
}

#[cfg(feature = "cuda")]
#[test]
fn star_bracket_and_bisection_share_the_residual_budget_cuda() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    shared_residual_budget::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
