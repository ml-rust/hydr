use super::PrismaticRiemannConfig;
use super::potential_difference_test_data::{CASES, GRAVITY, MOMENTUM, PROFILES, WATER};
use super::potential_difference_test_support::{batch, values};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms};

fn flux<R>(
    actual: &Interval<R>,
    reference: [u64; 2],
    reverse: bool,
    label: &str,
) -> Result<(), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
{
    let reference = reference.map(f64::from_bits);
    let reference = if reverse {
        [-reference[1], -reference[0]]
    } else {
        reference
    };
    for (&lower, &upper) in actual
        .lower()
        .try_to_vec::<f64>()?
        .iter()
        .zip(&actual.upper().try_to_vec::<f64>()?)
    {
        assert!(
            lower <= reference[0] && upper >= reference[1],
            "{label}: actual=[{lower},{upper}], reference={reference:?}"
        );
        let allowance = 1e-12 + 1e-12 * lower.abs().max(upper.abs());
        assert!(
            upper - lower <= allowance,
            "{label}: width={}, allowance={allowance}",
            upper - lower
        );
    }
    Ok(())
}

fn references<R>(client: R::Client) -> Result<(), Box<dyn std::error::Error>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + HypotOps<R>
        + MidpointOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    let case = &CASES[0];
    assert_eq!(case.id, "captured_weak_face");
    let batch = batch::<R>(&client, &PROFILES[case.profile])
        .map_err(|error| super::error::into_public(error, Default::default()))?;
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(GRAVITY),
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        initial_simpson_panels: 4,
        max_simpson_panels: 4096,
        ..Default::default()
    };
    config.root.atol = 1e-15;
    config.root.rtol = 1e-15;
    config.root.max_iterations = 256;
    config.root.max_points = 1 << 20;
    for strided in [false, true] {
        for reverse in [false, true] {
            let left = f64::from_bits(0x4024000000000000);
            let right = f64::from_bits(0x4023f45a904bb43e);
            let discharge = f64::from_bits(0x3fac91cb1810b76a);
            let (la, lq, ra, rq) = if reverse {
                (right, -discharge, left, 0.0)
            } else {
                (left, 0.0, right, discharge)
            };
            let input = |value| {
                values::<R, _>(&client, &[value; 3], strided)
                    .map_err(|error| super::error::into_public(error, Default::default()))
            };
            let owners = values::<R, _>(&client, &[0_i64; 3], strided)
                .map_err(|error| super::error::into_public(error, Default::default()))?;
            let result = batch.riemann_flux_tensor(
                &owners,
                &input(la)?,
                &input(lq)?,
                &input(ra)?,
                &input(rq)?,
                &config,
            )?;
            flux(
                result.water_flux_m3_per_s(),
                WATER,
                reverse,
                "captured water flux",
            )?;
            flux(
                result.momentum_flux_m4_per_s2(),
                MOMENTUM,
                false,
                "captured momentum flux",
            )?;
            assert_eq!(result.left_area_m2().try_to_vec::<f64>()?, [la; 3]);
            assert_eq!(result.right_area_m2().try_to_vec::<f64>()?, [ra; 3]);
        }
    }
    Ok(())
}

#[test]
fn captured_weak_face_encloses_independent_flux_cpu() -> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    references::<CpuRuntime>(client)
}

#[cfg(feature = "cuda")]
#[test]
fn captured_weak_face_encloses_independent_flux_cuda() -> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
