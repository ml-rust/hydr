use super::PrismaticRiemannConfig;
use super::error::{HydraulicResult, PrismaticRiemannError, checked, into_public};
use super::integration::potential;
use super::potential_difference::direct_increment;
use super::potential_difference_test_data::{CASES, GRAVITY, PROFILES};
use super::potential_difference_test_support::{batch, contains, target, values};
use super::work::WorkLedger;
use crate::channel::section::{GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;

fn fallback<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    original: &Tensor<R>,
    target: &Interval<R>,
    config: &PrismaticRiemannConfig,
    work: &WorkLedger,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = geometry.client();
    let mut endpoints = Vec::new();
    for area in [original, target.lower(), target.upper()] {
        let state = checked(c, "fallback endpoint recovery", |a| {
            // Public entry points capture contiguous conserved operands before private recovery.
            let captured = a.point(area)?;
            geometry.recover(a, bands, captured.lower(), &config.recovery)
        })?;
        endpoints.push(potential(geometry, bands, &state, config, 4096, work)?);
    }
    checked(c, "complete endpoint potential subtraction", |a| {
        let full = a.bounds(endpoints[1].lower(), endpoints[2].upper())?;
        Ok(a.sub(&full, &endpoints[0])?)
    })
}

fn references<R>(client: R::Client) -> HydraulicResult<()>
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
        + CertifiedSimpsonAlgorithms<R>,
{
    assert_eq!(CASES.len(), 18);
    let config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(GRAVITY),
        ..Default::default()
    };
    for strided in [false, true] {
        for case in CASES {
            let batch = batch::<R>(&client, &PROFILES[case.profile])?;
            let geometry = HydraulicGeometry::from_batch(&batch);
            let owners = values::<R, _>(&client, &[0_i64; 3], strided)?;
            let bands = checked(&client, "repeated original section owners", |a| {
                geometry.gather(a, &owners)
            })?;
            let original = values::<R, _>(&client, &[f64::from_bits(case.original); 3], strided)?;
            let target = target::<R>(&client, case.target, strided)?;
            let work = WorkLedger::new(&config);
            let direct =
                direct_increment(&geometry, &bands, &original, &target, &config, 4096, &work)?;
            let available = direct.available.try_to_vec::<f64>()?;
            let required = case.gate != "containment_only";
            if required {
                assert_eq!(
                    available, [1.0; 3],
                    "{} direct admission, strided={strided}",
                    case.id
                );
            }
            if matches!(
                case.id,
                "knot_overlap"
                    | "knot_crossing"
                    | "dry_target"
                    | "dry_overlap"
                    | "wide_positive_span"
            ) {
                assert_eq!(available, [0.0; 3], "{} fallback admission", case.id);
                assert_eq!(
                    work.snapshot().simpson_value_points,
                    0,
                    "{} inactive callbacks",
                    case.id
                );
                assert_eq!(
                    work.snapshot().simpson_derivative_points,
                    0,
                    "{} inactive derivatives",
                    case.id
                );
            }
            let result = if available.iter().all(|&value| value == 1.0) {
                direct.bounds
            } else {
                fallback(&geometry, &bands, &original, &target, &config, &work)?
            };
            contains(
                &result,
                case.reference,
                case.id,
                case.gate == "common_band_width",
            )?;
            for endpoint in case.endpoints {
                contains(&result, endpoint, case.id, false)?;
            }
            if case.gate == "exact_zero" {
                assert_eq!(result.lower().try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(result.upper().try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(work.snapshot().simpson_value_points, 0);
                assert_eq!(work.snapshot().simpson_derivative_points, 0);
            }
            if case.id == "straddling_original" {
                assert!(
                    result
                        .lower()
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&value| value < 0.0)
                );
                assert!(
                    result
                        .upper()
                        .try_to_vec::<f64>()?
                        .iter()
                        .all(|&value| value > 0.0)
                );
            }
        }
    }
    Ok(())
}

#[test]
fn signed_potential_increments_enclose_independent_references_cpu()
-> Result<(), PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    references::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
}

#[cfg(feature = "cuda")]
#[test]
fn signed_potential_increments_enclose_independent_references_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}
