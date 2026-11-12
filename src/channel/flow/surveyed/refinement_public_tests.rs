use super::super::{PrismaticRiemannConfig, PrismaticRiemannError, PrismaticRiemannRejection};
use crate::channel::section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms};

fn endpoints<R: Runtime<DType = DType>>(
    value: &Interval<R>,
    row: usize,
) -> Result<[f64; 2], numr::error::Error> {
    Ok([
        value.lower().narrow(0, row, 1)?.item::<f64>()?,
        value.upper().narrow(0, row, 1)?.item::<f64>()?,
    ])
}

fn contains(actual: [f64; 2], bits: [u64; 2]) {
    let truth = bits.map(f64::from_bits);
    assert!(
        actual[0] <= truth[0] && actual[1] >= truth[1],
        "actual={actual:?}, frozen={truth:?}"
    );
}

fn mixed<R>(client: R::Client) -> Result<(), Box<dyn std::error::Error>>
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
    let definition = CrossSectionDefinition {
        id: "trapezoid".into(),
        vertical_datum: "Frozen local survey datum".into(),
        points: [(0.0, 4.0), (4.0, 0.0), (5.0, 0.0), (9.0, 4.0)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 4.0,
        profile_provenance: "Original frozen trapezoid Riemann references".into(),
    };
    let definitions = (0..3)
        .map(|index| {
            let mut value = definition.clone();
            value.id = format!("trapezoid-{index}");
            value
        })
        .collect();
    let batch = CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?;
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: 1.0,
        ..Default::default()
    };
    config.root.atol = 1e-15;
    config.root.rtol = 1e-15;
    let owners = [2_i64, 1, 2, 1];
    let left = [6.0, 2.0, 6.0, 2.0];
    let right = [0.5, 2.0, 0.5, 2.0];
    let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], client.device());
    let q = tensor(&[0.0; 4])?;
    let result = batch.riemann_flux_tensor(
        &Tensor::from_slice(&owners, &[4], client.device())?,
        &tensor(&left)?,
        &q,
        &tensor(&right)?,
        &q,
        &config,
    )?;
    assert_eq!(result.owners().try_to_vec::<i64>()?, owners);
    assert_eq!(result.left_area_m2().try_to_vec::<f64>()?, left);
    assert_eq!(result.right_area_m2().try_to_vec::<f64>()?, right);
    assert_eq!(
        result.left_discharge_m3_per_s().try_to_vec::<f64>()?,
        [0.0; 4]
    );
    assert_eq!(
        result.right_discharge_m3_per_s().try_to_vec::<f64>()?,
        [0.0; 4]
    );
    let mut independent_values = 0;
    let mut independent_derivatives = 0;
    let mut easy_panels = 0;
    let mut strong_panels = 0;
    let mut easy_values = 0;
    let mut easy_derivatives = 0;
    for row in 0..4 {
        let zero = tensor(&[0.0])?;
        let singleton = batch.riemann_flux_tensor(
            &Tensor::from_slice(&[owners[row]], &[1], client.device())?,
            &tensor(&[left[row]])?,
            &zero,
            &tensor(&[right[row]])?,
            &zero,
            &config,
        )?;
        for (mixed, single) in [
            (
                result.water_flux_m3_per_s(),
                singleton.water_flux_m3_per_s(),
            ),
            (
                result.momentum_flux_m4_per_s2(),
                singleton.momentum_flux_m4_per_s2(),
            ),
            (result.trace_area_m2(), singleton.trace_area_m2()),
            (
                result.trace_discharge_m3_per_s(),
                singleton.trace_discharge_m3_per_s(),
            ),
            (result.trace_depth_m(), singleton.trace_depth_m()),
            (result.wave_speed_m_per_s(), singleton.wave_speed_m_per_s()),
        ] {
            assert_eq!(
                endpoints(mixed, row)?,
                endpoints(single, 0)?,
                "scattered row={row}"
            );
        }
        for (mixed, single) in [
            (result.trace_dry_possible(), singleton.trace_dry_possible()),
            (result.trace_wet_possible(), singleton.trace_wet_possible()),
        ] {
            assert_eq!(
                mixed.narrow(0, row, 1)?.item::<f64>()?,
                single.item::<f64>()?
            );
        }
        let work = singleton.work();
        independent_values += work.simpson_value_points;
        independent_derivatives += work.simpson_derivative_points;
        if row % 2 == 0 {
            strong_panels = work.maximum_simpson_panels;
            contains(
                endpoints(result.water_flux_m3_per_s(), row)?,
                [0x4001218d797712e1, 0x4001218d797712e2],
            );
            contains(
                endpoints(result.momentum_flux_m4_per_s2(), row)?,
                [0x40082ebacda719ac, 0x40082ebacda719ad],
            );
            contains(
                endpoints(result.trace_area_m2(), row)?,
                [0x4003c9f122462c1a, 0x4003c9f122462c1b],
            );
            contains(
                endpoints(result.trace_depth_m(), row)?,
                [0x3ff267c54e60a775, 0x3ff267c54e60a776],
            );
        } else {
            easy_panels = work.maximum_simpson_panels;
            easy_values += work.simpson_value_points;
            easy_derivatives += work.simpson_derivative_points;
            assert_eq!(endpoints(result.water_flux_m3_per_s(), row)?, [0.0; 2]);
            contains(
                endpoints(result.momentum_flux_m4_per_s2(), row)?,
                [0x3feaaaaaaaaaaaaa, 0x3feaaaaaaaaaaaab],
            );
        }
    }
    assert!(
        strong_panels > easy_panels,
        "strong panels={strong_panels}, easy={easy_panels}"
    );
    assert!(result.work().simpson_value_points <= independent_values);
    assert!(result.work().simpson_derivative_points <= independent_derivatives);
    assert!(easy_values > 0 && easy_derivatives > 0);
    // Global refinement repeats each nonanalytic easy row at least once beyond its singleton acceptance.
    assert!(result.work().simpson_value_points < independent_values + easy_values);
    assert!(result.work().simpson_derivative_points < independent_derivatives + easy_derivatives);
    let mut limited = config;
    limited.max_refinements = 0;
    limited.max_simpson_panels = limited.initial_simpson_panels;
    let failed = batch.riemann_flux_tensor(
        &Tensor::from_slice(&[1_i64, 2, 1], &[3], client.device())?,
        &tensor(&[2.0, 6.0, 2.0])?,
        &tensor(&[0.0; 3])?,
        &tensor(&[2.0, 0.5, 2.0])?,
        &tensor(&[0.0; 3])?,
        &limited,
    );
    assert!(matches!(
        failed,
        Err(PrismaticRiemannError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            query: Some(1),
            section_owner: Some(2),
            ..
        })
    ));
    Ok(())
}

#[test]
fn mixed_faces_retain_only_unresolved_native_work_cpu() -> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    mixed::<CpuRuntime>(client)
}

#[cfg(feature = "cuda")]
#[test]
fn mixed_faces_retain_only_unresolved_native_work_cuda() -> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    mixed::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
