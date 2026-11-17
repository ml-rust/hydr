use super::PrismaticRiemannConfig;
use super::error::{HydraulicError, HydraulicResult, checked, into_public};
use super::reflection::tighten;
use super::sample_merge::Samples;
use super::types::CapturedInput;
use crate::channel::section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
use numr::dtype::DType;
use numr::ops::{
    CompareOps, HypotOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::integrate::CertifiedSimpsonAlgorithms;
use solvr::optimize::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms};

fn private_contract<R>(client: &R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], client.device());
    let input = CapturedInput {
        owners: Tensor::from_slice(&[0_i64; 4], &[4], client.device())?,
        left_area: tensor(&[0.125, 0.125, 0.125, 0.0])?,
        right_area: tensor(&[0.125, 0.125_f64.next_up(), 0.125, 0.0])?,
        left_discharge: tensor(&[2.5, 2.5, 2.5, 0.0])?,
        right_discharge: tensor(&[-2.5, -2.5, (-2.5_f64).next_up(), 0.0])?,
    };
    checked::<R, _>(client, "exact reflection row selection", |a| {
        let mut solution = Samples::new(a, 4)?.solution;
        let lower = tensor(&[-1.0; 4])?;
        let upper = tensor(&[2.0; 4])?;
        solution.water_flux = a.bounds(&lower, &upper)?;
        solution.trace_discharge = a.bounds(&lower, &upper)?;
        solution.momentum_flux = a.bounds(&lower, &upper)?;
        solution.trace_area = a.bounds(&lower, &upper)?;
        solution.trace_depth = a.bounds(&lower, &upper)?;
        solution.wave_speed = a.bounds(&lower, &upper)?;
        solution.root_work_limited = true;
        solution.root_work_limited_rows = tensor(&[1.0; 4])?;
        tighten(a, &input, &mut solution)?;
        for value in [&solution.water_flux, &solution.trace_discharge] {
            assert_eq!(value.lower().try_to_vec::<f64>()?, [0.0, -1.0, -1.0, -1.0]);
            assert_eq!(value.upper().try_to_vec::<f64>()?, [0.0, 2.0, 2.0, 2.0]);
        }
        for value in [
            &solution.momentum_flux,
            &solution.trace_area,
            &solution.trace_depth,
            &solution.wave_speed,
        ] {
            assert_eq!(value.lower().try_to_vec::<f64>()?, [-1.0; 4]);
            assert_eq!(value.upper().try_to_vec::<f64>()?, [2.0; 4]);
        }
        assert!(solution.root_work_limited);
        assert_eq!(
            solution.root_work_limited_rows.try_to_vec::<f64>()?,
            [1.0; 4]
        );
        Ok(())
    })?;
    for trace in [false, true] {
        for (lower, upper) in [(1.0, 2.0), (-2.0, -1.0)] {
            let rejected = checked::<R, _>(
                client,
                "reflection contradiction rejects before intersection",
                |a| {
                    let mut solution = Samples::new(a, 4)?.solution;
                    let lower = tensor(&[lower, -1.0, -1.0, -1.0])?;
                    let upper = tensor(&[upper, 2.0, 2.0, 2.0])?;
                    let contradictory = a.bounds(&lower, &upper)?;
                    if trace {
                        solution.trace_discharge = contradictory;
                    } else {
                        solution.water_flux = contradictory;
                    }
                    let result = tighten(a, &input, &mut solution);
                    let retained = if trace {
                        &solution.trace_discharge
                    } else {
                        &solution.water_flux
                    };
                    assert_eq!(
                        retained.lower().try_to_vec::<f64>()?,
                        lower.try_to_vec::<f64>()?
                    );
                    assert_eq!(
                        retained.upper().try_to_vec::<f64>()?,
                        upper.try_to_vec::<f64>()?
                    );
                    result
                },
            );
            assert!(matches!(
                rejected,
                Err(HydraulicError::Rejected {
                    reason: super::PrismaticRiemannRejection::SamplingUnresolved,
                    ..
                })
            ));
        }
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
    private_contract::<R>(&client).map_err(|error| into_public(error, Default::default()))?;
    let definition = CrossSectionDefinition {
        id: "integer-trapezoid".into(),
        vertical_datum: "Local survey datum".into(),
        points: [(-5.0, 4.0), (-1.0, 0.0), (1.0, 0.0), (5.0, 4.0)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 4.0,
        profile_provenance: "Original integer trapezoid source acceptance".into(),
    };
    let batch =
        CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
    let mut config = PrismaticRiemannConfig {
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        ..Default::default()
    };
    config.root.atol = 1e-15;
    config.root.rtol = 1e-15;
    config.root.max_iterations = 256;
    config.root.max_points = 1 << 20;
    let left_area = [0.125, 10.0, 0.125, 0.0, 10.0, 10.0];
    let right_area = left_area;
    let left_discharge = [2.5, -0.1, -2.5, 0.0, -0.0, 0.2];
    let right_discharge = [-2.5, 0.1, 2.5, 0.0, 0.0, 0.2];
    for strided in [false, true] {
        let values = |input: &[f64]| -> Result<Tensor<R>, numr::error::Error> {
            if strided {
                let storage: Vec<f64> = input.iter().flat_map(|&value| [f64::NAN, value]).collect();
                Ok(
                    Tensor::from_slice(&storage, &[input.len(), 2], client.device())?
                        .narrow(1, 1, 1)?
                        .squeeze(Some(1)),
                )
            } else {
                Tensor::from_slice(input, &[input.len()], client.device())
            }
        };
        let owner_storage = Tensor::from_slice(&[0_i64; 12], &[6, 2], client.device())?;
        let owners = if strided {
            owner_storage.narrow(1, 1, 1)?.squeeze(Some(1))
        } else {
            Tensor::from_slice(&[0_i64; 6], &[6], client.device())?
        };
        let result = batch.riemann_flux_tensor(
            &owners,
            &values(&left_area)?,
            &values(&left_discharge)?,
            &values(&right_area)?,
            &values(&right_discharge)?,
            &config,
        )?;
        for value in [
            result.water_flux_m3_per_s(),
            result.trace_discharge_m3_per_s(),
        ] {
            let lower = value.lower().try_to_vec::<f64>()?;
            let upper = value.upper().try_to_vec::<f64>()?;
            for index in 0..5 {
                assert_eq!(
                    lower[index], 0.0,
                    "reflected row={index}, strided={strided}"
                );
                assert_eq!(
                    upper[index], 0.0,
                    "reflected row={index}, strided={strided}"
                );
            }
            assert!(lower[5] <= 0.2 && upper[5] >= 0.2);
            assert!(lower[5] > 0.0);
        }
        assert_eq!(result.left_area_m2().try_to_vec::<f64>()?, left_area);
        assert_eq!(result.right_area_m2().try_to_vec::<f64>()?, right_area);
    }
    Ok(())
}

#[test]
fn exact_reflection_preserves_zero_discharge_and_other_rows_cpu()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    references::<CpuRuntime>(client)
}

#[cfg(feature = "cuda")]
#[test]
fn exact_reflection_preserves_zero_discharge_and_other_rows_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
