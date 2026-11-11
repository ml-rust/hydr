use super::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    CrossSectionRecoveryConfig,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn native_recovery<R: Runtime<DType = DType>>(client: R::Client) -> TestResult
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let definitions = ["first", "second"]
        .into_iter()
        .map(|id| CrossSectionDefinition {
            id: id.into(),
            vertical_datum: "Analytical section datum".into(),
            points: [(0.0, 4.0), (0.0, 0.0), (2.0, 0.0), (2.0, 4.0)]
                .into_iter()
                .map(|(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect(),
            maximum_stage_m: 4.0,
            profile_provenance: "Explicit two-metre rectangular geometry".into(),
        })
        .collect();
    let batch = CrossSectionBatch::<R>::new(client, definitions)?;
    let context = Arc::clone(&batch.context);
    let config = CrossSectionRecoveryConfig::default();
    let strided = |values: [f64; 2]| -> Result<Tensor<R>, numr::error::Error> {
        Ok(Tensor::from_slice(
            &[f64::NAN, values[0], f64::INFINITY, values[1]],
            &[2, 2],
            context.client.device(),
        )?
        .narrow(1, 1, 1)?
        .squeeze(Some(1)))
    };
    let original = strided([-0.0, 2.0])?;
    assert!(!original.is_contiguous());
    let recovery = batch.recover_stages_tensor(&original, &config)?;
    assert!(Arc::ptr_eq(&context, &recovery.geometry.context));
    assert!(!recovery.requested_area.is_contiguous());
    let retained = recovery.requested_area.contiguous()?.try_to_vec::<f64>()?;
    assert_eq!(retained[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(retained[1].to_bits(), 2.0_f64.to_bits());
    let properties = recovery.properties()?;
    assert_eq!(
        properties[0].requested_area_m2.to_bits(),
        (-0.0_f64).to_bits()
    );
    assert_eq!(
        properties[0].geometry.query_stage_m.to_bits(),
        0.0_f64.to_bits()
    );
    assert!(properties[0].geometry.dry);
    assert_eq!(properties[1].geometry.query_stage_m, 1.0);
    assert_eq!(properties[1].geometry.area_m2, 2.0);
    let wrong_rank = Tensor::<R>::from_slice(&[0.0_f64, 2.0], &[1, 2], context.client.device())?;
    let wrong_dtype = Tensor::<R>::from_slice(&[0.0_f32, 2.0], &[2], context.client.device())?;
    for areas in [&wrong_rank, &wrong_dtype] {
        let error = batch
            .recover_stages_tensor(areas, &config)
            .err()
            .ok_or("Invalid native area layout was accepted")?;
        assert!(matches!(
            error,
            CrossSectionError::InvalidInput { section: None, .. }
        ));
    }
    for (values, expected) in [
        ([0.0, f64::NAN], 1),
        ([f64::NAN, f64::INFINITY], 0),
        ([9.0, f64::NAN], 1),
        ([-1.0, f64::NAN], 1),
    ] {
        let error = batch
            .recover_stages_tensor(&strided(values)?, &config)
            .err()
            .ok_or("Nonfinite native area was accepted")?;
        assert!(
            matches!(error, CrossSectionError::InvalidInput { section: Some(index), .. } if index == expected)
        );
    }
    assert_eq!(recovery.properties()?, properties);
    assert_eq!(
        batch.recover_stages(&[0.0, 2.0], &config)?.properties()?[1]
            .geometry
            .area_m2,
        2.0
    );
    Ok(())
}

#[test]
fn cpu_strided_recovery_retains_operands_context_and_admission() -> TestResult {
    native_recovery::<CpuRuntime>(
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None)),
    )
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_strided_recovery_retains_operands_context_and_admission() -> TestResult {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    native_recovery::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
