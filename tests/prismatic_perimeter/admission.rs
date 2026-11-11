use super::support::{self, Client, Result};
use hydr::channel::{
    CrossSectionBatch, CrossSectionDefinition, CrossSectionError, CrossSectionPoint,
    PRISMATIC_BAND_WORK_LIMIT, PrismaticPerimeterBatch, PrismaticRecoveryConfig,
};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

fn rejected<T>(result: std::result::Result<T, CrossSectionError>, label: &str) -> Result {
    let error = result.err().ok_or_else(|| format!("Accepted {label}"))?;
    assert!(
        matches!(error, CrossSectionError::InvalidInput { .. }),
        "{label}: {error}"
    );
    Ok(())
}

fn both<R>(
    batch: &PrismaticPerimeterBatch<R>,
    owners: &Tensor<R>,
    values: &Tensor<R>,
    label: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    rejected(batch.evaluate_depths_tensor(owners, values), label)?;
    rejected(
        batch.recover_depths_tensor(owners, values, &PrismaticRecoveryConfig::default()),
        label,
    )
}

fn metadata<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let owner = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?;
    let value = Tensor::<R>::from_slice(&[1.0f64], &[1], client.device())?;
    let owner_scalar = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
    let value_scalar = Tensor::<R>::from_slice(&[1.0f64], &[], client.device())?;
    let owner_matrix = Tensor::<R>::from_slice(&[0i64], &[1, 1], client.device())?;
    let value_matrix = Tensor::<R>::from_slice(&[1.0f64], &[1, 1], client.device())?;
    let owners_two = Tensor::<R>::from_slice(&[0i64, 0], &[2], client.device())?;
    let values_two = Tensor::<R>::from_slice(&[1.0f64, 1.0], &[2], client.device())?;
    let owner_float = Tensor::<R>::from_slice(&[0.0f64], &[1], client.device())?;
    let value_f32 = Tensor::<R>::from_slice(&[1.0f32], &[1], client.device())?;
    let empty_owner = Tensor::<R>::empty(&[0], DType::I64, client.device())?;
    let empty_value = Tensor::<R>::empty(&[0], DType::F64, client.device())?;
    let empty_owner_float = Tensor::<R>::empty(&[0], DType::F64, client.device())?;
    let empty_value_f32 = Tensor::<R>::empty(&[0], DType::F32, client.device())?;
    let empty_owner_matrix = Tensor::<R>::empty(&[0, 1], DType::I64, client.device())?;
    let empty_value_matrix = Tensor::<R>::empty(&[0, 1], DType::F64, client.device())?;
    for (label, owners, values) in [
        ("scalar owners", &owner_scalar, &value),
        ("scalar values", &owner, &value_scalar),
        ("matrix owners", &owner_matrix, &value),
        ("matrix values", &owner, &value_matrix),
        ("both matrices", &owner_matrix, &value_matrix),
        ("extra owners", &owners_two, &value),
        ("extra values", &owner, &values_two),
        ("floating owners", &owner_float, &value),
        ("F32 values", &owner, &value_f32),
        ("empty floating owners", &empty_owner_float, &empty_value),
        ("empty F32 values", &empty_owner, &empty_value_f32),
        ("empty matrix owners", &empty_owner_matrix, &empty_value),
        ("empty matrix values", &empty_owner, &empty_value_matrix),
    ] {
        both(&batch, owners, values, label)?;
    }
    let query = batch.evaluate_depths_tensor(&empty_owner, &empty_value)?;
    assert!(query.properties()?.is_empty());
    support::snapshots(
        query.geometry().owners(),
        query.geometry().requested_depths_m(),
        query.geometry().dry_mask(),
        &[],
        &[],
    )?;
    for interval in [
        query.wetted_perimeter_m(),
        query.hydraulic_radius_m(),
        query.geometry().depth_m(),
        query.geometry().area_m2(),
    ] {
        support::native(interval, 0);
    }
    let recovered = batch.recover_depths_tensor(
        &empty_owner,
        &empty_value,
        &PrismaticRecoveryConfig::default(),
    )?;
    assert!(recovered.properties()?.is_empty());
    support::snapshots(
        recovered.geometry().owners(),
        recovered.geometry().requested_areas_m2(),
        recovered.geometry().dry_mask(),
        &[],
        &[],
    )?;
    for interval in [
        recovered.wetted_perimeter_m(),
        recovered.hydraulic_radius_m(),
        recovered.geometry().depth_m(),
        recovered.geometry().forward_area_m2(),
    ] {
        support::native(interval, 0);
    }
    Ok(())
}

fn ranges_and_work<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let batch = support::batch::<R>(client)?;
    let definitions = batch.definitions().to_vec();
    for strided in [false, true] {
        for owner in [-1, 9, i64::MAX] {
            let (owners, values) =
                support::inputs::<R>(client, &[0, owner, 0], &[1.0; 3], strided)?;
            both(&batch, &owners, &values, "owner range")?;
        }
        for value in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
            let (owners, values) =
                support::inputs::<R>(client, &[0, 0, 0], &[1.0, value, 1.0], strided)?;
            both(&batch, &owners, &values, "nonphysical or over-cap operand")?;
        }
    }
    let (owners, values) = support::inputs::<R>(client, &[0], &[1.0], false)?;
    for count in [
        PRISMATIC_BAND_WORK_LIMIT / batch.maximum_band_count() + 1,
        usize::MAX / 8 + 1,
        usize::MAX / batch.maximum_band_count() + 1,
    ] {
        // Broadcast views retain one valid allocation, including overflow-sized logical shapes.
        let owners = owners.broadcast_to(&[count])?;
        let values = values.broadcast_to(&[count])?;
        assert_eq!(owners.storage().len(), 1);
        assert_eq!(values.storage().len(), 1);
        both(&batch, &owners, &values, "padded work or byte overflow")?;
    }
    for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for config in [
            PrismaticRecoveryConfig {
                depth_atol_m: invalid,
                depth_rtol: 0.0,
            },
            PrismaticRecoveryConfig {
                depth_atol_m: 0.0,
                depth_rtol: invalid,
            },
        ] {
            rejected(
                batch.recover_depths_tensor(&owners, &values, &config),
                "recovery width config",
            )?;
        }
    }
    assert_eq!(batch.definitions(), definitions.as_slice());
    assert_eq!(
        batch
            .evaluate_depths_tensor(&owners, &values)?
            .properties()?
            .len(),
        1
    );
    Ok(())
}

fn original_bank_range_isolation<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let definition = CrossSectionDefinition {
        id: "shallow-cap-with-tall-original-banks".into(),
        vertical_datum: "Original bank range admission datum".into(),
        points: [(-1e150, 1e200), (-1.0, 0.0), (1.0, 0.0), (1e150, 1e200)]
            .into_iter()
            .map(|(station_m, elevation_m)| CrossSectionPoint {
                station_m,
                elevation_m,
            })
            .collect(),
        maximum_stage_m: 1.0,
        profile_provenance: "Shallow cap with finite original coordinate differences".into(),
    };
    let source = CrossSectionBatch::<R>::new(client.clone(), vec![definition])?;
    let definitions = source.definitions().to_vec();
    let geometry = source.certify_prismatic()?;
    let (owners, depths) = support::inputs::<R>(client, &[0], &[0.5], false)?;
    let retained = geometry.evaluate_depths_tensor(&owners, &depths)?;
    let expected = retained.properties()?;
    let row = expected
        .first()
        .ok_or("Missing shallow capped geometry row")?;
    assert_eq!(row.depth_m, [0.5, 0.5]);
    assert!(!row.dry);
    assert!(row.area_m2[0] > 0.0 && row.area_m2[1].is_finite());

    // The original bank height squares beyond F64, despite finite shallow geometry.
    let error = geometry
        .certify_perimeter()
        .err()
        .ok_or("Perimeter certification accepted an overflowing original bank-height square")?;
    assert!(
        matches!(error, CrossSectionError::Numerical { .. }),
        "{error}"
    );
    assert_eq!(source.definitions(), definitions.as_slice());
    assert_eq!(geometry.definitions(), definitions.as_slice());
    assert_eq!(retained.properties()?, expected);
    assert_eq!(
        geometry
            .evaluate_depths_tensor(&owners, &depths)?
            .properties()?,
        expected
    );
    Ok(())
}

pub(crate) fn run<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    metadata::<R>(&client)?;
    ranges_and_work::<R>(&client)?;
    original_bank_range_isolation::<R>(&client)
}

#[test]
fn cpu_checks_every_operand_before_native_contact_execution() -> Result {
    run::<super::runtime::Runtime>(super::runtime::client())
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_checks_every_operand_before_native_contact_execution() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    run::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[cfg(feature = "cuda")]
#[test]
fn actual_cuda_rejects_each_foreign_device_operand_including_empty() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    let batch = support::batch::<CudaRuntime>(&client)?;
    let foreign = CudaDevice::new(usize::MAX);
    let local_owner = Tensor::<CudaRuntime>::empty(&[0], DType::I64, client.device())?;
    let local_value = Tensor::<CudaRuntime>::empty(&[0], DType::F64, client.device())?;
    let foreign_owner = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &foreign)?;
    let foreign_value = Tensor::<CudaRuntime>::empty(&[0], DType::F64, &foreign)?;
    both(&batch, &foreign_owner, &local_value, "foreign empty owners")?;
    both(&batch, &local_owner, &foreign_value, "foreign empty values")?;
    // Real nonempty foreign operands require a second physical CUDA device.
    let (owners, values) = support::inputs::<CudaRuntime>(&client, &[0], &[1.0], false)?;
    if CudaDevice::count()? > 1 {
        let other = CudaClient::new(CudaDevice::new(1))?;
        let (foreign_owner, foreign_value) =
            support::inputs::<CudaRuntime>(&other, &[0], &[1.0], false)?;
        both(&batch, &foreign_owner, &values, "foreign nonempty owners")?;
        both(&batch, &owners, &foreign_value, "foreign nonempty values")?;
    }
    Ok(())
}

#[cfg(feature = "wgpu")]
#[test]
fn actual_wgpu_rejects_f64_without_runtime_substitution() -> Result {
    use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
    assert!(!WgpuRuntime::computes_float_dtype(DType::F64));
    let client = WgpuClient::new(WgpuDevice::new(0))?;
    let profile = super::fixtures::PROFILES
        .first()
        .ok_or("Missing frozen profile")?;
    assert!(
        CrossSectionBatch::<WgpuRuntime>::new(client, vec![support::definition(*profile)]).is_err()
    );
    Ok(())
}
